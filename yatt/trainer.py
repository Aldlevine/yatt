import abc as _abc
import os as _os
import time as _time
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Generic as _Generic
from typing import NamedTuple as _NamedTuple
from typing import TypeVar as _TypeVar

import torch as _torch
import yaml as _yaml
from fyst import Cel as _Cel
from fyst import Row as _Row
from fyst import Table as _Table
from IPython.display import clear_output as _clear_output
from torch import Tensor as _Tensor
from torch import nn as _nn
from torch.utils import data as _data
from tqdm import tqdm as _tqdm
from tqdm.auto import tqdm as _tqdm_auto
from tqdm.notebook import tqdm_notebook as _tqdm_notebook

from .summary_writer import SummaryWriter as _SummaryWriter

_IS_NOTEBOOK = issubclass(_tqdm_auto, _tqdm_notebook)


@_dataclass
class DataLoaderConfig:
    train: _data.DataLoader[_Any] | None = None
    val: _data.DataLoader[_Any] | None = None
    test: _data.DataLoader[_Any] | None = None


HParams = _NamedTuple
_HParamType = _TypeVar("_HParamType", bound=HParams)


@_dataclass
class _CheckpointSaveData(_Generic[_HParamType]):
    # user inputs
    hparams: _HParamType

    # training info
    epoch: int
    loss: float

    # training states
    model_state: dict[_Any, _Any]
    optim_state: dict[_Any, _Any]


@_dataclass
class _ModelStats:
    param_count: int
    param_size: float
    buffer_count: int
    buffer_size: float


@_dataclass
class _CheckpointStats:
    file: str
    epoch: int
    loss: float


def _sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class Trainer(_abc.ABC, _Generic[_HParamType]):

    def __init__(
        self,
        exp_name: str,
        device: _torch.device = (_torch.device("cuda")
                                 if _torch.cuda.is_available() else
                                 _torch.device("cpu")),
        max_epochs: int = 500,
        log_interval: int = 50,
        save_best_count: int = 1,
    ) -> None:
        super().__init__()
        self.exp_name = exp_name
        self.device = device
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.save_best_count = save_best_count

        self.logdir = _time.strftime(f"runs/{self.exp_name}/%Y-%m-%d@%T")
        self.ckptdir = _os.path.join(self.logdir, "checkpoints")
        self.writer = _SummaryWriter(self.logdir)
        self.pbar: _tqdm = _tqdm_auto(bar_format="", leave=False, position=0)
        self.reserved_lines = 0

    @_abc.abstractmethod
    def configure_model(self) -> _nn.Module:
        raise NotImplementedError(self.configure_model.__name__)

    @_abc.abstractmethod
    def configure_optimizer(self) -> _torch.optim.Optimizer:
        raise NotImplementedError(self.configure_optimizer.__name__)

    @_abc.abstractmethod
    def configure_data_loaders(self) -> DataLoaderConfig:
        raise NotImplementedError(self.configure_data_loaders.__name__)

    @_abc.abstractmethod
    def train_step(self, batch: list[_Tensor], batch_idx: int) -> _Tensor:
        raise NotImplementedError(self.train_step.__name__)

    @_abc.abstractmethod
    def val_step(self, batch: list[_Tensor], batch_idx: int) -> _Tensor:
        raise NotImplementedError(self.val_step.__name__)

    @_abc.abstractmethod
    def test_step(self, batch: list[_Tensor], batch_idx: int) -> _Tensor:
        raise NotImplementedError(self.test_step.__name__)

    @_abc.abstractmethod
    def train_epoch_end(self) -> None:
        pass

    @_abc.abstractmethod
    def val_epoch_end(self) -> None:
        pass

    def configure(
        self,
        hparams: _HParamType,
        start_epoch: int = 0,
        model_state: dict[_Any, _Any] | None = None,
        optim_state: dict[_Any, _Any] | None = None,
    ) -> None:
        self.hparams = hparams
        self.epoch = start_epoch
        self.model = self.configure_model().to(self.device)
        if model_state:
            self.model.load_state_dict(model_state)
        self.optimizer = self.configure_optimizer()
        if optim_state:
            self.optimizer.load_state_dict(optim_state)  # type: ignore
        # hparams_log = { k:v for k,v in self.hparams._asdict() }
        hparams_log = {}
        for k, v in self.hparams._asdict().items():
            if not isinstance(v, (float, str, bool, _Tensor)):
                try:
                    hparams_log[k] = _torch.tensor(*v)
                except:
                    pass
            else:
                hparams_log[k] = v

        self.writer.add_hparams(hparams_log, {"hp_metric": 0})  # type: ignore

        with open(f"{self.logdir}/hparams.yaml", "w") as file:
            _yaml.dump(self.hparams._asdict(), file)

    def configure_checkpoint(self, checkpoint_path: str) -> None:
        save_data: _CheckpointSaveData = _torch.load(checkpoint_path)
        self.configure(
            save_data.hparams,
            save_data.epoch + 1,
            save_data.model_state,
            save_data.optim_state,
        )
        del save_data

    def train(self) -> None:
        self._display_model_stats()
        self.pbar.close()
        self.pbar: _tqdm = _tqdm_auto(
            bar_format="Configuring Data Loaders...",
            total=0,
            position=0,
            leave=False,
        )
        self.data_loaders = self.configure_data_loaders()
        self._display_model_stats()
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            if self.data_loaders.train != None:
                self._train_loop()
                self.train_epoch_end()
            if self.data_loaders.val != None:
                loss = self._val_loop()
                self._save_best_ckpt(loss)
                self._save_latest_ckpt(loss)
                self._display_model_stats()
                self.val_epoch_end()
        if self.data_loaders.test != None:
            self._test_loop()

    def log_msg(self, msg: str) -> None:
        self.pbar.write(msg)

    def log_scalar(self, tag: str, value: _Tensor, step: int) -> None:
        self.writer.add_scalar(tag, value.cpu(), step, new_style=True)
        self.pbar.set_postfix({
            tag: value.item(),
        })

    def log_image(self, tag: str, image: _Tensor, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_graph(self, input: _Tensor) -> None:
        self.writer.add_graph(self.model, input)

    def _display_lines(self, lines_str: str, start: int = -1) -> None:
        if _IS_NOTEBOOK:
            _clear_output()
            self.log_msg(lines_str)
            return
        lines = lines_str.split("\n")
        nrows = self.pbar.nrows or 0
        ncols = self.pbar.ncols or 0
        rows_needed = min(len(lines) - 1, nrows - 2)
        self.reserved_lines = min(self.reserved_lines, nrows - 2)
        if rows_needed > self.reserved_lines:
            self.pbar.write("\n" * (rows_needed - self.reserved_lines - 1))
        elif self.reserved_lines > rows_needed:
            for i in range(rows_needed, self.reserved_lines):
                if i > self.pbar.nrows - 2:
                    break
                self.pbar.display(" " * ncols, start - (i + 1))
        for i, line in enumerate(reversed(lines)):
            if i > nrows - 2:
                break
            self.pbar.display(" " * ncols, start - i)
            self.pbar.display(line, start - i)
        self.reserved_lines = max(rows_needed, self.reserved_lines)

    def _get_model_stats(self) -> _ModelStats:
        param_count: int = 0
        param_size: int = 0
        for param in self.model.parameters():
            param_count += 1
            param_size += param.nelement() * param.element_size()
        buffer_count: int = 0
        buffer_size: int = 0
        for buffer in self.model.buffers():
            buffer_count += 1
            buffer_size += buffer.nelement() * buffer.element_size()

        return _ModelStats(
            param_count,
            param_size,
            buffer_count,
            buffer_size,
        )

    def _display_model_stats(self) -> None:
        stats = self._get_model_stats()

        hp_data = [
            _Row(k, v, border=(1, 0))
            for k, v in self.hparams._asdict().items()
        ]
        hp_data[-1].border = (1, 0, 1, 1)

        model_stats = {
            "Parameter Count": stats.param_count,
            "Parameter Size": _sizeof_fmt(stats.param_size),
            "Buffer Count": stats.buffer_count,
            "Buffer Size": _sizeof_fmt(stats.buffer_size),
            "Total Size": _sizeof_fmt(stats.param_size + stats.buffer_size),
        }
        model_data = [
            _Row(k, v, border=(1, 0)) for k, v in model_stats.items()
        ]
        model_data[-1].border = (1, 0, 1, 1)

        data = _Table(
            _Row(_Cel(self.model.__class__.__name__, span=(2, 1)),
                 halign="middle"),
            [None],
            [_Cel("HParams", span=(2, 1), halign="middle")],
            *hp_data,
            [None],
            [_Cel("Stats", span=(2, 1), halign="middle")],
            *model_data,
            padding=(1, 0),
        )

        ckpts = self._get_best_ckpts()
        self.shown_count = (self.shown_count
                            if hasattr(self, "shown_count") else 0)
        self.shown_count += 1

        if len(ckpts) > 0 or self.shown_count > 10:
            ckpt_stats = {}
            for ckpt in ckpts:
                ckpt_stats[ckpt.epoch] = ckpt.loss
            ckpt_data = [
                _Row(k, v, border=(1, 0)) for k, v in ckpt_stats.items()
            ]
            ckpt_data[-1].border = (1, 0, 1, 1)
            data.extend([
                _Row(None),
                _Row(
                    _Cel(
                        "Best Checkpoints",
                        span=(2, 1),
                        border=(1, 1, 1, 0),
                        padding=(0, 0, 0, 1),
                        halign="middle",
                    )),
                _Row(
                    _Cel("Epoch", border=(1, 0, 1, 1)),
                    _Cel("Loss", border=(1, 0, 1, 1)),
                    halign="middle",
                    padding=(0, -1, 0, 0),
                ),
                *ckpt_data,
            ])

        self._display_lines(str(data))

    def _save_checkpoint(self, filename: str, loss: float) -> None:
        save_data = _CheckpointSaveData[_HParamType](
            model_state=self.model.state_dict(),
            optim_state=self.optimizer.state_dict(),
            hparams=self.hparams,
            epoch=self.epoch,
            loss=loss,
        )
        _torch.save(save_data, _os.path.join(self.ckptdir, filename))

    def _get_best_ckpts(self) -> list[_CheckpointStats]:
        if not _os.path.exists(self.ckptdir):
            _os.mkdir(self.ckptdir)

        result: list[_CheckpointStats] = []

        files = [
            fname for fname in _os.listdir(self.ckptdir)
            if fname.startswith("best.") and fname.endswith(".ckpt")
        ]
        for file in files:
            ckpt: _CheckpointSaveData = _torch.load(
                _os.path.join(self.ckptdir, file))
            loss = ckpt.loss
            epoch = ckpt.epoch
            result.append(_CheckpointStats(file, epoch, loss))
            del ckpt
        result.sort(key=lambda c: c.loss)
        return result

    def _get_latest_ckpt(self) -> str | None:
        if not _os.path.exists(self.ckptdir):
            _os.mkdir(self.ckptdir)

        files = [
            fname for fname in _os.listdir(self.ckptdir)
            if fname.startswith("latest.") and fname.endswith(".ckpt")
        ]

        if len(files) > 0:
            return files[0]

        return None

    def _save_best_ckpt(self, loss: float) -> None:
        ckpts = self._get_best_ckpts()
        should_save = len(ckpts) < self.save_best_count
        should_delete = False
        if not should_save:
            should_save = loss < ckpts[-1].loss
            should_delete = should_save

        if should_delete:
            _os.remove(_os.path.join(self.ckptdir, ckpts[-1].file))
        if should_save:
            self._save_checkpoint(
                f"best.loss={loss}.epoch={self.epoch}.ckpt",
                loss,
            )

    def _save_latest_ckpt(self, loss: float) -> None:
        ckpt = self._get_latest_ckpt()
        if ckpt != None:
            _os.remove(_os.path.join(self.ckptdir, ckpt))
        self._save_checkpoint(
            f"latest.loss={loss}.epoch={self.epoch}.ckpt",
            loss,
        )

    def _loop(
        self,
        stage: str,
        get_loss: _Callable[[list[_Tensor], int], _Tensor],
        log_epoch_only: bool = True,
    ) -> float:
        self.pbar = _tqdm_auto(
            getattr(self.data_loaders, stage),
            desc=f"{stage.capitalize()} {self.epoch}",
            position=0,
            leave=_IS_NOTEBOOK,
        )
        num_batches: int = len(self.pbar)
        epoch_loss: _Tensor = _torch.zeros(1)
        total_loss: _Tensor = _torch.zeros(1)
        total_loss_count: int = 0
        for batch_idx, batch in enumerate(self.pbar):
            batch = [t.to(self.device) for t in batch]
            loss = get_loss(batch, batch_idx)
            epoch_loss += loss.cpu()
            total_loss += loss.cpu()
            total_loss_count += 1

            if not log_epoch_only and ((total_loss_count >= self.log_interval)
                                       or (batch_idx == len(self.pbar) - 1)):
                self.log_scalar(f"{stage}/loss", total_loss / total_loss_count,
                                self.epoch * num_batches + batch_idx)
                total_loss_count = 0
                total_loss.fill_(0)

        epoch_loss /= num_batches

        if log_epoch_only:
            self.log_scalar(f"{stage}/loss", epoch_loss, self.epoch)

        return epoch_loss.item()

    def _train_loop(self) -> float:
        self.model.train()
        def get_loss(batch: list[_Tensor], batch_idx: int) -> _Tensor:
            self.model.zero_grad()
            loss = self.train_step(batch, batch_idx)
            loss.backward()
            self.optimizer.step()
            return loss

        return self._loop("train", get_loss, log_epoch_only=False)

    @_torch.no_grad()
    def _val_loop(self, ) -> float:
        self.model.eval()
        return self._loop("val", self.val_step)

    @_torch.no_grad()
    def _test_loop(self, ) -> float:
        self.model.eval()
        return self._loop("test", self.test_step)
