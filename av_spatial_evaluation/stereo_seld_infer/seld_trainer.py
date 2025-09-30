# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn
import torch.nn.functional
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from timm.scheduler import CosineLRScheduler

from seld_data_loader import create_data_loader
from net.net_seld import create_net_seld


class SELDTrainer(object):
    def __init__(self, args):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._data_loader = create_data_loader(self._args)

        self._net = create_net_seld(self._args)
        if self._args.parallel_gpu:
            self._net = torch.nn.DataParallel(self._net)
        self._net.to(self._device)
        self._net.train()

        self._criterion = SEDDOA(self._args.range_azi, self._args.loss_azi)
        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._args.learning_rate,
            weight_decay=self._args.weight_decay
        )
        self._lr_scheduler = CosineLRScheduler(self._optimizer,
                                               t_initial=25, lr_min=self._args.learning_rate * 0.5,
                                               warmup_t=5, warmup_lr_init=self._args.learning_rate * 0.2, warmup_prefix=True)  # hard coding

        self._swa_model = AveragedModel(self._net)
        self._swa_start = 30  # hard coding
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=self._args.learning_rate * 0.5)  # hard coding

    def receive_input(self, sample):
        _input_a, _label_cat, _label_azi, _ = sample

        self._input_a = _input_a.to(self._device, non_blocking=True)
        self._label_cat = _label_cat.to(self._device, non_blocking=True)
        self._label_azi = _label_azi.to(self._device, non_blocking=True)

    def back_propagation(self):
        self._net.train()
        self._optimizer.zero_grad()

        output_net = self._net(self._input_a)
        self._output_cat = output_net[0]
        self._output_azi = output_net[1]
        self._loss = self._criterion(self._output_cat, self._output_azi, self._label_cat, self._label_azi)
        self._loss.backward()

        self._optimizer.step()

    def save(self, each_monitor_path=None, iteration=None, start_time=None):
        self._each_checkpoint_path = '{}/params_{}_{:07}.pth'.format(
            each_monitor_path,
            start_time,
            iteration)
        if self._args.parallel_gpu:
            torch_net_state_dict = self._net.module.state_dict()
        else:
            torch_net_state_dict = self._net.state_dict()
        checkpoint = {'model_state_dict': torch_net_state_dict,
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'scheduler_state_dict': self._lr_scheduler.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state()}
        torch.save(checkpoint, self._each_checkpoint_path)
        print('save checkpoint to {}.'.format(self._each_checkpoint_path))

        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            iter_times = 10
            batches = torch.zeros((iter_times,
                                   self._input_a.shape[0],
                                   self._input_a.shape[1],
                                   self._input_a.shape[2],
                                   self._input_a.shape[3])).to(self._device)
            for i, each_batch in enumerate(self._data_loader):
                if i == iter_times:
                    break
                batches[i] = each_batch[0]
            my_update_bn(batches, self._swa_model)

            self._each_swa_checkpoint_path = self._each_checkpoint_path.replace("params_", "params_swa_")
            if self._args.parallel_gpu:
                torch_net_state_dict = self._swa_model.module.module.state_dict()
            else:
                torch_net_state_dict = self._swa_model.module.state_dict()
            checkpoint = {'model_state_dict': torch_net_state_dict,
                          'optimizer_state_dict': self._optimizer.state_dict(),
                          'scheduler_state_dict': self._lr_scheduler.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state()}
            torch.save(checkpoint, self._each_swa_checkpoint_path)
            print('save checkpoint to {}.'.format(self._each_swa_checkpoint_path))

    def lr_step(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            self._swa_scheduler.step()
        else:
            self._lr_scheduler.step(pseudo_epoch)

    def swa_update(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            self._swa_model.update_parameters(self._net)

    def get_loss(self):
        return self._loss.cpu().detach().numpy()

    def get_lr(self):
        return self._optimizer.state_dict()['param_groups'][0]['lr']

    def get_each_model_path(self, iteration):
        pseudo_epoch = int(iteration / self._args.model_save_interval)
        if pseudo_epoch > self._swa_start:
            return self._each_swa_checkpoint_path
        else:
            return self._each_checkpoint_path


def my_update_bn(loader, model, device=None):  # mainly from torch.optim.swa_utils.update_bn
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = None
            module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class MaskedMSELoss(torch.nn.MSELoss):
    def __init__(self, reduction='none'):
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def forward(self, input, target, mask):
        return torch.nn.functional.mse_loss(mask * input, target, reduction=self.reduction)


class SEDDOA(object):
    def __init__(self, range_azi, loss_azi, coef_cat_loss=0.5, coef_azi_loss=0.5):
        super().__init__()
        self.range_azi = range_azi
        self.loss_azi = loss_azi
        self.coef_cat_loss = coef_cat_loss
        self.coef_azi_loss = coef_azi_loss
        self.each_cat_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        if self.loss_azi == "mse":
            self.each_azi_loss = torch.nn.MSELoss(reduction='none')
        elif self.loss_azi == "masked_mse":
            self.each_azi_loss = MaskedMSELoss(reduction='none')

    def _each_cat_calc(self, output, target):
        return self.each_cat_loss(output, target).mean(dim=(1, 2))  # frame-level

    def _each_azi_calc(self, output, target, mask):
        if self.loss_azi == "mse":
            return self.each_azi_loss(output, target).mean(dim=(1, 2))  # frame-level
        elif self.loss_azi == "masked_mse":
            return self.each_azi_loss(output, target, mask).mean(dim=(1, 2))  # frame-level

    def _change_range_azi(self, target_azi):
        if self.range_azi == "0to256_0to1":
            target_azi_range = target_azi / 256  # 0<=azi<256 -> 0<=azi_range<1
        elif self.range_azi == "0to256_-1to1":
            target_azi_range = (target_azi / 128) - 1 # 0<=azi<256 -> -1<=azi_range<1
        return target_azi_range

    def __call__(self, output_cat, output_azi, target_cat, target_azi):
        """
        Args:
            output_cat: [batch_size, num_track=1, num_cat=13, num_frames]
            output_azi: [batch_size, num_track=1, num_azi=13, num_frames]
            target_cat: [batch_size, num_track=1, num_cat=13, num_frames]
            target_azi: [batch_size, num_track=1, num_azi=13, num_frames]
        Return:
            loss: scalar
        """
        target_azi_range = self._change_range_azi(target_azi)
        loss_cat = self._each_cat_calc(output_cat, target_cat)
        loss_azi = self._each_azi_calc(output_azi, target_azi_range, target_cat)
        loss0 = self.coef_cat_loss * loss_cat + self.coef_azi_loss * loss_azi
        loss = loss0.mean()
        return loss
