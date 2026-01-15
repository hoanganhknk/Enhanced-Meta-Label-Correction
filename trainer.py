import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import tocuda, accuracy, log_losses, log_acc
from meta import teacher_backward, teacher_backward_ms


class Trainer:
    def __init__(self, rank, args, main_net, meta_net, enhancer,
                 gold_loader, silver_loader, valid_loader, test_loader,
                 num_classes, logger, exp_id=None):
        self.rank = rank
        self.args = args
        self.main_net = main_net
        self.meta_net = meta_net
        self.enhancer = enhancer
        self.gold_loader = gold_loader
        self.silver_loader = silver_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.logger = logger
        self.exp_id = exp_id

        if rank == 0:
            self.writer = SummaryWriter(args.logdir + '/' + exp_id)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.best_val = 0
        self.best_main_state = None
        self.best_meta_state = None
        self.best_enhancer_state = None

        self._setup_training()

    def _setup_training(self):
        ''' Fetch optimizers and schedulers for
          training the student, teacher and its enhancer'''
        args = self.args
        main_params = self.main_net.parameters()
        meta_params = self.meta_net.parameters()
        enhancer_params = self.enhancer.parameters()

        self.main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)

        self.meta_opt = torch.optim.SGD(meta_params, lr=args.meta_lr,
                                        weight_decay=args.wdecay)
        self.enhancer_opt = torch.optim.SGD(enhancer_params, lr=args.meta_lr,
                                            weight_decay=args.wdecay)

        milestones = args.sched_milestones
        if isinstance(milestones, str):
            # Accept formats like '20', '20,30', or '20 30'
            milestones = [int(x) for x in milestones.replace(',', ' ').split() if x.strip()]
        gamma = args.sched_gamma
        self.main_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.main_opt, milestones=milestones, gamma=gamma)
        self.meta_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.meta_opt, milestones=milestones, gamma=gamma)
        self.enhancer_schdlr = torch.optim.lr_scheduler.MultiStepLR(self.enhancer_opt, milestones=milestones, gamma=gamma)

    def _training_iter(self, data_s, target_s, data_g, target_g):
        ''' Perform a single training iteration'''
        data_g, target_g = tocuda(self.rank, data_g), tocuda(self.rank, target_g)
        data_s, target_s = tocuda(self.rank, data_s), tocuda(self.rank, target_s)

        # bi-level optimization stage
        eta = self.main_schdlr.get_last_lr()[0]
        kwargs = {'rank': self.rank, 'args': self.args,
                  'main_net': self.main_net, 'main_opt': self.main_opt,
                  'teacher': self.meta_net, 'teacher_opt': self.meta_opt,
                  'enhancer': self.enhancer, 'enhancer_opt': self.enhancer_opt,
                  'data_s': data_s, 'target_s': target_s,
                  'data_g': data_g, 'target_g': target_g,
                  'eta': eta, 'num_classes': self.num_classes}
        teacher_backward_fn = teacher_backward if self.args.gradient_steps == 1 else teacher_backward_ms
        loss_g, loss_s, t_loss = teacher_backward_fn(**kwargs)

        return loss_g.item(), loss_s.item(), t_loss.item()

    def _training_epoch(self, epoch):
        self.main_net.train()
        self.meta_net.train()
        self.enhancer.train()

        losses_g = []
        losses_s = []
        losses_t = []

        gold_iter = iter(self.gold_loader)
        for i, (data_s, target_s) in enumerate(self.silver_loader):
            try:
                data_g, target_g = next(gold_iter)
            except StopIteration:
                gold_iter = iter(self.gold_loader)
                data_g, target_g = next(gold_iter)

            loss_g, loss_s, t_loss = self._training_iter(data_s, target_s, data_g, target_g)
            losses_g.append(loss_g)
            losses_s.append(loss_s)
            losses_t.append(t_loss)

            if self.rank == 0 and (i + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{self.args.epochs}] Iter [{i+1}/{len(self.silver_loader)}] '
                                 f'loss_g: {loss_g:.4f} loss_s: {loss_s:.4f} t_loss: {t_loss:.4f}')

        self.main_schdlr.step()
        self.meta_schdlr.step()
        self.enhancer_schdlr.step()

        return float(np.mean(losses_g)), float(np.mean(losses_s)), float(np.mean(losses_t))

    @torch.no_grad()
    def _eval(self, loader, net, epoch=None, prefix='val'):
        net.eval()
        total_acc = 0
        total = 0
        total_loss = 0

        for data, target in loader:
            data, target = tocuda(self.rank, data), tocuda(self.rank, target)
            outputs = net(data)
            loss = F.cross_entropy(outputs, target)
            acc1 = accuracy(outputs, target, topk=(1,))[0]

            bs = data.size(0)
            total += bs
            total_loss += loss.item() * bs
            total_acc += acc1.item() * bs

        avg_loss = total_loss / total
        avg_acc = total_acc / total

        if self.rank == 0 and epoch is not None:
            self.logger.info(f'[{prefix}] Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}')

        return avg_loss, avg_acc

    def train(self):
        for epoch in range(self.args.epochs):
            loss_g, loss_s, t_loss = self._training_epoch(epoch)

            if self.rank == 0:
                log_losses(self.writer, epoch, loss_g, loss_s, t_loss)

            # validation on main net
            val_loss, val_acc = self._eval(self.valid_loader, self.main_net, epoch=epoch, prefix='val')

            if self.rank == 0:
                log_acc(self.writer, epoch, val_acc, prefix='val/main')

            # save best
            if val_acc > self.best_val:
                self.best_val = val_acc
                if self.rank == 0:
                    self.logger.info('Saving best models...')
                self.best_main_state = copy.deepcopy(self.main_net.state_dict())
                self.best_meta_state = copy.deepcopy(self.meta_net.state_dict())
                self.best_enhancer_state = copy.deepcopy(self.enhancer.state_dict())

    @torch.no_grad()
    def final_eval(self):
        if self.best_main_state is not None:
            self.main_net.load_state_dict(self.best_main_state, strict=True)

        test_loss, test_acc = self._eval(self.test_loader, self.main_net, epoch=None, prefix='test')
        if self.rank == 0:
            self.logger.info(f'[test] loss={test_loss:.4f}, acc={test_acc:.2f}')
        return test_acc
