import torch
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class ETDSModel(SRModel):
    """ ETDS model for single image super-resolution. """
    def __init__(self, opt):
        super(ETDSModel, self).__init__(opt)

        self.fixed_residual_model_iters = opt['train']['fixed_residual_model_iters']        # number of rounds with K_r and K_{r2b} fixed parameters
        self.interpolation_loss_weight = opt['train']['interpolation_loss_weight']          # \alpha in Eq. 15
        

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        ''' update learning rate '''
        if current_iter == 1:
            # Set K_r, K_{r2b}, and K_{b2r} not to be trained at the beginning
            for name, param in self.net_g.named_parameters():
                if 'residual' in name:
                    param.requires_grad_(False)
        elif current_iter == self.fixed_residual_model_iters:
            # Set K_r, and K_{r2b} to participate in training after {fixed_residual_model_iters} rounds
            for name, param in self.net_g.named_parameters():
                if 'residual' in name:
                    param.requires_grad_(True)
        super(ETDSModel, self).update_learning_rate(current_iter, warmup_iter)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # double branch outputs
        self.output, output1 = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            # double pixel loss
            l_pix = self.cri_pix(self.output, self.gt) + self.interpolation_loss_weight * self.cri_pix(output1, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def test(self):
        ''' test '''
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)[0]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)[0]
            self.net_g.train()
