import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SR_Distill_Model(SRModel):
    def __init__(self, opt):
        super(SR_Distill_Model, self).__init__(opt)
        
        # 初始化网络
        self._init_networks(opt)
        
        # 注册特征钩子
        self._register_feature_hooks()
        
        # 加载预训练模型
        self.load_pretrained_models(opt)

        if self.is_train:
            self.init_training_settings()

    def _register_feature_hooks(self):
        """注册特征钩子（无适配器版本）"""
        self.teacher_features = OrderedDict()
        self.student_features = OrderedDict()

        # 从配置获取特征层名称
        train_opt = self.opt.get('train', {})
        self.teacher_layers = train_opt.get('teacher_feature_layers', [
            'block_1.body',  # 示例层路径
            'block_3.body',
            'block_6.body'
        ])
        self.student_layers = train_opt.get('student_feature_layers', [
            'block_1.body', 
            'block_3.body',
            'block_6.body'
        ])

        # 教师网络钩子
        def get_teacher_hook(layer_name):
            def hook(module, input, output):
                self.teacher_features[layer_name] = output.detach()
            return hook

        # 学生网络钩子
        def get_student_hook(layer_name):
            def hook(module, input, output):
                self.student_features[layer_name] = output
            return hook

        # 绑定教师钩子
        for layer in self.teacher_layers:
            module = self._get_layer(self.net_teacher, layer)
            module.register_forward_hook(get_teacher_hook(layer))

        # 绑定学生钩子
        for layer in self.student_layers:
            module = self._get_layer(self.net_g, layer)
            module.register_forward_hook(get_student_hook(layer))

    def _get_layer(self, model, layer_path):
        """通过点分路径获取网络层"""
        modules = layer_path.split('.')
        curr_module = model
        for m in modules:
            curr_module = getattr(curr_module, m)
        return curr_module

    def init_training_settings(self):
        """初始化训练设置（简化版）"""
        super().init_training_settings()
        
        # 特征蒸馏损失
        train_opt = self.opt['train']
        self.cri_feature = build_loss(train_opt.get('feature_loss', {'type': 'L1Loss'}))
        self.feature_weight = train_opt.get('feature_weight', 0.1)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        # 清空特征缓存
        self.teacher_features.clear()
        self.student_features.clear()

        # 学生前向
        self.output = self.net_g(self.lq)
        
        # 教师前向（无梯度）
        with torch.no_grad():
            _ = self.net_teacher(self.lq)

        # 计算总损失
        l_total = 0
        loss_dict = OrderedDict()

        # 像素损失
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 输出蒸馏
        l_distill = self.cri_distill(self.output, self.teacher_output)
        l_total += l_distill * self.opt['train'].get('distill_weight', 1.0)
        loss_dict['l_distill'] = l_distill

        # 特征蒸馏
        l_feature = 0
        matched_pairs = 0
        for s_layer, t_layer in zip(self.student_layers, self.teacher_layers):
            if s_layer in self.student_features and t_layer in self.teacher_features:
                s_feat = self.student_features[s_layer]
                t_feat = self.teacher_features[t_layer]
                
                # 简单尺寸对齐（仅空间维度）
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                
                l_feature += self.cri_feature(s_feat, t_feat)
                matched_pairs += 1
        
        if matched_pairs > 0:
            l_feature /= matched_pairs  # 平均各层损失
            l_total += l_feature * self.feature_weight
            loss_dict['l_feature'] = l_feature

        # 感知损失
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 反向传播
        l_total.backward()
        self.optimizer_g.step()

        # 日志记录
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA更新
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)