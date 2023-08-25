# 三路：
# 一路输入TE拼接，预测T
# 一路输入TE拼接，预测E
# 一路输入TE拼接，预测TE拼接
# 在两层K5P2之间，添加一层RNN
# 特征解耦
# 两个解耦器
# T一路将提取的T与主路融合
# E一路将提取的E与主路融合
# T一路提取的E与E一路提取的T之间做相似度loss
# T一路提取的T与E一路提取的E相加，T一路提取的E与E一路提取的T相加，二者做相似度loss
# 特征解耦的mask的维度从T变为D，即在特征维度做了mask
# 
# 使用Instance Norm做特征解耦，先提取出Domain-invarant特征

# cross_loss_T 和 cross_Loss_E 使用

import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules.criterions_contrastive import ContrastiveLoss
from modules import BiLSTMLayer
from modules.tconv import TemporalConv
from modules.feature_disentangle_channel_instance import FeatureDisentangle
from modules.attention import DotProductAttention,LinearDotProductAttention

import random
import itertools

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
# matplotlib.use('pdf')


import matplotlib.pyplot as plt


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SLRModel(nn.Module):
    def __init__(self, num_classes, num_classes_T,num_classes_E,c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size=1024, gloss_dict=None, gloss_dict_T=None,gloss_dict_E=None,loss_weights=None):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.num_classes_T=num_classes_T
        self.num_classes_E=num_classes_E
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        # T E 专用的时序卷积，少一层卷积
        # self.conv1d_less = TemporalConv(input_size=512,
        #                            hidden_size=hidden_size,
        #                            conv_type=2,
        #                            use_bn=use_bn,
        #                            num_classes=num_classes)
        # 组合的时序卷积
        self.conv1d_combine_first = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=2,
                                   use_bn=use_bn,
                                   num_classes=num_classes)

        self.gloss_dict=gloss_dict
        self.gloss_dict_T=gloss_dict_T
        self.gloss_dict_E=gloss_dict_E

        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())

        self.decoder_combine = utils.Decode(gloss_dict, num_classes, 'beam')

        self.temporal_model_combine_first = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

        self.classifier_combine = nn.Linear(hidden_size, self.num_classes)

        self.register_backward_hook(self.backward_hook)








    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[max(len_x) * idx:max(len_x) * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], max(len_x))
                       for idx, lgt in enumerate(len_x)])

        # x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = self.conv2d(x)
        # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
        #                for idx, lgt in enumerate(len_x)])
        return x

    def cosine_similarity_print(self,a,figure_name):
        length_a,dim=a.size()
        cosine_similarity_image=torch.zeros(length_a,length_a)
        for j in range(0,length_a):
            for k in range(0,length_a):
                cosine_similarity_image[j,k]=torch.cosine_similarity(a[j],a[k],dim=0)
        cosine_similarity_image_np=cosine_similarity_image.cpu().detach().numpy()
        # plt.figure()
        # plt.imshow(cosine_similarity_image_np[i])
        # plt.colorbar()
        # ax=plt.gca()
        # ax.xaxis.set_ticks_position('top')
        # ax.spines['bottom'].set_position(('data',0))
        # plt.savefig(figure_name)
        # plt.close()
        # print('')

    def sim_matrix(self,a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def cosine_similarity_onevideo(self, frames, input_lengths,right_pad=None, bound=20):
        """
        input:
        a: tensor[length,dim] one video
        b: tensor[length,dim] another video which size is same as a, often a=b
        input_lengths: tensor[] size=(1) the real length of video exclude empty frame which generated by collect_fn
        bound: int  避免无手的空白帧超过一定帧数

        output:
        hand_start:动作起始帧
        hand_end:动作结束帧

        """
        # print('start')
        # print(datetime.datetime.now())
        a=frames
        b=frames
        length_a, dim = a.size()
        length_b, dim = b.size()
        cosine_similarity_image = self.sim_matrix(a, b)
        
        # 前六帧是重复的第一帧
        hand_start=0
        # hand_start =0
        # hand_end = input_lengths.item()
        hand_end = input_lengths.item()-1

        for i in range(0, input_lengths.item()):
            square_similarity = cosine_similarity_image[0:i, 0:i]
            if_no_hand = torch.gt(square_similarity, 0.95)
            if if_no_hand.all():
                hand_start += 1
            else:
                break
        for i in range(input_lengths.item()-1, 0, -1):
            square_similarity = cosine_similarity_image[i:input_lengths.item(), i:input_lengths.item()]
            if_no_hand = torch.gt(square_similarity, 0.95)
            if if_no_hand.all():
                hand_end -= 1
            else:
                break

        # 为避免得到的动作起始帧出大错，人为约束前后空白帧不超过xx帧
        hand_start = min(hand_start, bound)
        hand_end = max(hand_end, input_lengths.item() - bound)
        if hand_end-hand_start<30:
            hand_end=input_lengths.item()-1
            if hand_end-hand_start<30:
                hand_start=0

        # print('hand start{}  hand end{}  length of entity{}'.format(hand_start, hand_end, input_lengths.item()))
        return hand_start, hand_end

    def forward_train(self, Tx, Ex , Tlen_x_all, Elen_x_all, label=None, label_lgt=None,T_label=None, T_label_lgt=None,E_label=None,E_label_lgt=None,insert_word_before=None,insert_word_after=None,insert_frame_before=None,insert_frame_after=None,right_pad_list_T=None,right_pad_list_E=None):
        # videos
        batch, Ttemp, channel, height, width = Tx.shape
        _, Etemp, _, _, _ = Ex.shape
        Clen_x_all = Tlen_x_all + Elen_x_all

        Tinputs = Tx.reshape(batch * Ttemp, channel, height, width)
        Tframewise_all = self.masked_bn(Tinputs, Tlen_x_all)
        Tframewise_all = Tframewise_all.reshape(batch, Ttemp, -1)

        Einputs = Ex.reshape(batch * Etemp, channel, height, width)
        Eframewise_all = self.masked_bn(Einputs, Elen_x_all)
        Eframewise_all = Eframewise_all.reshape(batch, Etemp, -1)

        ### remove blank frames
        Tlen_x=torch.zeros_like(Tlen_x_all).cuda()
        Elen_x=torch.zeros_like(Elen_x_all).cuda()
        Clen_x=torch.zeros_like(Clen_x_all).cuda()
        hand_start_T_list=[]
        hand_end_T_list=[]
        hand_start_E_list=[]
        hand_end_E_list=[]

        for i in range(0,batch):
            hand_start_T,hand_end_T=self.cosine_similarity_onevideo(Tframewise_all[i],Tlen_x_all[i])
            hand_start_E,hand_end_E=self.cosine_similarity_onevideo(Eframewise_all[i],Elen_x_all[i])
            hand_start_T_list.append(hand_start_T)
            hand_start_E_list.append(hand_start_E)
            hand_end_T_list.append(hand_end_T)
            hand_end_E_list.append(hand_end_E)
            Tlen_x[i]=hand_end_T-hand_start_T
            Elen_x[i]=hand_end_E-hand_start_E
            Clen_x[i]=Tlen_x[i]+Elen_x[i]

        Tlen_x_max=torch.max(Tlen_x)
        Elen_x_max=torch.max(Elen_x)
        Tframewise=torch.zeros(batch,Tlen_x_max,Tframewise_all.size(2)).cuda()
        Eframewise=torch.zeros(batch,Elen_x_max,Eframewise_all.size(2)).cuda()
        
        for i in range(0,batch):
            Tframewise[i,0:hand_end_T_list[i]-hand_start_T_list[i]]=Tframewise_all[i,hand_start_T_list[i]:hand_end_T_list[i]]
            Eframewise[i,0:hand_end_E_list[i]-hand_start_E_list[i]]=Eframewise_all[i,hand_start_E_list[i]:hand_end_E_list[i]]


        # 找出e要插入位置前后的两个词
        before_e_word=[]
        after_e_word=[]
        for i in range(batch):
            before_e_word.append(insert_word_before[i].item())
            after_e_word.append(insert_word_after[i].item())

        # blank combine
        seg_T=[]
        for i in range(batch):          
            # 在template开头插入
            if before_e_word[i]==0:
                seg = 0
                seg_T.append(seg)
            # 在template末尾插入
            elif after_e_word[i]==0:
                seg =  Tlen_x[i]
                seg_T.append(seg)
            else:
                seg_primitive = insert_frame_before[i]
                seg = int(seg_primitive)//2+6-hand_start_T_list[i]
                seg = min(seg,Tlen_x[i])
                seg = max(seg,0)
                seg_T.append(seg)

        seg_T=torch.Tensor(seg_T).int()


        Comframewise=torch.zeros(batch, max(Clen_x), 512).cuda()
        # combine
        for i in range(batch):
            try:
                Cframewise=torch.cat((Tframewise[i][:seg_T[i]], Eframewise[i][:Elen_x[i]], Tframewise[i][seg_T[i]:Tlen_x[i]]), dim=0)
                Comframewise[i,:Clen_x[i],:]=Cframewise
            except:
                print('seg_T[i]',seg_T[i])
                print('Tframewise.size()',Tframewise.size())
            
        Comframewise = Comframewise.transpose(1, 2)
        # Comframewise = Tframewise.transpose(1, 2)
        # Clen_x=Tlen_x


        # 对拼接视频做手语识别
        conv1d_outputs_first = self.conv1d_combine_first(Comframewise, Clen_x)
        # x: T, B, C
        x_first = conv1d_outputs_first['visual_feat']
        lgt_first = conv1d_outputs_first['feat_len']
        tm_outputs_first = self.temporal_model_combine_first(x_first, lgt_first)

        outputs = self.classifier_combine(tm_outputs_first['predictions'])
        pred = self.decoder_combine.decode(outputs, lgt_first, batch_first=False, probs=False)
        conv_pred = self.decoder_combine.decode(conv1d_outputs_first['conv_logits'], lgt_first, batch_first=False,
                                                probs=False)

        return {
            "framewise_features": Comframewise,
            "visual_features": x_first,
            "feat_len": lgt_first,
            "conv_logits": conv1d_outputs_first['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def forward_test(self, x, len_x_all, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise_all = self.masked_bn(inputs, len_x_all)
            framewise_all = framewise_all.reshape(batch, temp, -1)
        else:
            # frame-wise features
            framewise_all = x

        hand_start_list=[]
        hand_end_list=[]
        len_x=torch.zeros_like(len_x_all).cuda()
        for i in range(0,batch):
            hand_start,hand_end=self.cosine_similarity_onevideo(framewise_all[i],len_x_all[i])
            hand_start_list.append(hand_start)
            hand_end_list.append(hand_end)

            len_x[i]=hand_end-hand_start

        len_x_max=torch.max(len_x)
        framewise=torch.zeros(batch,len_x_max,framewise_all.size(2)).cuda()
        len_x_max=torch.max(len_x)
        for i in range(0,batch):
            framewise[i,0:hand_end_list[i]-hand_start_list[i]]=framewise_all[i,hand_start_list[i]:hand_end_list[i]]
        
        # First step
        # 整体识别 
        conv1d_outputs_first = self.conv1d_combine_first(framewise.transpose(1, 2), len_x)
        # x: T, B, C
        x_first = conv1d_outputs_first['visual_feat']
        lgt_first = conv1d_outputs_first['feat_len']
        tm_outputs_first = self.temporal_model_combine_first(x_first, lgt_first)
        outputs = self.classifier_combine(tm_outputs_first['predictions'])


        pred = None if self.training \
            else self.decoder_combine.decode(outputs, lgt_first, batch_first=False, probs=False)

        conv_pred = None if self.training \
            else self.decoder_combine.decode(conv1d_outputs_first['conv_logits'], lgt_first, batch_first=False, probs=False)

        # print(pred)
        # print(conv_pred)

        return {
            "framewise_features": framewise,
            "visual_features": x_first,
            "feat_len": lgt_first,
            "conv_logits": conv1d_outputs_first['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }


    def criterion_calculation(self, ret_dict, label, label_lgt,T_label, T_label_lgt,E_label,E_label_lgt,epoch):
        loss = 0
        # E_label_lgt=torch.ones(4)
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                # if epoch>2:
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      T_label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      T_label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                # if epoch>2:
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      T_label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      T_label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                # if epoch>2:
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['distillation_disentangle'] = SeqKD(T=8)
        self.loss['contrastive'] = ContrastiveLoss()
        return self.loss
