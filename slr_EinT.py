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
import math
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
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes)

        self.conv1d_T_first = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes_T)
        
        self.conv1d_E_first = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes_E)

        self.feature_disentangle_T = FeatureDisentangle(input_size=hidden_size,hidden_size=hidden_size)
        self.feature_disentangle_E = FeatureDisentangle(input_size=hidden_size,hidden_size=hidden_size)

        self.conv1d_combine_second = TemporalConv(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes)

        self.conv1d_T_second = TemporalConv(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes_T)
        
        self.conv1d_E_second = TemporalConv(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   conv_type=1,
                                   use_bn=use_bn,
                                   num_classes=num_classes_E)
        self.gloss_dict=gloss_dict
        self.gloss_dict_T=gloss_dict_T
        self.gloss_dict_E=gloss_dict_E

        self.gloss_dict_word2index=dict((v, k) for k, v in gloss_dict.items())

        self.decoder_combine = utils.Decode(gloss_dict, num_classes, 'beam')
        self.decoder_T = utils.Decode(gloss_dict_T, num_classes_T, 'beam')
        self.decoder_E = utils.Decode(gloss_dict_E, num_classes_E, 'beam')
        self.temporal_model_combine_first = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
                                              
        self.temporal_model_T_first = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
        self.temporal_model_E_first = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
        self.temporal_model_combine_second = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
                                              
        self.temporal_model_T_second = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
        self.temporal_model_E_second = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, dropout=0,bidirectional=True)
        self.classifier_combine = nn.Linear(hidden_size, self.num_classes)
        self.classifier_T = nn.Linear(hidden_size, self.num_classes_T)
        self.classifier_E = nn.Linear(hidden_size, self.num_classes_E)
        self.register_backward_hook(self.backward_hook)

        self.feature_fusion = nn.Linear(hidden_size*3, hidden_size)
        # self.feature_fusion_activation=nn.ReLU(inplace=True)

        self.attention_T=LinearDotProductAttention(dropout=0,in_features=hidden_size,out_features=hidden_size)
        self.attention_E=LinearDotProductAttention(dropout=0,in_features=hidden_size,out_features=hidden_size)
        self.layer_norm=nn.LayerNorm(hidden_size,elementwise_affine=False)
        self.p = nn.AdaptiveMaxPool1d(100)
        self.p1 = nn.AdaptiveMaxPool1d(1)
        self.MLP1 = nn.Linear(512, 2)
        self.S = torch.nn.Sigmoid()
        self.loca_num=2
        self.sliding=4
        self.query = nn.Linear(512, 512)
        self.key = nn.Linear(512, 512)
        self.value = nn.Linear(512, 512)
        self.rnn = nn.LSTM(512, 512 // 2,
                           bidirectional=True,
                           dropout=0.5, batch_first=True)
        self.predict = nn.Linear(512, 1)
        self.tau = 5
        self.tau_decay = -0.05
        self.dropout = nn.Dropout(p=0.2)
        self.params_a = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.params_a = torch.tensor([1.44], requires_grad=True)
        self.params_a.data.fill_(0.95)
        self.params_b = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.params_b.data.fill_(0.95)
        # self.soft = nn.Softmax(dim=1)
        # self.weight = nn.Linear(512, 1)
        # self.rnn = nn.LSTM(512, 512 // 2,
        #                    bidirectional=True,
        #                    dropout=0.5, batch_first=True)

        # self.predict = nn.Linear(512, 1)

        # self.tau = 5

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
        plt.figure()    
        plt.imshow(cosine_similarity_image_np[i])
        plt.colorbar()
        ax=plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.spines['bottom'].set_position(('data',0))
        plt.savefig(figure_name)
        plt.close()
        print('')

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
            # Tframewise_list.append(Tframewise_all[i][hand_start_T:hand_end_T])
            # Eframewise_list.append(Tframewise_all[i][hand_start_E:hand_end_E])
            Tlen_x[i]=hand_end_T-hand_start_T
            Elen_x[i]=hand_end_E-hand_start_E
            Clen_x[i]=Tlen_x[i]+Elen_x[i]
        
        # print('hand_start_T_list',hand_start_T_list)
        # print('hand_end_T_list',hand_end_T_list)
        # print('hand_start_E_list',hand_start_E_list)
        # print('hand_end_E_list',hand_end_E_list)
        # print('Tlen_x_all',Tlen_x_all)
        # print('Elen_x_all',Elen_x_all)

        Tframewise = []
        Eframewise = []
        for i in range(0, batch):
            # Tframewise[i, 0:hand_end_T_list[i] - hand_start_T_list[i]] = features_input[i][
            #                                                              hand_start_T_list[i]:hand_end_T_list[i]]
            Tframewise.append(Tframewise_all[i][hand_start_T_list[i]:hand_end_T_list[i]])
            # Eframewise[i, 0:hand_end_E_list[i] - hand_start_E_list[i]] = features_e_input[i][
            #                                                              hand_start_E_list[i]:hand_end_E_list[i]]
            Eframewise.append(Eframewise_all[i][hand_start_E_list[i]:hand_end_E_list[i]])


        
        # greedyPred=self.decoder_T.decode(outputs_T, lgt_T, batch_first=False, probs=False,search_mode='max')
        # greedyPredList=[]


        #### EinT
        e_length = torch.max(Elen_x)
        t_length = torch.max(Tlen_x)
        c=Elen_x+Tlen_x
        e_feature = []
        c_length = torch.max(c)
        concatenated_feature_final = torch.zeros(batch, c_length, 512).cuda()
        for i in range(batch):
            com_feature_batch = torch.cat((Tframewise[i], Tframewise[i][-1, :].unsqueeze(0)), dim=0)
            # print('t_len:',Tframewise[i].size(0))
            mixed_query_layer = self.query(com_feature_batch)
            mixed_key_layer = self.key(Eframewise[i])
            mixed_value_layer = self.value(Eframewise[i])
            attention_scores = torch.matmul(
                mixed_query_layer, mixed_key_layer.transpose(0, 1))
            attention_scores = attention_scores / \
                               math.sqrt(512)
            attention_probs = nn.Softmax(dim=1)(attention_scores)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, mixed_value_layer)
            com_feature_output_batch = context_layer + com_feature_batch
            com_feature_output_batch, _ = self.rnn(com_feature_output_batch.unsqueeze(0))
            location = self.predict(com_feature_output_batch.squeeze(0)).transpose(0, 1)
            location = torch.log(location.clamp(min=1e-8))
            if self.training:
                action = F.gumbel_softmax(location, self.tau, hard=True, dim=1)  # B*(M+1)
            else:
                action = F.gumbel_softmax(location, 1e-5, hard=True, dim=1)  # B*(M+1)
            index_action = torch.max(action, 1)[1]
            # print('predict:',index_action)
            b = action[:, index_action[0]:index_action[0] + 1][0].repeat(Elen_x[i])
            b = b.unsqueeze(0)
            action_list = torch.cat([action[:, :index_action[0]], b, action[:, index_action[0] + 1:]], 1)  #### M'
            action_list = action_list.squeeze()
            action_clone_tensor = action_list
            action_clone_tensor_t = torch.ones(Tlen_x[i] + Elen_x[i]).cuda() - action_clone_tensor  #### 1-M'

            concatenated_feature_t = torch.zeros(Tlen_x[i] + Elen_x[i], 512).cuda()
            concatenated_feature_e = torch.zeros(Tlen_x[i] + Elen_x[i], 512).cuda()
            concatenated_feature_e[action_list == True] = Eframewise[i]
            concatenated_feature_t[action_list == False] = Tframewise[i]
            concatenated_feature = torch.einsum('tc,t->tc', concatenated_feature_t,
                                                action_clone_tensor_t) + torch.einsum('tc,t->tc',
                                                                                      concatenated_feature_e,
                                                                                      action_clone_tensor)
            length = concatenated_feature.size(0)
            concatenated_feature_final[i, :length, :] = concatenated_feature
            total_e = concatenated_feature[index_action[0]:(index_action[0] + Elen_x[i])]
            e_feature.append(total_e)
            ###############FICP
            # features_t_one = torch.cat((concatenated_feature[:index_action[0]],
            #                             concatenated_feature[(index_action[0] + Elen_x[i]):]), dim=0)
            # total_e = concatenated_feature[index_action[0]:(index_action[0] + Elen_x[i])]
            #
            # total_t_interpolation_before = torch.zeros(total_e.size(0), features_t_one.size(1)).cuda()
            # total_t_interpolation_after = torch.zeros(total_e.size(0), features_t_one.size(1)).cuda()
            # if index_action[0] == features_t_one.size(0):
            #
            #     for order in range(0, min(self.sliding, total_e.size(0))):
            #         total_t_interpolation_before[order] = features_t_one[-1]
            #     before = torch.zeros(total_e.size(0), 1).cuda()
            #     after = torch.zeros(total_e.size(0), 1).cuda()
            #     for order in range(0, self.sliding):
            #         before[order] = (-order - 1)
            #         after[-(order + 1)] = (-order - 1)
            #     before_beta = before * self.params_a
            #     total_e_factor = torch.ones(total_e.size(0), 1).cuda()
            #     if total_e.size(0) > self.sliding:
            #         total_e_factor[:self.sliding] = 1 - (torch.exp(before_beta)[:self.sliding])
            #     total_e_new = total_t_interpolation_before * torch.exp(
            #         before_beta) + total_e * total_e_factor
            #     ##### style
            #     # cFF = total_e_new
            #     # sFF = Tframewise[i]
            #     # cMean = torch.mean(cFF, dim=1, keepdim=True)
            #     # cMean = cMean.expand_as(cFF)
            #     # cF = cFF - cMean
            #     # sMean = torch.mean(sFF, dim=1, keepdim=True)
            #     # sMean = sMean.expand_as(sFF)
            #     #
            #     # sF = sFF - sMean
            #     # cMatrix = self.cnet(cF.unsqueeze(0).transpose(1, 2))
            #     # sMatrix = self.snet(sF.unsqueeze(0).transpose(1, 2))
            #     # G_s = self.fc_s(torch.mm(sMatrix, sMatrix))
            #     # G_c = self.fc_c(torch.mm(cMatrix, cMatrix))
            #     #
            #     # transmatrix = torch.mm(G_s, G_c)
            #     # # transmatrix = torch.mm(sMatrix, cMatrix)
            #     # transfeature = torch.mm(cF, transmatrix)
            #     # total_e_new = transfeature
            #     e_feature.append(total_e_new)
            #     concatenated_feature_final[i, :length, :] = torch.cat(
            #         (features_t_one, total_e_new),
            #         dim=0)
            # elif index_action[0] == 0:
            #
            #     for order in range(total_e.size(0) - 1, max(total_e.size(0) - self.sliding, 0), -1):
            #         total_t_interpolation_after[order] = features_t_one[index_action[0]]
            #     before = torch.zeros(total_e.size(0), 1).cuda()
            #     after = torch.zeros(total_e.size(0), 1).cuda()
            #     for order in range(0, self.sliding):
            #         before[order] = (-order - 1)
            #         after[-(order + 1)] = (-order - 1)
            #     # after_beta = after * self.params_a
            #     after_beta = after * self.params_b
            #     total_e_factor = torch.ones(total_e.size(0), 1).cuda()
            #     if total_e.size(0) > self.sliding:
            #         total_e_factor[-self.sliding:] = 1 - (torch.exp(after_beta)[-self.sliding:])
            #     total_e_new = total_t_interpolation_after * torch.exp(after_beta) + total_e * total_e_factor
            #     ##### style
            #     # cFF = total_e_new
            #     # sFF = Tframewise[i]
            #     # cMean = torch.mean(cFF, dim=1, keepdim=True)
            #     # cMean = cMean.expand_as(cFF)
            #     # cF = cFF - cMean
            #     # sMean = torch.mean(sFF, dim=1, keepdim=True)
            #     # sMean = sMean.expand_as(sFF)
            #     #
            #     # sF = sFF - sMean
            #     # cMatrix = self.cnet(cF.unsqueeze(0).transpose(1, 2))
            #     # sMatrix = self.snet(sF.unsqueeze(0).transpose(1, 2))
            #     #
            #     # G_s = self.fc_s(torch.mm(sMatrix, sMatrix))
            #     # G_c = self.fc_c(torch.mm(cMatrix, cMatrix))
            #     #
            #     # transmatrix = torch.mm(G_s, G_c)
            #     # transfeature = torch.mm(cF, transmatrix)
            #     # total_e_new = transfeature
            #     e_feature.append(total_e_new)
            #     concatenated_feature_final[i, :length, :] = torch.cat(
            #         (total_e_new, features_t_one),
            #         dim=0)
            # else:
            #
            #     for order in range(total_e.size(0) - 1, max(total_e.size(0) - self.sliding, 0), -1):
            #         total_t_interpolation_after[order] = features_t_one[index_action[0]]
            #     for order in range(0, min(self.sliding, total_e.size(0))):
            #         total_t_interpolation_before[order] = features_t_one[index_action[0] - 1]
            #
            #     before = torch.zeros(total_e.size(0), 1).cuda()
            #     after = torch.zeros(total_e.size(0), 1).cuda()
            #     for order in range(0, self.sliding):
            #         before[order] = (-order - 1)
            #         after[-(order + 1)] = (-order - 1)
            #     before_beta = before * self.params_a
            #     after_beta = after * self.params_b
            #     total_e_factor = torch.ones(total_e.size(0), 1).cuda()
            #     if total_e.size(0) > self.sliding:
            #         total_e_factor[:self.sliding] = 1 - (torch.exp(before_beta)[:self.sliding])
            #         total_e_factor[-self.sliding:] = 1 - (torch.exp(after_beta)[-self.sliding:])
            #
            #     total_e_new = total_t_interpolation_before * torch.exp(
            #         before_beta) + total_t_interpolation_after * torch.exp(after_beta) + total_e * total_e_factor
            #     ##### style
            #     # cFF = total_e_new
            #     # sFF = Tframewise[i]
            #     # cMean = torch.mean(cFF, dim=1, keepdim=True)
            #     # cMean = cMean.expand_as(cFF)
            #     # cF = cFF - cMean
            #     # sMean = torch.mean(sFF, dim=1, keepdim=True)
            #     # sMean = sMean.expand_as(sFF)
            #     # sF = sFF - sMean
            #     # cMatrix = self.cnet(cF.unsqueeze(0).transpose(1, 2))
            #     # sMatrix = self.snet(sF.unsqueeze(0).transpose(1, 2))
            #     # G_s = self.fc_s(torch.mm(sMatrix, sMatrix))
            #     # G_c = self.fc_c(torch.mm(cMatrix, cMatrix))
            #     #
            #     # transmatrix = torch.mm(G_s, G_c)
            #     # # transmatrix = torch.mm(sMatrix, cMatrix)
            #     # transfeature = torch.mm(cF, transmatrix)
            #     # total_e_new=transfeature
            #     e_feature.append(total_e_new)
            #     concatenated_feature_final[i, :length, :] = torch.cat(
            #         (features_t_one[:(index_action[0])], total_e_new, features_t_one[(index_action[0]):]), dim=0)


        Comframewise=concatenated_feature_final.transpose(1, 2)

        input_lengths_entire_com = []
        for i in range(batch):
            input_lengths_entire_com.append(Tlen_x[i] + Elen_x[i])
        # input_lengths_entire_com = torch.IntTensor(input_lengths_entire_com).cuda()
        input_lengths_entire_loca = torch.zeros(batch , requires_grad=True).int().cuda()
        for i in range(batch):
                input_lengths_entire_loca[i] = input_lengths_entire_com[i]
        Clen_x=input_lengths_entire_loca


        # 对entity做手语识别
        Comframewise_e = torch.zeros(batch, e_length, 512).cuda()
        for i in range(batch):
            Comframewise_e[i,:Elen_x[i],:]=e_feature[i]
        conv1d_outputs_E_first = self.conv1d_E_first(Comframewise_e.transpose(1, 2), Elen_x)
        # x: T, B, C
        x_E_first = conv1d_outputs_E_first['visual_feat']
        lgt_E_first = conv1d_outputs_E_first['feat_len']
        tm_outputs_E_first = self.temporal_model_E_first(x_E_first, lgt_E_first)
               
        # pred_E = None if self.training \
        #     else self.decoder_E.decode(outputs_E, lgt_E, batch_first=False, probs=False)
        # conv_pred_E = None if self.training \
        #     else self.decoder_E.decode(conv1d_outputs_E['conv_logits'], lgt_E, batch_first=False, probs=False)
        
        # 对拼接视频做手语识别
        # Comframewise=Tframewise_all.transpose(1, 2)
        # Clen_x=Tlen_x_all
        conv1d_outputs_first = self.conv1d_combine_first(Comframewise, Clen_x)
        # x: T, B, C
        x_first = conv1d_outputs_first['visual_feat']
        lgt_first = conv1d_outputs_first['feat_len']
        tm_outputs_first = self.temporal_model_combine_first(x_first, lgt_first)
        





        # 对entity做手语识别
        tm_outputs_fusion_e = tm_outputs_E_first['predictions']
        conv1d_outputs_E_second = self.conv1d_E_second(tm_outputs_fusion_e.permute(1,2,0), lgt_E_first)
        # x: T, B, C
        x_E_second = conv1d_outputs_E_second['visual_feat']
        lgt_E_second = conv1d_outputs_E_second['feat_len']
        tm_outputs_E_second = self.temporal_model_E_second(x_E_second, lgt_E_second)
        outputs_E = self.classifier_E(tm_outputs_E_second['predictions'])

        ### stage2
        tm_outputs_fusion=tm_outputs_first['predictions']
        conv1d_outputs_second = self.conv1d_combine_second(tm_outputs_fusion.permute(1,2,0), lgt_first)
        # x: T, B, C
        x_second = conv1d_outputs_second['visual_feat']
        lgt_second = conv1d_outputs_second['feat_len']
        tm_outputs_second = self.temporal_model_combine_second(x_second, lgt_second)
        #
        outputs = self.classifier_combine(tm_outputs_second['predictions'])


        pred = self.decoder_combine.decode(outputs, lgt_second, batch_first=False, probs=False)
        # pred = self.decoder_combine.decode(outputs_T, lgt_T_second, batch_first=False, probs=False)
        # pred_no_text=self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='no_text')
        # greedyPredTE=self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='max')


        conv_pred = self.decoder_combine.decode(conv1d_outputs_second['conv_logits'], lgt_second, batch_first=False, probs=False)
        # conv_pred = self.decoder_combine.decode(conv1d_outputs_T_second['conv_logits'], lgt_T_second, batch_first=False,
        #                                         probs=False)
        return {
            "framewise_features": Comframewise,
            "visual_features": x_second,
            "feat_len": lgt_second,
            "conv_logits": conv1d_outputs_second['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,

            # "framewise_features": Tframewise,
            # "visual_features": x_T_second,
            # "feat_len_T":lgt_T_second,
            # "conv_logits_T":conv1d_outputs_T_second['conv_logits'],
            # "sequence_logits_T":outputs_T,


            "feat_len_E":lgt_E_second,
            "conv_logits_E":conv1d_outputs_E_second['conv_logits'],
            "sequence_logits_E":outputs_E,

            # "feature_T_disentangle":feature_T,
            # "feature_E_disentangle":feature_E,
            # "feature_T_disentangle_complement":feature_T_complement,
            # "feature_E_disentangle_complement":feature_E_complement,
            # "feat_len_disentangle":lgt_first


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

        #T识别
        # conv1d_outputs_T_first = self.conv1d_T_first(framewise.transpose(1, 2), len_x)
        # # x: T, B, C
        # x_T_first = conv1d_outputs_T_first['visual_feat']
        # lgt_T_first = conv1d_outputs_T_first['feat_len']
        # tm_outputs_T_first = self.temporal_model_T_first(x_T_first, lgt_T_first)

        # conv1d_outputs_E_first = self.conv1d_E_first(framewise.transpose(1, 2), len_x)
        # # x: T, B, C
        # x_E_first = conv1d_outputs_E_first['visual_feat']
        # lgt_E_first = conv1d_outputs_E_first['feat_len']
        # tm_outputs_E_first = self.temporal_model_E_first(x_E_first, lgt_E_first)

        # 整体识别 
        conv1d_outputs_first = self.conv1d_combine_first(framewise.transpose(1, 2), len_x)
        # x: T, B, C
        x_first = conv1d_outputs_first['visual_feat']
        lgt_first = conv1d_outputs_first['feat_len']
        tm_outputs_first = self.temporal_model_combine_first(x_first, lgt_first)




        # for i in range(batch):          
             
        #     plt.figure()    
        #     mask_to_print=mask1[i].cpu().detach().numpy()
        #     figure_name='/disk1/shipeng/vac_insert_multitask/只有首尾拼接_不共享参数_disentangle_3_test/20220124/'+str(random.randint(0,30))+'testmask1'+'.png'
        #     plt.plot(mask_to_print)
        #     plt.savefig(figure_name)
        #     plt.close()

        #     plt.figure()    
        #     mask_to_print=mask2[i].cpu().detach().numpy()
        #     figure_name='/disk1/shipeng/vac_insert_multitask/只有首尾拼接_不共享参数_disentangle_3_test/20220124/'+str(random.randint(0,30))+'testmask2'+'.png'
        #     plt.plot(mask_to_print)
        #     plt.savefig(figure_name)
        #     plt.close()
            

        # Second step




        # 整体识别
        tm_outputs_fusion=tm_outputs_first['predictions']
        conv1d_outputs_second = self.conv1d_combine_second(tm_outputs_fusion.permute(1,2,0), lgt_first)
        # x: T, B, C
        x_second = conv1d_outputs_second['visual_feat']
        lgt_second = conv1d_outputs_second['feat_len']
        tm_outputs_second = self.temporal_model_combine_second(x_second, lgt_second)

        outputs = self.classifier_combine(tm_outputs_second['predictions'])


        pred = None if self.training \
            else self.decoder_combine.decode(outputs, lgt_second, batch_first=False, probs=False)
        # pred_T=self.decoder_T.decode(outputs_T, lgt_T_second, batch_first=False, probs=False)
        # pred_E=self.decoder_E.decode(outputs_E, lgt_E_second, batch_first=False, probs=False)

        # pred_no_text=self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='no_text')
        
        # greedyPred=self.decoder_combine.decode(outputs, lgt, batch_first=False, probs=False,search_mode='max')
        # greedyPred_conv=self.decoder_combine.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False,search_mode='max')

        conv_pred = None if self.training \
            else self.decoder_combine.decode(conv1d_outputs_second['conv_logits'], lgt_second, batch_first=False, probs=False)
        
        # print('greedyPred: ',greedyPred)
        # print('pred_no_text: ',pred_no_text)
        # # print('conv_pred:',conv_pred)
        # result_file=open('/disk1/shipeng/vac_insert/TE联合训练_E不拆_都只用两层卷积_去除空白帧_interval2_只有首尾拼接_不共享参数_test/result.txt','a')
        
        print('pred:', pred)
        # print('pred_T:', pred_T)
        # print('pred_E:', pred_E)
        
        label_index=0
        print('label:')
        for i in range(0,batch):
            label_list_temp=[self.gloss_dict_word2index[j.item()] for j in label[label_index:label_index+label_lgt[i]]]
            print(label_list_temp)
            # result_file.write('pred:'+ str(pred[i]) +'\n')
            # result_file.write('pred_T:'+ str(pred_T[i]) +'\n')
            # result_file.write('pred_E:'+ str(pred_E[i]) +'\n')
            # result_file.write('label: \n')
            # result_file.write(str(label_list_temp)+'\n')
            # result_file.write('---------- \n')
        # result_file.close()
        
        # print('label:' ,label)
        # print('label_lgt:',label_lgt)
        print('-----')
        return {
            "framewise_features": framewise,
            "visual_features": x_second,
            "feat_len": lgt_second,
            "conv_logits": conv1d_outputs_second['conv_logits'],
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
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                # if epoch>2:
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                # if epoch>2:
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            # elif k=="ConvCTC_T":
            #     loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits_T"].log_softmax(-1),
            #                                           T_label.cpu().int(), ret_dict["feat_len_T"].cpu().int(),
            #                                           T_label_lgt.cpu().int()).mean()
            # elif k == 'SeqCTC_T':
            #     loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits_T"].log_softmax(-1),
            #                                           T_label.cpu().int(), ret_dict["feat_len_T"].cpu().int(),
            #                                           T_label_lgt.cpu().int()).mean()
            # elif k == 'Dist_T':
            #     loss += weight * self.loss['distillation'](ret_dict["conv_logits_T"],
            #                                                ret_dict["sequence_logits_T"].detach(),
            #                                                use_blank=False)
            # elif k=="ConvCTC_E":
            #     loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits_E"].log_softmax(-1),
            #                                           E_label.cpu().int(), ret_dict["feat_len_E"].cpu().int(),
            #                                           E_label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC_E':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits_E"].log_softmax(-1),
                                                      E_label.cpu().int(), ret_dict["feat_len_E"].cpu().int(),
                                                      E_label_lgt.cpu().int()).mean()
            # elif k == 'Dist_E':
            #     loss += weight * self.loss['distillation'](ret_dict["conv_logits_E"],
            #                                                ret_dict["sequence_logits_E"].detach(),
            #                                                use_blank=False)
            # elif k == 'Contractive_T':
            #     loss += weight * self.loss['contrastive'](ret_dict["feature_T_disentangle"],
            #                                                ret_dict["feature_T_disentangle_complement"],
            #                                                ret_dict["feature_E_disentangle_complement"],
            #                                                ret_dict["feature_E_disentangle"],
            #                                                )
            # elif k == 'Contractive_E':
            #     loss += weight * self.loss['contrastive'](ret_dict["feature_E_disentangle"],
            #                                                ret_dict["feature_E_disentangle_complement"],
            #                                                ret_dict["feature_T_disentangle"],
            #                                                ret_dict["feature_T_disentangle_complement"],
            #                                                )
            # elif k == 'Dist_disentangle_C':
            #     loss += weight * self.loss['distillation_disentangle'](ret_dict["feature_T_disentangle"]+ret_dict["feature_E_disentangle_complement"],
            #                                                ret_dict["feature_T_disentangle_complement"].detach()+ret_dict["feature_E_disentangle"].detach(),
            #                                                use_blank=True
            #                                                )
            
        # if np.isinf(loss.item()) or np.isnan(loss.item()):
        #     print('error')
        # for k, weight in self.loss_weights.items():
        #     if k == 'ConvCTC':
        #         temp1 = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
        #                                               label.cpu().int(), ret_dict["feat_len"].cpu().int(),
        #                                               label_lgt.cpu().int()).mean()
        #     elif k == 'SeqCTC':
        #         temp2 = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
        #                                               label.cpu().int(), ret_dict["feat_len"].cpu().int(),
        #                                               label_lgt.cpu().int()).mean()
        #     elif k == 'Dist':
        #         temp3 = weight * self.loss['distillation'](ret_dict["conv_logits"],
        #                                                    ret_dict["sequence_logits"].detach(),
        #                                                    use_blank=False)


        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['distillation_disentangle'] = SeqKD(T=8)
        self.loss['contrastive'] = ContrastiveLoss()
        return self.loss
