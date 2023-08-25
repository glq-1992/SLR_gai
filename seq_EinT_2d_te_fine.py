import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.autograd import Variable

def seq_train(loader, model, optimizer, optimizer_d, device, epoch_idx, recoder):
    # model=device.model_to_device(model)
    model.train()
    network=['conv2d','conv1d_combine_first','temporal_model_combine_first','conv1d_combine_second','temporal_model_combine_second']
    netwrok=['conv2d','conv1d_combine_first','temporal_model_combine_first']
    for name,param in model.named_parameters():
    #     # if name.startswith("conv2d") or name.startswith("conv1d_combine_first") or name.startswith("temporal_model_combine_first") or name.startswith("conv1d_combine_second") or name.startswith("temporal_model_combine_second"):
    #     if name.startswith("conv2d") or name.startswith("conv1d_combine_first") or name.startswith("temporal_model_combine_first"):
        if name.startswith("G"):
            param.requires_grad=False

    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    for batch_idx, data in enumerate(loader):
        Tvid = device.data_to_device(data[0])
        Evid = device.data_to_device(data[1])

        Tvid_lgt = device.data_to_device(data[2])
        Evid_lgt = device.data_to_device(data[3])

        label = device.data_to_device(data[4])
        label_lgt = device.data_to_device(data[5])

        T_label = device.data_to_device(data[6])
        T_label_lgt = device.data_to_device(data[7])

        E_label = device.data_to_device(data[8])
        E_label_lgt = device.data_to_device(data[9])

        insert_word_before = device.data_to_device(data[10])
        insert_word_after = device.data_to_device(data[11])

        insert_frame_before = data[12]
        insert_frame_after = data[13]
        file_info = data[16]

        right_pad_list_T = None
        right_pad_list_E = None

        # input_true=Cvid
        # input_true_len=Cvid_lgt
        input_flase_t=Tvid
        input_flase_e=Evid
        input_flase_t_len=Tvid_lgt
        input_flase_e_len = Evid_lgt
        input_true_label=label
        input_false_label=label
        label_len_true=label_lgt
        label_len_false=label_lgt

        # input_true=[]
        # input_flase_t=[]
        # input_flase_e = []
        # input_true_label=[]
        # input_false_label = []
        # label_len_true=[]
        # label_len_false = []
        # for i in range(len(Tvid_lgt)):
        #
        #     if Tvid_lgt[i]==Evid_lgt[i]:
        #         input_true.append(Tvid[i])
        #         input_true_label.append(label)
        #         label_len_true.append(label_lgt)
        #     else:
        #         input_flase_t.append(Tvid[i])
        #         input_flase_e.append(Evid[i])
        #         input_false_label.append(label)
        #         label_len_false.append(label_lgt)
        # ret_dict_true = model.forward_train_true(input_true[0].unsqueeze(0), Tvid_lgt, label=label, label_lgt=label_lgt,
        #                                          # 生成true特征
        #                                          )
        # decision_true = model.forward_train_d1(ret_dict_true["clipwise_features"])
        # label_true = Variable(torch.ones([1])).cuda()
        # loss_d1_true = model.criterion_calculation_d1(decision_true["decision"][0], label_true)
        # predict_true = model.forward_train_d2(ret_dict_true)
        # loss_d2_true = model.criterion_calculation_d2(predict_true, input_true_label[0], label_len_true[0],
        #                                               # false特征经过鉴别器2，CTC
        #                                               )
        # ret_dict_false = model.forward_train_false(input_flase_t[0].unsqueeze(0), input_flase_e[0].unsqueeze(0),
        #                                            Tvid_lgt, Evid_lgt, label=label, label_lgt=label_lgt,  # 生成false特征
        #                                            )
        # decision_false = model.forward_train_d1(ret_dict_false["clipwise_features"],
        #                                         )
        # label_false = Variable(torch.zeros([1])).cuda()
        # loss_d1_false = model.criterion_calculation_d1(decision_false["decision"][0], label_false)
        #
        # predict_false = model.forward_train_d2(ret_dict_false)
        # loss_d2_false = model.criterion_calculation_d2(predict_false, input_false_label[0], label_len_false[0],
        #                                                # false特征经过鉴别器2，负CTC
        #                                                )
        #
        # d_loss = loss_d1_true + loss_d1_false + loss_d2_true - loss_d2_false   ####?
        # # d_loss = loss_d1_false  - loss_d2_false
        # if np.isinf(d_loss.item()) or np.isnan(d_loss.item()):
        #     # print(loss.item())
        #     print(data[-1])
        #     print('loss is nan or inf!')
        #     continue
        # optimizer_d.zero_grad()
        # d_loss.backward()
        # optimizer_d.step()
        #
        # # =================train generator
        # ret_dict_false1 = model.forward_train_false(input_flase_t[0].unsqueeze(0), input_flase_e[0].unsqueeze(0),
        #                                             Tvid_lgt, Evid_lgt, label=label, label_lgt=label_lgt,  # 生成false特征
        #                                             )
        # decision_false1 = model.forward_train_d1(ret_dict_false1["clipwise_features"],
        #                                          )
        # label_false1 = Variable(torch.zeros([1])).cuda()
        # loss_d1_false1 = model.criterion_calculation_d1(decision_false1["decision"][0], label_false1)
        #
        # predict_false1 = model.forward_train_d2(ret_dict_false1)
        # loss_d2_false1 = model.criterion_calculation_d2(predict_false1, input_false_label[0], label_len_false[0],
        #                                                 # false特征经过鉴别器2，负CTC
        #                                                 )
        # g_loss = loss_d1_false1 + loss_d2_false1
        #
        # if np.isinf(g_loss.item()) or np.isnan(g_loss.item()):
        #     # print(loss.item())
        #     print(data[-1])
        #     print('loss is nan or inf!')
        #     continue
        #
        # optimizer.zero_grad()
        # g_loss.backward()
        # # for name, parms in model.named_parameters():
        # #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        # #           ' -->grad_value:', parms.grad)
        # # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        # optimizer.step()
        # =================train discriminator
        # d_true or false
        # ret_dict_true = model.forward_train_true(input_true, input_true_len, label=input_true_label, label_lgt=label_len_true,   # 生成true特征
        #                                      )
        # decision_true = model.forward_train_d1(ret_dict_true["clipwise_features"])
        # label_true = Variable(torch.ones(2,1)).cuda()
        # # label_true=Variable(torch.ones([1])).cuda()   #### 2×1？？？？？？
        # loss_d1_true = model.criterion_calculation_d1(decision_true["decision"], label_true)
        # predict_true = model.forward_train_d2(ret_dict_true)
        # loss_d2_true = model.criterion_calculation_d2(predict_true, input_true_label, label_len_true,# false特征经过鉴别器2，CTC
        #                                                 )
        # ret_dict_false = model.forward_train_false(input_flase_t, input_flase_e, input_flase_t_len, input_flase_e_len, label=input_false_label, label_lgt=label_len_false,    # 生成false特征
        #                                )
        # decision_false = model.forward_train_d1(ret_dict_false["clipwise_features"],
        #                                            )
        # # label_false = Variable(torch.zeros([1])).cuda()  #### 2×1？？？？？？
        # label_false = Variable(torch.zeros(2,1)).cuda()  #### 2×1？？？？？？
        # loss_d1_false = model.criterion_calculation_d1(decision_false["decision"], label_false)
        #
        # predict_false = model.forward_train_d2(ret_dict_false)
        # loss_d2_false = model.criterion_calculation_d2(predict_false, input_false_label, label_len_false,   # false特征经过鉴别器2，负CTC
        #                                               )
        # location_false = model.forward_train_d3(ret_dict_false["framewise_features"],
        #                                         )      ######????
        # loss_d3_false = model.criterion_calculation_d3(location_false["d_location"], ret_dict_false["location"],ret_dict_false["T_length"],  #####?????
        #                                                # false特征经过鉴别器2，负CTC
        #                                                )
        # # d_loss = loss_d1_true + loss_d1_false+loss_d2_true+loss_d2_false   #####？？？？
        # # d_loss = loss_d1_true + loss_d1_false + loss_d2_true + loss_d3_false #####final
        # d_loss = loss_d1_true + loss_d1_false + loss_d2_true   #####without location
        # # d_loss = loss_d2_true+loss_d3_false  #####without discriminator
        # # d_loss = loss_d1_true + loss_d1_false + loss_d2_true + loss_d2_false+loss_d3_false  #####？？？？
        # # d_loss = loss_d1_true + loss_d1_false  #####？？？？
        # # d_loss = loss_d1_false  - loss_d2_false
        # if np.isinf(d_loss.item()) or np.isnan(d_loss.item()):
        #     # print(loss.item())
        #     print(data[-1])
        #     print('loss is nan or inf!')
        #     continue
        # optimizer_d.zero_grad()
        # d_loss.backward()
        # optimizer_d.step()

        # =================train generator
        ret_dict_false1 = model.forward_train_false(input_flase_t, input_flase_e, input_flase_t_len, input_flase_e_len, label=input_false_label, label_lgt=label_len_false,    # 生成false特征
                                       )
        # decision_false1 = model.forward_train_d1(ret_dict_false1["clipwise_features"],
        #                                            )
        # label_false1 = Variable(torch.ones([1])).cuda()  #### 2×1？？？？？？
        # label_false1 = Variable(torch.ones(2,1)).cuda()  #### 2×1？？？？？？
        # loss_d1_false1 = model.criterion_calculation_d1(decision_false1["decision"], label_false1)

        predict_false1 = model.forward_train_d2(ret_dict_false1)
        loss_d2_false1 = model.criterion_calculation_d2(predict_false1, input_false_label, label_len_false,   # false特征经过鉴别器2，负CTC
                                                      )
        # location_false1 = model.forward_train_d3(ret_dict_false1["framewise_features"],
        #                                         )  ######????
        # loss_d3_false1 = model.criterion_calculation_d3(location_false1["d_location"], ret_dict_false1["location"],
        #                                                ret_dict_false1["T_length"],  #####?????
        #                                                # false特征经过鉴别器2，负CTC
        #                                                )
        # g_loss = loss_d1_false1 + loss_d2_false1+(1-loss_d3_false1)       ### final
        # g_loss = loss_d1_false1 + loss_d2_false1    ### without location
        g_loss = loss_d2_false1   ### without location, te fine-tune
        # g_loss = loss_d2_false1+(1-loss_d3_false1)  ### without discriminator
        # g_loss = loss_d2_true + loss_d2_false
        # g_loss = loss_d2_false
        if np.isinf(g_loss.item()) or np.isnan(g_loss.item()):
            # print(loss.item())
            print(data[-1])
            print('loss is nan or inf!')
            continue
        
        optimizer.zero_grad()
        g_loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad)
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(g_loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), g_loss.item(), clr[0]))
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    # exit()
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    # stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        file_info = data[4] #### print label

        with torch.no_grad():
            # original
            ret_dict = model.forward_test(vid, vid_lgt, label=label, label_lgt=label_lgt)
            # CTC fenci
            # ret_dict = model.forward_test(vid, vid_lgt, label=label, label_lgt=label_lgt,file_info=file_info)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
        # print('')
        

    python_eval = True if evaluate_tool == "python" else False
    write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
    write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
               total_conv_sent)
    # conv_ret = evaluate(
    #     prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
    #     evaluate_dir=cfg.dataset_info['evaluation_dir'],
    #     evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
    #     output_dir="epoch_{}_result/".format(epoch),
    #     python_evaluate=python_eval,
    # )
    lstm_ret = evaluate(
        prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
        evaluate_dir=cfg.dataset_info['evaluation_dir'],
        evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=python_eval,
        triplet=True,
    )

    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")

    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
