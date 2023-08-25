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


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    # model=device.model_to_device(model)
    model.train()
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

        insert_word_before=device.data_to_device(data[10])
        insert_word_after=device.data_to_device(data[11])

        insert_frame_before=data[12]
        insert_frame_after=data[13]
        file_info = data[16]


        right_pad_list_T=None
        right_pad_list_E=None

        # TE_info=[i['original_info'] for i in data[14]]
        # oiginal
        ret_dict = model.forward_train(Tvid, Evid, Tvid_lgt, Evid_lgt, label=label, label_lgt=label_lgt,T_label=T_label,T_label_lgt=T_label_lgt,E_label=E_label,E_label_lgt=E_label_lgt,insert_word_before=insert_word_before,insert_word_after=insert_word_after,insert_frame_before=insert_frame_before,insert_frame_after=insert_frame_after,right_pad_list_T=right_pad_list_T,right_pad_list_E=right_pad_list_E)
        # CTC fenci
        # ret_dict = model.forward_train(Tvid, Evid, Tvid_lgt, Evid_lgt, label=label, label_lgt=label_lgt,T_label=T_label,T_label_lgt=T_label_lgt,E_label=E_label,E_label_lgt=E_label_lgt,insert_word_before=insert_word_before,insert_word_after=insert_word_after,insert_frame_before=insert_frame_before,insert_frame_after=insert_frame_after,right_pad_list_T=right_pad_list_T,right_pad_list_E=right_pad_list_E,file_info=file_info)

        # word_split_file=open('/home/gaoliqing/shipeng/code/VAC_CSLR_TE/DatasetFile/Split_insert_com_T/word_split_T_onlytrainT.txt','a+')
        
        
        # for i in range(Tvid_lgt.size(0)):
        #     # split_result=ret_dict['greedyPred_frame'][i]
        #     split_result_string=''
        #     for j in ret_dict['greedyPred_frame'][i]:
        #         split_result_string+=("".join(list(map(str, j))))
        #         split_result_string+='|'
        #     word_split_file.write(data[8][i]['Tfolder'][0:-6]+'Q'+data[8][i]['Efolder'][0:-6]
        #     +'Q'+data[8][i]['label']+'Q'+data[8][i]['Tlabel']+'Q'+split_result_string+'\n')
        #     word_split_file.write('------------------------------------'+'\n')
            

        # word_split_file.close()
        batch=Tvid.size(0)
        # target_lengths_local = torch.zeros(batch * 2, requires_grad=True).int().cuda()
        # for i in range(batch):
        #     for j in range(2):
        #         target_lengths_local[i * 2 + j] = label_lgt[i]

        # target_batch_loca = torch.zeros((batch * 2, label.size()[-1]),
        #                                 dtype=label.dtype).cuda()
        # for i in range(batch):
        #     for j in range(2):
        #         target_batch_loca[i * 2 + j, :] = label[i]
        # label_new=target_batch_loca
        # label_lgt_new=target_lengths_local
        loss = model.criterion_calculation(ret_dict, label, label_lgt,T_label,T_label_lgt,E_label,E_label_lgt,epoch_idx)


        if np.isinf(loss.item()) or np.isnan(loss.item()):
            # print(loss.item())
            print(data[-1])
            print('loss is nan or inf!')
            continue
        
        optimizer.zero_grad()
        loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        #           ' -->grad_value:', parms.grad)
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
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
    # recoder.print_log("Epoch {}, {} {}".format(epoch, mode, lstm_ret),
    #                   '{}/{}.txt'.format(work_dir, mode))

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
