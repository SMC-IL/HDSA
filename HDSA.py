import copy
import sys
sys.path.append('/media/zxr/DATA1/lzy/XinCL')
sys.path.append('../XincL_local')
sys.path.append('../XinCL')
import scipy
import time
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from modules.model import PETCGDNN, PETCGDNN2,MCLDNN, DAE, SCF_CNN_16,CLDNN, Wir_CNN,SupConMCLDNN, Wir_CNN_Conv, feat_bottleneck, LinearClassifier, Classifier_SCF, Classifier_CLDNN,MCLDNN_SVD_Conv, Classifier_MCLDNN, Classifier_Wir, PETCGDNN_SVD_Conv, SCF_CNN_16_Conv, LSTMModel, GRUModel, MCLDNN_SVD, CLDNN_SVD_Conv, PETCGDNN_SVD, CLDNN_SVD, ICAMC_SVD_Conv, orthogonal_loss, ReverseLayerF, cosine_similarity_loss
from data.dataset_sample_inc import splitRML2016A, splitDomainRML2016A, JointDataset, loadRML2016A, load_six_data, load_six_data_six_zq, loadDomainRML2018_SVD, loadDomainRML2016A, loadDomainRML2016A_SVD, load_SCF_SVD
from utils.util import plot_confusion_matrix, FocalLoss, setup_seed
from sklearn.metrics import confusion_matrix
import os
from utils.scheduler import PolynomialLR, WarmupLR
import tqdm
import matplotlib.pyplot as plt
from buf_inc_Buf.save_buffer import load_buffer, save_buffer
from torch import optim
from scipy import stats
import torch.nn.functional as F
import random
import argparse
from thop import clever_format, profile

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

proto_all_domain, proto_all_domain_2 = {}, {}
outs = []
max_proto = 0
mean_proto = 1e10





def evaluate_model_stats(model, One_flag=False, batch_size=512):
    """
    一次性计算模型的 MACs, 参数量, 推理显存 和 训练显存
    注意：显存测试依赖 GPU 环境
    """
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，显存测试将跳过或不准确。建议在 CUDA 环境下运行。")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # 1. 确保模型在正确的设备上
    model = model.to(device)
    
    # 2. 构造正确维度的输入数据 (修复了之前的 Shape Mismatch 问题)
    # input1 形状: [Batch_Size, Channels, Height, Width]
    # input1 = torch.randn(batch_size, 1, 2, 1024).to(device)
    # input2 = torch.randn(batch_size, 1, 1024).to(device)
    # input3 = torch.randn(batch_size, 1, 1024).to(device)
    # input1 = torch.randn(batch_size, 1, 16, 16).to(device)
    # input2 = torch.randn(batch_size, 16, 16).to(device)
    # input3 = torch.randn(batch_size, 16, 16).to(device)
    input1 = torch.randn(batch_size, 1, 2, 4096).to(device)
    if One_flag:
        inputs = (input1,)
    else:
        inputs = (input1, input2, input3)

    print("="*50)
    print("开始评估模型...")
    print(f"Batch Size: {batch_size}")
    print("="*50)

    # ---------------------------------------------------------
    # 第一部分：计算静态指标 (MACs & Params) 使用 thop
    # ---------------------------------------------------------
    print("[1/3] 正在分析计算量 (MACs) 和参数量...")
    # 置为 eval 模式以保证算力评估不受 dropout 等随机层影响
    model.eval() 
    
    # verbose=False 可以关闭烦人的逐层打印
    macs, params = profile(model, inputs=inputs, verbose=False)
    GFLOPs, all_GFLOPs = macs*2, macs*4  # 1 MAC ≈ 2 FLOPs，训练阶段总计算量约为前向传播的 2 倍
    macs_str, params_str, GFLOPs_str, all_GFLOPs_str = clever_format([macs, params, GFLOPs, all_GFLOPs], "%.3f")

    if device.type == 'cuda':
        # ---------------------------------------------------------
        # 第二部分：计算【推理阶段】峰值显存
        # ---------------------------------------------------------
        print("[2/3] 正在测试纯推理 (Inference) 显存占用...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(*inputs)
            
        inference_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # 转换为 MB

        # ---------------------------------------------------------
        # 第三部分：计算【训练阶段】峰值显存 (包含激活值和梯度)
        # ---------------------------------------------------------
        print("[3/3] 正在测试训练 (Training) 显存占用...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model.train() # 切换回训练模式
        
        # 强制要求计算梯度以模拟真实的训练显存缓存
        input1.requires_grad = True
        
        out = model(*inputs)
        loss = out.sum()
        loss.backward() # 反向传播会分配额外的显存用于梯度
        
        training_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        inference_mem = 0
        training_mem = 0

    # ---------------------------------------------------------
    # 打印最终报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🎯 模型综合评估报告")
    print("="*50)
    print(f"总参数量 (Parameters) : {params_str}")
    print(f"总计算量 (MACs)       : {macs_str} (注意: 1 MAC 约等于 2 FLOPs)")
    print(f"前向传播总计算量 (FLOPs)       : {GFLOPs_str} (注意: 1 MAC 约等于 2 FLOPs)")
    print(f"总计算量 (FLOPs)       : {all_GFLOPs_str} (注意: 1 MAC 约等于 2 FLOPs)")

    if device.type == 'cuda':
        print(f"推理峰值显存 (Batch={batch_size}) : {inference_mem:.2f} MB")
        print(f"训练峰值显存 (Batch={batch_size}) : {training_mem:.2f} MB")
    print("="*50)


# 钩子函数
def layer_hook(module, inp, out):
    outs.append(inp[0].data)

# 创建钩子
def create_hook(model):
    for name, module in model.named_modules():
        if ("fc1" in name or "dense1" in name) and 'encoder' not in name:
            handle = module.register_forward_hook(layer_hook)
            return handle

# 删除钩子
def pop_hook(model):
    for name, module in model.named_modules():
        if "lstm1" in name:
            module.register_forward_hook(None)



def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)


def proto_new_loss(proto_new, x):
    proto_new = proto_new.reshape(-1, proto_new.shape[-1])
    # 自身与自身越近越好
    # 计算每一行与其他行的欧几里得距离之和
    distances = torch.cdist(x, x, p=2)  # p=2 表示欧几里得距离
    # 将对角线元素（行与自身的距离）设为0
    distances = distances * (1 - torch.eye(distances.size(0), device=x.device))

    # 自身与其它域越远越好
    sum = 0
    for i in range(proto_new.shape[0]):
        distances_1 = torch.cdist(x, proto_new[i].reshape(1,-1), p=2)  # p=2 表示欧几里得距离
        sum += distances_1.sum()
    if proto_new.shape[0] > 1:
        # 其它域与其它域越远越好
        distances_2 = torch.cdist(proto_new, proto_new, p=2)  # p=2 表示欧几里得距离
        # 将对角线元素（行与自身的距离）设为0
        distances_2 = distances_2 * (1 - torch.eye(distances_2.size(0), device=x.device))
        print((x.shape[0]*proto_new.shape[0])/sum , distances.mean(), proto_new.shape[0]**2/distances_2.sum())
        # return (x.shape[0]*proto_new.shape[0])/sum + a*distances.mean()
        return (x.shape[0]*proto_new.shape[0])/sum + a*distances.mean() +  b*proto_new.shape[0]**2/distances_2.sum()
        # return (x.shape[0]*proto_new.shape[0])/sum + distances.mean()/distances_2.mean()
    else:
        return (x.shape[0]*proto_new.shape[0])/sum + a*distances.mean()


def Ensemble_proto(out_list, logit_list=None, xs=1, return_logit=False):
    distances_list = []
    if logit_list == None:
        logit_list = out_list
    for i in range(0, len(out_list)):
        logit = out_list[i]
        distances = torch.norm(proto_all_domain_2[i].unsqueeze(0) - logit, dim=1)
        distances_list.append(distances.reshape(-1, 1))
    distances = torch.cat(distances_list, dim=1)
    row_max = distances.max(dim=1, keepdim=True).values
    row_min = distances.min(dim=1, keepdim=True).values
    # 计算每个元素与该行最大值之间的差
    print(distances[0], row_max[0], row_min[0])
    differences = row_max - row_min
    differences_2 = distances - row_min
    # 步骤3: 将差值转换为权值，这里我们使用指数函数进行归一化
    weights = differences_2/differences
    # 步骤3: 将差值转换为权值，这里我们使用指数函数进行归一化
    weights = torch.exp(-weights/xs)
    ret_logit = torch.zeros(logit_list[0].shape).cuda()
    for i in range(0, len(logit_list)):
        logit = logit_list[i]
        ret_logit += weights[:, [i]] * softmax_temperature(logit, dim=1, tau=tau)
        # ret_logit += weights[:, [i]] * logit
    predicts = ret_logit.argmax(dim=1, keepdim=True)
    if return_logit:
        return predicts, ret_logit
    return predicts




# -*- coding: utf-8 -*-
#### 参数
# dataset = "SCF"
datasets = [2016, 2018, 'SCF', 'Wir']
dataset = datasets[-1]
if dataset == 2016:
    batchsize = 512
else:
    batchsize = 512
start_epoch = 0
training_epoch = 20
#classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

"""
=== RML2016 数据读取 === 
"""
start = time.time()
path = r'./data/RML2016.10a_dict.pkl'
seed_list = [0, 3407, 2024]
backbone_list = ['MCLDNN', 'PETCGDNN', 'CLDNN', 'ICAMC', 'SCF_CNN_16', 'Wir_CNN']
for backbone in backbone_list[5:6]:
    One_flag = False
    for seed in seed_list:
        setup_seed(seed)
        for ind_dom in range(0, 4, 1):
            setup_seed(seed)

            proto_all_domain, proto_all_domain_2 = {}, {}
            if dataset == 2018:
                # 2018
                num_class, domain_num, sto_rate, amt = 24, 5, 0.001, batchsize//4
                save_path, input_size = r'origin_data/2018_domain_First10_inc4', [2, 1024]
                random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2018_SVD(save_path, domain_num=5, train_bz=batchsize, index_domain=ind_dom)
            elif dataset == 2016:
                # 2016
                num_class, domain_num, sto_rate, amt = 11, 49, 0.005, batchsize//4
                save_path, input_size = r'origin_data/First8_domain_num_49_7_zq', [2, 128]
                random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                    val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = loadDomainRML2016A_SVD(
                    save_path, domain_num=domain_num, train_bz=batchsize, index_domain=ind_dom)
            elif dataset == 'SCF':
                num_class, domain_num, sto_rate, amt = 4, 5, 0.005, batchsize // 4
                save_path = r'data/SCF'
                random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader, \
                    test_loader, inc_train_loaders, inc_test_loaders = load_SCF_SVD(save_path, domain_num=5, train_bz=batchsize, index_domain=ind_dom, len_one=16)
                
            elif dataset == 'Wir':
                num_class, domain_num, sto_rate, amt = 3, 5, 0.005, batchsize // 4
                save_path = r'data/Wir/wir'
                random_domain, train_dataset, val_dataset, test_dataset, inc_train_datasets, inc_val_datasets, inc_test_datasets, train_loader, \
                    val_loader, test_loader, inc_train_loaders, inc_val_loaders, inc_test_loaders = load_six_data_six_zq(
                    save_path, batchsize=512, index_domain=ind_dom)
                if type(train_dataset) == list:
                    train_dataset = JointDataset(train_dataset)
                    val_dataset = JointDataset(val_dataset)
                    test_dataset = JointDataset(test_dataset)
                    
                train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)
            """
            === 增量 ===
            """
            Train = True
            CrossLoss = nn.CrossEntropyLoss()
            accs = np.zeros((domain_num, domain_num))
            if backbone == "MCLDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'check_point/MCLDNN/model_2018/domain_split/First10_inc4/MCLDNN_epoch_23_valAcc_0.9377136752136752.pth')

                else:
                    pre_model = torch.load(
                        r'/media/zxr/DATA1/lzy/XinCL/check_point/MCLDNN/model_2016A/domain_split/First8_domain_num_49_7_zq/MCLDNN_epoch_95_valAcc_0.9280113636363636.pth')
                    
                    # pre_model = torch.load(
                    #     r'check_point/MCLDNN/model_2016A/domain_split/First8_domain_num_5/PETCGDNN_epoch_74_valAcc_0.9217045454545455.pth')
            if backbone == "PETCGDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'check_point/PETCGDNN2/model_2018/domain_split/First10_inc4/PETCGDNN_epoch_28_valAcc_0.9610907610907611.pth')
                else:
                    pre_model = torch.load(
                        r'check_point/PETCGDNN2/model_2016A/domain_split/First8_domain_num_5/PETCGDNN_epoch_195_valAcc_0.9194886363636363.pth')
            if backbone == "CLDNN":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'check_point/CLDNN/model_2018/domain_split/First10_inc4/CLDNN_epoch_170_valAcc_0.7356583231583231.pth')
                else:
                    pre_model = torch.load(
                        r'check_point/CLDNN/model_2016A/domain_split/First8_domain_num_5/CLDNN_epoch_187_valAcc_0.7263636363636363.pth')
                One_flag = True
            if backbone == "ICAMC":
                if dataset == 2018:
                    pre_model = torch.load(
                        r'check_point/MCLDNN/model_2018/domain_split/First10_inc4/MCLDNN_epoch_23_valAcc_0.9377136752136752.pth')
                else:
                    pre_model = torch.load(
                        r'../check_point/ICAMC/model_2016A/domain_split/First8_domain_num_5/ICAMC_epoch_186_valAcc_0.8082386363636364.pth')
                One_flag = True
            if backbone == "SCF_CNN_16":
                pre_model = torch.load(r'check_point/SCF_CNN_16/First7/SCF_CNN_16_epoch_23_valAcc_0.94125.pth')
                One_flag=True
            if backbone == 'Wir_CNN':
                pre_model = torch.load(r'/media/zxr/DATA1/lzy/XinCL/check_point/Wir_CNN/six_zq/Wir_CNN_BN_epoch_99_valAcc_0.7154269972451791.pth')
                One_flag = True

            pre_model.eval()
            pretrained_state_dict = pre_model.state_dict()
            print(pretrained_state_dict.keys())
            encoder_fc_dict = {k: v for k, v in pretrained_state_dict.items() if ('fc' in k or "dense" in k) and 'encoder' in k}
            fc_dict = {k: v for k, v in pretrained_state_dict.items() if ('fc' in k or "dense" in k) and 'encoder' not in k}
            if backbone == "ICAMC":
                fc_dict = {k: v for k, v in pretrained_state_dict.items() if 'dense' in k and 'encoder' not in k}
            conv2d_names_w, conv2d_names_b, conv2d_weights, conv2d_bias = [], [], [], []
            for name, module in pre_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv2d_names_w.append(name + '.weight')
                    conv2d_names_b.append(name + '.bias')
                    conv2d_weights.append(module.weight.data)
                    conv2d_bias.append(module.bias.data)
            zip1 = zip(conv2d_names_w, conv2d_weights)
            zip2 = zip(conv2d_names_b, conv2d_bias)
            # 使用dict函数将元组列表转换为字典
            conv_2d_dict, conv_2d_bias_dict = dict(zip1), dict(zip2)
            print(list(conv_2d_dict.values())[-1].shape, list(conv_2d_bias_dict.values())[-1].shape)
            xs = 1
            if backbone == "MCLDNN":
                if dataset == 2016:
                    a, b, lr, tau = 1, 10, 1e-1, 5
                else:
                    a, b, lr, tau = 1, 10, 1e-1, 5
                model = MCLDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
                pre_class = Classifier_MCLDNN()
            if backbone == "PETCGDNN":
                if dataset == 2016:
                    a, b, lr, tau = 100, 100, 1e-1, 5
                else:
                    a, b, lr, tau = 1, 1, 1e-1, 1
                model = PETCGDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, encoder_fc_dict=encoder_fc_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
                pre_class = Classifier_MCLDNN()
            if backbone == "CLDNN":
                if dataset == 2016:
                    a, b, lr, tau = 10, 10, 1e-2, 1
                else:
                    a, b, lr, tau = 1, 10, 1e-1, 5
                pre_class = Classifier_CLDNN()
                model = CLDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
            if backbone == "ICAMC":
                a, b, lr, xs = 1, 1e-2, 1e-1, 1
                model = ICAMC_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
            if backbone == "SCF_CNN_16":
                a, b, lr, xs, tau = 1e0, 1e0, 1e-1, 1, 1
                model = SCF_CNN_16_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
                pre_class = Classifier_SCF()
            if backbone == "Wir_CNN":
                a, b, lr, xs, tau = 1e2, 1e1, 1e-1, 1, 1
                model = Wir_CNN_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=1, classes=num_class)
                pre_class = Classifier_Wir()
            model_state_dict = model.state_dict()
            new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
            for name, param in model.named_parameters():
                print(name, param.data.shape)
            print(new_state_dict.keys())

            model.load_state_dict(new_state_dict, strict=False)

            # 统计模型参数
            total_params = sum(p.numel() for p in model.parameters())

            print(f'{total_params:,} total parameters.')
            print(f'{total_params / (1024):.2f}K total parameters.')
            model.cuda()
            model.eval()
            
            # 计算内存与FLOPs
            print('-' * 20 + f"增量第0个域计算FLOPs和内存" + '-' * 20)
            # evaluate_model_stats(model, One_flag=One_flag, batch_size=batchsize)






            print('-'*20+'预训练模型测试所有域数据'+'-'*20)
            # 预训练模型先测试一轮，保存accs
            
            for i in range(domain_num):
                all_predicts = torch.empty(0, 1).cuda()
                all_targets = torch.empty(0).cuda()
                if i == 0:
                    test_L = test_loader
                    test_D = test_dataset
                else:
                    test_L = inc_test_loaders[i - 1]
                    test_D = inc_test_datasets[i - 1]
                with torch.no_grad():
                    for imgs, targets, snr, _, _ in test_L:
                        st_time = time.time()
                        imgs = imgs.cuda().float()
                        targets = targets.cuda().long()

                        imgs1 = imgs[:, :, 0, :]
                        imgs2 = imgs[:, :, 1, :]
                        if One_flag:
                            outputs = model(imgs)
                        else:
                            outputs = model(imgs, imgs1, imgs2)
                        predicts = outputs.argmax(dim=1, keepdim=True)
                        all_targets = torch.cat([all_targets, targets])
                        all_predicts = torch.cat([all_predicts, predicts], dim=0)

                        end_time = time.time()
                    correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
                    accuracy_ = correct_ / float(len(test_D))
                    accs[0][i] = accuracy_ * 100
            print(accs[0])
            pre_model = copy.deepcopy(model)
            print('-' * 20 + '保存第' + str(1) + '个域数据原型' + '-' * 20)
            proto_train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)
            # 创建钩子函数
            with torch.no_grad():
                outputs_list = []
                outs = []
                for imgs, targets, snr, dlabel, _ in proto_train_loader:
                    handle = create_hook(model)
                    imgs = imgs.cuda().float()
                    targets = targets.cuda().long()
                    imgs1 = imgs[:, :, 0, :]
                    imgs2 = imgs[:, :, 1, :]
                    if One_flag:
                        outputs = model(imgs)
                    else:
                        outputs = model(imgs, imgs1, imgs2)
                    for i in range(len(outs)):
                        outs[i] = outs[i].reshape(outs[i].shape[0], -1)
                    
                    outs_proto = torch.cat(outs, dim=1)
                    handle.remove()
                    outputs_list.append(outs_proto)
                    outs = []
            outputs = torch.cat(outputs_list, dim=0)
            proto_all_domain[0] = torch.mean(outputs, dim=0)
            abs_dis = torch.cdist(outputs, proto_all_domain[0].reshape(1, -1), p=2)
            max_proto = abs_dis.max()
            mean_proto = min(mean_proto, abs_dis.mean())
            print(max_proto)
            '''
                增量训练
            '''

            start_time = time.time()
            total_pas = []
            for k in range(1, domain_num):
                pre_model.eval()
                pretrained_state_dict = pre_model.state_dict()
                if backbone == "MCLDNN":
                    model = MCLDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=k+1, classes=num_class)
                    
                if backbone == "PETCGDNN":
                    model = PETCGDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict,
                                              encoder_fc_dict=encoder_fc_dict, fc_dict=fc_dict, domain_num=k+1,
                                              classes=num_class)

                if backbone == "CLDNN":
                    model = CLDNN_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=k+1, classes=num_class)
                if backbone == "ICAMC":
                    model = ICAMC_SVD_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict,
                                           fc_dict=fc_dict, domain_num=k+1, classes=num_class)
                if backbone == "SCF_CNN_16":
                    model = SCF_CNN_16_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict,
                                           fc_dict=fc_dict, domain_num=k+1, classes=num_class)
                if backbone == "Wir_CNN":
                    model = Wir_CNN_Conv(conv_2d_dict=conv_2d_dict, conv_2d_bias_dict=conv_2d_bias_dict, fc_dict=fc_dict, domain_num=k+1, classes=num_class)
                    
                    
                model_state_dict = model.state_dict()
                new_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

                print(new_state_dict.keys())
                model.load_state_dict(new_state_dict, strict=False)

                # # 统计模型参数
                total_params1 = sum(p.numel() for p in model.parameters())
                # total_params += sum(p.numel() for p in model.buffers())
                total_pas.append(total_params1)
                print(f'{total_params1:,} total parameters.')
                print(f'{total_params1 / (1024):.2f}K total parameters.')
                print(f'Inc:{(total_params1-total_params)/1024:.2f}K.')
                print(f'Inc(%):{(total_params1-total_params) / total_params * 100:.2f}.')
                if k >= 2:
                    print(f'Inc:{(total_pas[len(total_pas)-1] - total_pas[len(total_pas)-2]) / 1024:.2f}K.')
                    print(f'Inc(%):{(total_pas[len(total_pas)-1] - total_pas[len(total_pas)-2]) / total_pas[0] * 100:.2f}.')

                best_acc = 0
                print('-' * 20 + f"增量第{k}个域开始训练" + '-' * 20)

                for name, param in model.named_parameters():
                    param.requires_grad = False
                for name, param in model.named_parameters():
                    if 'SS_'+str(k) in name:
                        param.requires_grad = True
                    if 'bias_'+str(k) in name and 'encoder' not in name:
                        param.requires_grad = True

                model.cuda()
                pre_model = copy.deepcopy(model)
                pre_model.eval()
                for name, param in model.named_parameters():
                    print(name, param.requires_grad, param.shape, param.device)
                
                if Train:

                    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.99), weight_decay=1e-5)
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

                    # 计算内存与FLOPs
                    print('-' * 20 + f"增量第{k}个域计算FLOPs和内存" + '-' * 20)
                    evaluate_model_stats(model=Wir_CNN(), One_flag=One_flag, batch_size=batchsize)

                    CrossLoss = nn.CrossEntropyLoss()#.cuda()

                    correct = torch.zeros(1).squeeze().cuda()
                    correct_ = list(0. for i in range(num_class))
                    epochs = []
                    train_losses = []
                    train_accs = []
                    val_losses = []
                    val_accs = []
                    best_acc = 0


                    for epoch in range(start_epoch+1, training_epoch+1, 1):
                        # 模型训练
                        model.train()
                        with tqdm.tqdm(inc_train_loaders[k-1], unit="batch") as tepoch:
                            for idx, (data, target, snr, dlabel, _) in enumerate(tepoch):
                                model.train()
                                tepoch.set_description('Epoch:'+ str(epoch))
                                data, target, dlabel = data.cuda().float(), target.cuda().long(), dlabel.cuda().long()   # Data to device
                                dd = dlabel[0]
                                # 获取IQ序列的单独向量
                                data1 = data[:, :, 0, :].cuda().float()
                                data2 = data[:, :, 1, :].cuda().float()
                                if One_flag:
                                    output = model(data)
                                else:
                                    output = model(data, data1, data2)
                                
                                losstr = CrossLoss(output, target)
                                optimizer.zero_grad()
                                losstr.backward()
                                optimizer.step()
                                predict_ = output.argmax(dim=1, keepdim=True)
                                correct = predict_.eq(target.view_as(predict_)).sum().item()
                                accuracy = correct/len(data)
                                tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy))
                                outs = []
                        if (epoch + 1) % 5 == 0:
                            scheduler.step()
                        epochs.append(epoch)
                        train_losses.append(losstr.item())
                        train_accs.append(accuracy)
                        model.eval()
                    if True:
                        pre_class.cuda()
                        for name, param in pre_class.named_parameters():
                            param.requires_grad = True
                        optimizer_class = torch.optim.AdamW(filter(lambda p: p.requires_grad, pre_class.parameters()), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-5)
                        for epoch in range(0, 5, 1):
                            # 模型训练
                            pre_class.train()
                            with tqdm.tqdm(inc_train_loaders[k-1], unit="batch") as tepoch:
                                for idx, (data, target, snr, dlabel, _) in enumerate(tepoch):
                                    
                                    outs = []
                                    handle = create_hook(model)
                                    tepoch.set_description('Epoch:'+ str(epoch))
                                    data, target, dlabel = data.cuda().float(), target.cuda().long(), dlabel.cuda().long()   # Data to device
                                    dd = dlabel[0]
                                    # 获取IQ序列的单独向量
                                    data1 = data[:, :, 0, :].cuda().float()
                                    data2 = data[:, :, 1, :].cuda().float()
                                    for j in range(k+1):
                                        if One_flag:
                                            logit = model.forward(data, pos=j)
                                        else:
                                            logit = model.forward(data, data1, data2, pos=j)
                                    for aa in range(len(outs)):
                                        outs[aa] = outs[aa].reshape(outs[aa].shape[0], -1)
                                    ps = list(proto_all_domain.values())
                                    proto_new = torch.cat([ps[i].reshape(1, -1) for i in range(len(ps))]).cuda()
                                    proto_new = pre_class(proto_new)
                                    outs_input = pre_class(torch.cat(outs, dim=0).cuda())
                                    handle.remove()
                                    # losstr = CrossLoss(output, target)
                                    loss_cl = proto_new_loss(proto_new, outs_input)
                                    optimizer_class.zero_grad()
                                    loss_cl.backward()
                                    optimizer_class.step()
                                    tepoch.set_postfix(loss=loss_cl.item())
                                    outs = []
                        pre_class.eval()
                        with torch.no_grad():
                            for ccc in range(k):
                                proto_all_domain_2[ccc] = pre_class(proto_all_domain[ccc].reshape(1, -1).cuda()).reshape(-1)
                print('-' * 20 + '保存第'+str(k)+'个域数据原型' + '-' * 20)
                outs = []
                proto_inc_train_loader = DataLoader(inc_train_datasets[k-1], batch_size=batchsize, shuffle=False)
                model.eval()
                pre_class.eval()
                with torch.no_grad():
                    outputs_list, outputs_list_2 = [], []
                    for imgs, targets, snr, dlabel, _ in proto_inc_train_loader:
                        imgs = imgs.cuda().float()
                        targets = targets.cuda().long()
                        imgs1 = imgs[:, :, 0, :]
                        imgs2 = imgs[:, :, 1, :]
                        handle = create_hook(model)
                        if One_flag:
                            outputs = model(imgs)
                        else:
                            outputs = model(imgs, imgs1, imgs2)
                        for i in range(len(outs)):
                            outs[i] = outs[i].reshape(outs[i].shape[0], -1)
                        outs_proto = torch.cat(outs, dim=1)
                        outs_proto_2 = pre_class(outs_proto)
                        handle.remove()
                        print(outs_proto_2.shape)
                        outputs_list.append(outs_proto)
                        outputs_list_2.append(outs_proto_2)
                        outs = []
                outputs = torch.cat(outputs_list, dim=0)
                outputs_2 = torch.cat(outputs_list_2, dim=0)
                proto_all_domain[k] = torch.mean(outputs, dim=0)
                proto_all_domain_2[k] = torch.mean(outputs_2, dim=0)
                
                outs = []
                print('-' * 20 + '增量第'+str(k)+'个域开始测试所有域' + '-' * 20)
                # 模型测试, 保存acc
                for i in range(domain_num):
                    model.eval()
                    pre_class.eval()
                    all_predicts = torch.empty(0, 1).cuda()
                    all_targets = torch.empty(0).cuda()
                    if i == 0:
                        test_L = test_loader
                        test_D = test_dataset
                    else:
                        test_L = inc_test_loaders[i - 1]
                        test_D = inc_test_datasets[i - 1]
                    with torch.no_grad():
                        for imgs, targets, snr, _, _ in test_L:
                            st_time = time.time()
                            imgs = imgs.cuda().float()
                            targets = targets.cuda().long()
                            imgs1 = imgs[:, :, 0, :]
                            imgs2 = imgs[:, :, 1, :]
                            logit_list, out_list = [], []
                            outs_proto_ss = None
                            for j in range(k+1):
                                handle = create_hook(model)
                                if One_flag:
                                    logit = model(imgs, pos=j)
                                else:
                                    logit = model(imgs, imgs1, imgs2, pos=j)
                                for v in range(len(outs)):
                                    outs[v] = outs[v].reshape(outs[v].shape[0], -1)
                                outs_proto = torch.cat(outs, dim=1)
                                outs_proto = pre_class(outs_proto)
                                handle.remove()
                                out_list.append(outs_proto)
                                logit_list.append(logit)
                                outs = []
                            end_time = time.time()
                            predicts = Ensemble_proto(out_list, logit_list, xs=xs)
                            all_targets = torch.cat([all_targets, targets])
                            all_predicts = torch.cat([all_predicts, predicts.reshape(-1, 1).cuda()], dim=0)
                            
                        correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
                        accuracy_ = correct_ / float(len(test_D))
                        accs[k][i] = accuracy_ * 100
                print(f'accs[{k}]:', accs[k])
                pre_model = copy.deepcopy(model)
            end_time = time.time()
            print('增量训练4次，时间为:', end_time-start_time, '秒')
            data_len = np.zeros((domain_num, ))
            for i in range(domain_num):
                if i == 0:
                    test_D = test_dataset
                else:
                    test_D = inc_test_datasets[i - 1]
                data_len[i] = len(test_D)
            print('-' * 20 + f'计算保存FAA和FF' + '-' * 20)
            accs_transfer, accs_trans_sum = accs.copy(), 0
            for i in range(accs_transfer.shape[0]):
                accs_transfer[i, 0:i+1] = 0
            for j in range(1, accs_transfer.shape[1]):
                accs_trans_sum += np.sum(accs_transfer[:, j])/j
            accs_transfer = accs_trans_sum / (domain_num-1)
            accs_avg = np.mean(accs, axis=0)
            print(accs_avg)
            avg = np.mean(accs_avg)
            avg_S = np.sum(accs_avg*data_len)/np.sum(data_len)
            for i in range(accs.shape[0]-1):
                accs[i, i+1:] = 0
            acc_path = os.path.join(r'check_point/MCLDNN_SVD/model/domain_split/First8_domain_num_5/SVD_inc', 'Accuracy.txt')
            FAA_D = accs[-1].sum() / float(domain_num)
            FAA_S = np.sum(accs[-1]*data_len)/np.sum(data_len)
            FF_arr = accs[:-1, :-1] - accs[-1, :-1]
            FF = np.zeros((domain_num-1, ))
            FF_D = 0
            for i in range(FF_arr.shape[1]):
                FF[i] = np.max(FF_arr[:, i])
                FF_D += np.max(FF_arr[:, i])
            FF_S = np.sum(FF*data_len[:-1])/np.sum(data_len[:-1])
            FF_D = FF_D / float(FF_arr.shape[1])
            # 增量平均准确率
            print('准确率矩阵:', accs)
            print('每次增量平均准确率:', np.sum(accs, axis=1).reshape(-1))
            print(FF_arr)
            print(f'{random_domain}:FAA:{"%.2f"%FAA_D}   FF:{"%.2f"%FF_D}    FAA_S:{"%.2f"%FAA_S}   FF_S:{"%.2f"%FF_S}  avg:{"%.2f"%avg} avg_S:{"%.2f"%avg_S} transfer:{"%.2f"%accs_transfer}')

            new_data = f'{backbone}-{str(seed)}-{random_domain}:FAA:{"%.2f" % FAA_D}   FF:{"%.2f" % FF_D}    FAA_S:{"%.2f" % FAA_S}   FF_S:{"%.2f" % FF_S}   avg:{"%.2f"%avg} avg_S:{"%.2f"%avg_S} transfer:{"%.2f"%accs_transfer}'
            filepath = '/output_csv/result.csv'

            if not os.path.exists(filepath):
                # 文件不存在，创建一个包含新数据的单列DataFrame
                df = pd.DataFrame([new_data])
                df.to_csv(filepath, header=False, index=False)  # 写入文件，不包含表头和索引
            else:
                # 文件存在，读取现有数据
                current_data = pd.read_csv(filepath, header=None)  # 读取单列数据
                # 将新数据追加到现有数据的末尾
                updated_data = pd.concat([current_data, pd.Series(new_data)], ignore_index=True)
                # 将更新后的数据写回文件
                updated_data.to_csv(filepath, mode='w', header=False, index=False)

            # 释放显存
            del random_domain, train_dataset, test_dataset, inc_train_datasets, inc_test_datasets, train_loader, \
                test_loader, inc_train_loaders, inc_test_loaders, pre_model, model, pre_class
            torch.cuda.empty_cache()
            time.sleep(5)


