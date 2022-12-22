import argparse
import os, sys
import os.path as osp
from collections import OrderedDict

import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist, svhn, usps


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def digit_load(args):
    train_bs = args.batch_size
    if args.t == 'm':
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.Lambda(lambda x: x.convert("RGB")),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(32),
                                      transforms.Lambda(lambda x: x.convert("RGB")),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))
    elif args.t == 's':
        train_target = svhn.SVHN_idx('./data/svhn/', split='train', download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
        test_target = svhn.SVHN('./data/svhn/', split='test', download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    elif args.t == 'u':
        train_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.Lambda(lambda x: x.convert("RGB")),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.Lambda(lambda x: x.convert("RGB")),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    dset_loaders = {}
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs * 2, shuffle=False,
                                      num_workers=args.worker, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    # netF = network.LeNetBase().cuda()
    netF_list = [network.DTNBase().cuda() for i in range(len(args.src))]

    w = 2 * torch.rand((len(args.src),)) - 1

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netC_list = [
        network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i
        in range(len(args.src))]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]

    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for k, v in netG_list[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0
    best_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                temp1, temp2 = obtain_label(dset_loaders['target_te'], netF_list[i], netB_list[i], netC_list[i], args)
                temp1 = torch.from_numpy(temp1).cuda()
                temp2 = torch.from_numpy(temp2).cuda()
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        init_ent = torch.zeros(1, len(args.src))

        for i in range(len(args.src)):
            features_test = netB_list[i](netF_list[i](inputs_test))
            outputs_test = netC_list[i](features_test)
            softmax_ = nn.Softmax(dim=1)(outputs_test)
            ent_loss = torch.mean(loss.Entropy(softmax_))
            init_ent[:, i] = ent_loss
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)

        z_2 = torch.sum(weights_all)
        z_ = z_ / z_2

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
            acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netG_list, args)
            log_str = 'Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
            print(log_str + '\n')
            if acc >= best_acc:
                best_acc = acc
                best_F_states = [copy.deepcopy(net.state_dict()) for net in netF_list]
                best_B_states = [copy.deepcopy(net.state_dict()) for net in netB_list]
                best_C_states = [copy.deepcopy(net.state_dict()) for net in netC_list]
                best_G_states = [copy.deepcopy(net.state_dict()) for net in netG_list]
    for net, state in zip(netF_list, best_F_states):
        net.load_state_dict(state)
    for net, state in zip(netB_list, best_B_states):
        net.load_state_dict(state)
    for net, state in zip(netC_list, best_C_states):
        net.load_state_dict(state)
    for net, state in zip(netG_list, best_G_states):
        net.load_state_dict(state)
    distill(netF_list, netB_list, netC_list, netG_list, dset_loaders, args)


def distill(netF_list, netB_list, netC_list, netG_list, dset_loaders, args):

    all_net_lists = netF_list, netB_list, netC_list, netG_list
    for net_list in all_net_lists:
        for i in range(len(args.src)):
            net_list[i].eval()
            for k, v in net_list[i].named_parameters():
                v.requires_grad = False
    netF, netB, netC = student_factory(args.stu_init, netF_list, netB_list, netC_list, netG_list)

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs = iter_source.next()
        except:
            iter_source = iter(dset_loaders["target"])
            inputs = iter_source.next()

        inputs = inputs[0]
        if inputs.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        labels, logits = aggregate_preds(inputs, netF_list, netB_list, netC_list, netG_list)

        inputs, labels, logits = inputs.cuda(), labels.cuda(), logits.cuda()
        labels, logits = labels.detach(), logits.detach()
        outputs = netC(netB(netF(inputs)))
        classifier_loss = nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Distilling, Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


def student_factory(init_strategy, netF_list, netB_list, netC_list, netG_list):
    assert init_strategy in ['scratch', 'most_rel', 'alpha_comb']
    netF = network.DTNBase().cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    with torch.no_grad():
        alphas = [g(torch.ones([1, 1], device='cuda')) for g in netG_list]
    alphas = [alpha / sum(alphas) for alpha in alphas]
    if init_strategy == 'most_rel':
        best_index = torch.cat(alphas).argmax(0)
        netF_state, netB_state, netC_state = netF_list[best_index].state_dict(), \
                                             netB_list[best_index].state_dict(), netC_list[best_index].state_dict()
        netF.load_state_dict(netF_state)
        netB.load_state_dict(netB_state)
        netC.load_state_dict(netC_state)
    if init_strategy == 'alpha_comb':
        netF_state = {k: sum([alphas[i].squeeze() * netF_list[i].state_dict().get(k) for i in range(len(netF_list))])
                      for k in netF_list[0].state_dict().keys()}
        netB_state = {k: sum([alphas[i].squeeze() * netB_list[i].state_dict().get(k) for i in range(len(netB_list))])
                      for k in netB_list[0].state_dict().keys()}
        netC_state = {k: sum([alphas[i].squeeze() * netC_list[i].state_dict().get(k) for i in range(len(netC_list))])
                      for k in netC_list[0].state_dict().keys()}
        netF.load_state_dict(OrderedDict(netF_state))
        netB.load_state_dict(OrderedDict(netB_state))
        netC.load_state_dict(OrderedDict(netC_state))

    return netF, netB, netC


def aggregate_preds(inputs, netF_list, netB_list, netC_list, netG_list):
    with torch.no_grad():
        inputs = inputs.cuda()
        outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
        weights_all = torch.ones(inputs.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

        for i in range(len(args.src)):
            features = netB_list[i](netF_list[i](inputs))
            outputs = netC_list[i](features)
            weights = netG_list[i](features)
            outputs_all[i] = outputs
            weights_all[:, i] = weights.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
        # print(weights_all.mean(dim=0))
        outputs_all = torch.transpose(outputs_all, 0, 1)
        for i in range(inputs.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        all_output = outputs_all_w.float().cpu()

    _, predict = torch.max(all_output, 1)

    return predict, all_output


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str + '\n')
    # return pred_label.astype('int')
    return initc, all_fea


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=str, required=True, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--stu_init', type=str, default='alpha_comb', choices=["alpha_comb", "scratch", "most_rel"])
    parser.add_argument('--output', type=str, default='ckps_digits')
    parser.add_argument('--output_src', type=str, default='ckps_digits')
    args = parser.parse_args()
    args.class_num = 10
    args.src = [x for x in ['s', 'm', 'u'] if x != args.t]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir_src = []
    for s in args.src:
        path = osp.join(args.output_src, 'seed' + str(args.seed), 'source', s)
        args.output_dir_src.append(path)
    print(args.output_dir_src)
    args.output_dir = path = osp.join(args.output_src, 'seed' + str(args.seed), 'adapt_multi', args.t)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)
