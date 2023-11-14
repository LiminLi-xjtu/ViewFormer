from torch import nn
import numpy as np
import random
import argparse
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MULTfeature import test_feature
from src import cross_view
from src.eval_metrics import acc
from src.utils import *
from sklearn.linear_model import LogisticRegression

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

parser = argparse.ArgumentParser(description='Cross View Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')



# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=20,
                    help='number of chunks per batch (default: 20)')
parser.add_argument('--position', action='store_true',default=False,
                    help='use the position encoding (default: True)')
parser.add_argument('--projection', action='store_true',default=False,
                    help='use the projection layer (default: True)')

# Logistics
parser.add_argument('--log_interval', type=int, default=20,
                    help='frequency of result logging (default: 20)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

parser.add_argument('--encoder_dim', type=int, default=768)
parser.add_argument('--train_num', type=int, default=15)
parser.add_argument('--val_num', type=int, default=6)
parser.add_argument('--test_num', type=int, default=4)

parser.add_argument('--way', type=int, default=3)
parser.add_argument('--shot', type=int, default=2)

parser.add_argument('--data_view', type=str, default='front')
parser.add_argument('--envs', type=str, default='grass')

parser.add_argument('--view_encoding', action='store_true',default=False,
                    help='use the position encoding (default: True)')
parser.add_argument('--shot_encoding', action='store_true',default=False,
                    help='use the position encoding (default: True)')


parser.add_argument('--test_root', type=str, default='./MULT_dataset/pre_process/test',
                    help='test_dataset root')
parser.add_argument('--model_root', type=str, default='./model',
                    help='the root to save model')



args = parser.parse_args()



def class_dic(c):
    if c == 0:
        return '潜艇'
    if c == 1:
        return '飞机'
    if c == 2:
        return '导弹'
    if c == 3:
        return '坦克'

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            # print("-" * 100)
            # print('meta-test task[{:2d}]:'.format(i_batch + 1))
            # print('对类别进行采样', [class_dic(c) for c in classes])

            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                # print(class_dic(c))
                # print('支持集:', pos[0:shot].cpu().numpy() + 1)
                # print('查询集:', pos[shot:].cpu().numpy() + 1)
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

test_embedding = test_feature(args)

per_class = 15
n_class, way = args.way, args.way
shot = args.shot
seed = args.seed

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(seed)


test_sampler = CategoriesSampler(test_embedding.label.cpu(),args.test_num,n_class,per_class)
test_loader = DataLoader(dataset=test_embedding,
                                  num_workers=0,
                                  batch_sampler=test_sampler,
                                  pin_memory=True)

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768,768,768
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = args.encoder_dim,args.encoder_dim,args.encoder_dim
hyp_params.layers = args.nlevels
hyp_params.use_cuda = True
hyp_params.when = args.when
hyp_params.n_test = len(test_embedding.token_feature)
hyp_params.model = str.upper(args.model.strip())


criterion = nn.CrossEntropyLoss()


def initiate(hyp_params, test_loader):
    models = cross_view
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return test_model(settings, hyp_params, test_loader)



def test_model(settings, hyp_params, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=1.0,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')

    def prepare_label():
        # prepare one-hot label

        support_label = torch.arange(way, dtype=torch.int16).repeat(shot)
        query_label = torch.arange(way, dtype=torch.int8).repeat(per_class - shot)

        support_label = support_label.type(torch.LongTensor)
        query_label = query_label.type(torch.LongTensor)

        if torch.cuda.is_available():
            support_label = support_label.cuda()
            query_label = query_label.cuda()

        return support_label, query_label

    def equal_mean(batch_X, batch_Y, net):
        batch_X = batch_X[:, :, 0, :]  # (45,3,768)
        eval_attr = batch_Y.squeeze(-1)

        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                batch_X, eval_attr = batch_X.cuda(), eval_attr.cuda()

        support_idx, query_idx = (torch.Tensor(np.arange(n_class * shot)).long().view(shot, n_class),
                                  torch.Tensor(np.arange(n_class * shot, n_class * per_class)).long())
        support_feature = batch_X[support_idx.contiguous().view(-1)].contiguous().view(
            *(support_idx.shape + (3, -1)))
        query_feature = batch_X[query_idx.contiguous().view(-1)].contiguous().view(
            *(query_idx.shape + (3, -1)))

        support_label, query_label = prepare_label()
        support_label = support_label.cpu().numpy()

        #single view

        support_feature_view1 = support_feature[:, :, 0, :].view(shot * n_class, -1).cpu().numpy()
        support_feature_view2 = support_feature[:, :, 1, :].view(shot * n_class, -1).cpu().numpy()
        support_feature_view3 = support_feature[:, :, 2, :].view(shot * n_class, -1).cpu().numpy()

        query_feature_view1 = query_feature[:, 0, :].cpu().numpy()
        query_feature_view2 = query_feature[:, 1, :].cpu().numpy()
        query_feature_view3 = query_feature[:, 2, :].cpu().numpy()

        # print(support_feature_view1, support_label)
        clf.fit(support_feature_view1, support_label)
        pred_1 = clf.predict_proba(query_feature_view1)

        clf.fit(support_feature_view2, support_label)
        pred_2 = clf.predict_proba(query_feature_view2)

        clf.fit(support_feature_view3, support_label)
        pred_3 = clf.predict_proba(query_feature_view3)


        #single view + gap view
        support_feature_mean = support_feature.mean(dim = -2).view(shot * n_class, -1).cpu().numpy()
        query_feature_mean = query_feature.mean(dim = -2).cpu().numpy()
        clf.fit(support_feature_mean, support_label)
        pred_4 = clf.predict_proba(query_feature_mean)


        # single view + gap view + cross
        support_feature_cross = net(support_feature)[:, :, :, 0].view(shot * n_class, -1) #(shot * way,768)
        support_feature_cross = F.dropout(support_feature_cross, p=0.2, training=True)
        BN = nn.BatchNorm1d(hyp_params.l_len)
        support_feature_cross = BN(support_feature_cross)
        support_feature_cross = support_feature_cross.cpu().numpy()
        # print(support_feature_cross.shape)

        query_feature_cross = net(query_feature)  # (39,768)
        # query_feature_cross = F.dropout(query_feature_cross, p=0.2, training=True)
        query_feature_cross = BN(query_feature_cross)
        query_feature_cross = query_feature_cross.cpu().numpy()
        # print(query_feature_cross.shape)

        clf.fit(support_feature_cross, support_label)
        pred_5 = clf.predict_proba(query_feature_cross)

        pred_6 = (pred_1 + pred_2 + pred_3) / 3

        if torch.cuda.is_available():
            pred_1 = torch.from_numpy(pred_1).cuda()
            pred_2 = torch.from_numpy(pred_2).cuda()
            pred_3 = torch.from_numpy(pred_3).cuda()
            pred_4 = torch.from_numpy(pred_4).cuda()
            pred_5 = torch.from_numpy(pred_5).cuda()
            pred_6 = torch.from_numpy(pred_6).cuda()


        preds = torch.cat((pred_1.unsqueeze(-1), pred_2.unsqueeze(-1), pred_3.unsqueeze(-1),
                           pred_4.unsqueeze(-1), pred_5.unsqueeze(-1), pred_6.unsqueeze(-1)), -1)

        return preds, query_label

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader
        preds = []
        label = []
        with torch.no_grad():
            for [i_batch, (batch_X, batch_Y)] in enumerate(loader):
                eval_attr = batch_Y.squeeze(-1)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        batch_X, eval_attr = batch_X.cuda(), eval_attr.cuda()
                net = model

                results, truths = equal_mean(batch_X, batch_Y, net)
                y_hat_view1 = torch.argmax(results[:, :, 0], dim=1)
                y_hat_view2 = torch.argmax(results[:, :, 1], dim=1)
                y_hat_view3 = torch.argmax(results[:, :, 2], dim=1)
                y_hat_mean = torch.argmax(results[:, :, 3], dim=1)
                y_hat_cross = torch.argmax(results[:, :, 4], dim=1)
                y_hat_co = torch.argmax(results[:, :, 5], dim=1)

                Acc_view1 = acc(results[:, :, 0], truths, True)
                Acc_view2 = acc(results[:, :, 1], truths, True)
                Acc_view3 = acc(results[:, :, 2], truths, True)
                Acc_mean = acc(results[:, :, 3], truths, True)
                Acc_cross = acc(results[:, :, 4], truths, True)
                Acc_co = acc(results[:, :, 5], truths, True)

                # impro_for_view1 = (Acc_cross - Acc_view1) * 100
                # impro_for_view2 = (Acc_cross - Acc_view2) * 100
                # impro_for_view3 = (Acc_cross - Acc_view3) * 100
                # impro_for_mean  = (Acc_cross - Acc_mean) * 100

                '''print('y_hat_view1', y_hat_view1.cpu().numpy())
                print('y_hat_view2', y_hat_view2.cpu().numpy())
                print('y_hat_view3', y_hat_view3.cpu().numpy())
                print('y_hat_mean ', y_hat_mean.cpu().numpy())
                print('y_hat_cross', y_hat_cross.cpu().numpy())
                print('y_hat_co   ', y_hat_co.cpu().numpy())
                print('y_truth    ', truths.cpu().numpy())

                print(
                    ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_cross {:5.4f} | Accuracy_co {:5.4f} |'.format(
                        Acc_view1, Acc_view2, Acc_view3, Acc_mean, Acc_cross, Acc_co))'''
                # print(
                #     'improvment: \n'
                #     ' |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} |  mean {:5.4f} '.format(
                #         impro_for_view1, impro_for_view2, impro_for_view3, impro_for_mean))

                preds.append(results)
                label.append(truths)
            preds = torch.cat(preds)
            label = torch.cat(label)
            Acc_view1 = acc(preds[:, :, 0], label, True)
            Acc_view2 = acc(preds[:, :, 1], label, True)
            Acc_view3 = acc(preds[:, :, 2], label, True)
            Acc_mean = acc(preds[:, :, 3], label, True)
            Acc_cross = acc(preds[:, :, 4], label, True)
            Acc_co = acc(preds[:, :, 5], label, True)
            impro_for_view1 = (Acc_cross - Acc_view1) * 100
            impro_for_view2 = (Acc_cross - Acc_view2) * 100
            impro_for_view3 = (Acc_cross - Acc_view3) * 100
            impro_for_mean = (Acc_cross - Acc_mean) * 100
            print("-" * 100)
            print('all task in test: \n'
                  ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_cross {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1, Acc_view2, Acc_view3, Acc_mean, Acc_cross, Acc_co))
            print(
                'improvment: \n'
                ' |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} |  mean {:5.4f} '.format(
                    impro_for_view1, impro_for_view2, impro_for_view3, impro_for_mean))

    model = load_model(hyp_params, name=hyp_params.name)

    evaluate(model, criterion, test=True)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

if __name__ == '__main__':
    test_loss = initiate(hyp_params, test_loader)