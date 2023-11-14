from torch import nn
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import argparse
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from NUS_WIDE_OBJECT import NUS_WIDE_OBJECTdataset
from sklearn.linear_model import LogisticRegression
from src.four_view_fsl import feature_proj, MULTModel, ViewFormer
from src.eval_metrics import acc
from src.utils import *

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

parser.add_argument('--view_encoding', action='store_true',default=False,
                    help='use the view encoding (default: True)')
parser.add_argument('--shot_encoding', action='store_true',default=False,
                    help='use the shot encoding (default: True)')


# Logistics
parser.add_argument('--log_interval', type=int, default=20,
                    help='frequency of result logging (default: 20)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

parser.add_argument('--encoder_dim', type=int, default=768)
parser.add_argument('--conv_dim_key', type=int, default=512)
parser.add_argument('--conv_dim_actor', type=int, default=512)

parser.add_argument('--train_num', type=int, default=15)
parser.add_argument('--val_num', type=int, default=6)
parser.add_argument('--test_num', type=int, default=4)

parser.add_argument('--view_num', type=int, default=3)
parser.add_argument('--way', type=int, default=3)
parser.add_argument('--shot', type=int, default=2)

parser.add_argument('--root', type=str, default='F:\下载\movies617\my_movies (CAP, REISO)',
                    help='train_dataset root')

parser.add_argument('--model_root', type=str, default='./model',
                    help='the root to save model')

parser.add_argument('--alpha_1', type=float, default=1,
                    help='alpha_1')
parser.add_argument('--alpha_2', type=float, default=1,
                    help='alpha_2')
parser.add_argument('--alpha_3', type=float, default=1,
                    help='alpha_3')
parser.add_argument('--alpha_4', type=float, default=1,
                    help='alpha_4')
parser.add_argument('--alpha_5', type=float, default=1,
                    help='alpha_5')
parser.add_argument('--lamda', type=float, default=1,
                    help='regular term')
parser.add_argument('--beta', type=float, default=1,
                    help='regular term for cross')



args = parser.parse_args()
shot = args.shot
per_class = 15 + shot
n_class, way = args.way, args.way
seed = args.seed

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(seed)

class Sampler():

    def __init__(self, label, label_index, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label.cpu())
        self.m_ind = []
        for i in range(label_index[0], label_index[1]):
            ind = np.argwhere(label == i + 1).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                c_true = c
                l = self.m_ind[c_true]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

multi_view_data = NUS_WIDE_OBJECTdataset(args)
test_sampler = Sampler(multi_view_data.label, [25, 31], args.test_num, way, per_class)
test_loader = DataLoader(dataset=multi_view_data,
                         num_workers=0,
                         batch_sampler=test_sampler,
                         pin_memory=False)

hyp_params = args
hyp_params.orig_d_1, hyp_params.orig_d_2, hyp_params.orig_d_3, hyp_params.orig_d_4, hyp_params.orig_d_5 = 65, 226, 145, 74, 129
# 65, 226, 145, 74, 129
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = args.encoder_dim,args.encoder_dim,args.encoder_dim
hyp_params.layers = args.nlevels
hyp_params.use_cuda = True
hyp_params.when = args.when
hyp_params.model = str.upper(args.model.strip())


criterion = nn.CrossEntropyLoss()


def initiate(hyp_params, test_loader):
    model = ViewFormer(feature_proj(hyp_params), MULTModel(hyp_params))

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

    def equal_mean(view1, view2, view3, view4, view5, batch_Z, net):

        eval_attr = batch_Z.squeeze(-1)

        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                view1, view2, view3, view4, view5, eval_attr = view1.cuda(), view2.cuda(), view3.cuda(), view4.cuda(), view5.cuda(), eval_attr.cuda()

        support_idx, query_idx = (torch.Tensor(np.arange(n_class * shot)).long().view(shot, n_class),
                                  torch.Tensor(np.arange(n_class * shot, n_class * per_class)).long())
        support_view1 = view1[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view1.shape[-1])
        support_view2 = view2[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view2.shape[-1])
        support_view3 = view3[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view3.shape[-1])
        support_view4 = view4[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view4.shape[-1])
        support_view5 = view5[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view5.shape[-1])

        query_view1 = view1[query_idx.contiguous().view(-1)].contiguous().view(-1, view1.shape[-1])
        query_view2 = view2[query_idx.contiguous().view(-1)].contiguous().view(-1, view2.shape[-1])
        query_view3 = view3[query_idx.contiguous().view(-1)].contiguous().view(-1, view3.shape[-1])
        query_view4 = view4[query_idx.contiguous().view(-1)].contiguous().view(-1, view4.shape[-1])
        query_view5 = view5[query_idx.contiguous().view(-1)].contiguous().view(-1, view5.shape[-1])

        support_label, query_label = prepare_label()
        support_label = support_label.cpu().numpy()

        support_feature_view1, support_feature_view2, support_feature_view3, support_feature_view4, support_feature_view5, support_feature_mean, support_feature_cross = \
            net(support_view1, support_view2, support_view3, support_view4, support_view5)
        query_feature_view1, query_feature_view2, query_feature_view3, query_feature_view4, query_feature_view5, query_feature_mean, query_feature_cross = \
            net(query_view1, query_view2, query_view3, query_view4, query_view5)

        support_feature_view1, support_feature_view2, support_feature_view3, support_feature_view4, support_feature_view5 = \
            support_feature_view1.view(shot * n_class, -1).cpu().numpy(), \
            support_feature_view2.view(shot * n_class, -1).cpu().numpy(), \
            support_feature_view3.view(shot * n_class, -1).cpu().numpy(), \
            support_feature_view4.view(shot * n_class, -1).cpu().numpy(), \
            support_feature_view5.view(shot * n_class, -1).cpu().numpy()


        query_feature_view1, query_feature_view2, query_feature_view3, query_feature_view4, query_feature_view5 = \
            query_feature_view1.cpu().numpy(), query_feature_view2.cpu().numpy(), \
            query_feature_view3.cpu().numpy(), query_feature_view4.cpu().numpy(), query_feature_view5.cpu().numpy()

        # print(support_feature_view1, support_label)
        clf.fit(support_feature_view1, support_label)
        pred_1 = clf.predict_proba(query_feature_view1)

        clf.fit(support_feature_view2, support_label)
        pred_2 = clf.predict_proba(query_feature_view2)

        clf.fit(support_feature_view3, support_label)
        pred_3 = clf.predict_proba(query_feature_view3)

        clf.fit(support_feature_view4, support_label)
        pred_4 = clf.predict_proba(query_feature_view4)

        clf.fit(support_feature_view5, support_label)
        pred_5 = clf.predict_proba(query_feature_view5)

        # single view + gap view

        support_feature_mean, query_feature_mean = support_feature_mean.view(shot * n_class,
                                                                             -1).cpu().numpy(), query_feature_mean.cpu().numpy()
        clf.fit(support_feature_mean, support_label)
        pred_6 = clf.predict_proba(query_feature_mean)

        # single view + gap view + cross

        support_feature_cross = support_feature_cross[:, :, :, 0].view(shot * n_class, -1)
        support_feature_cross = F.dropout(support_feature_cross, p=0.2, training=True)
        BN = nn.BatchNorm1d(args.encoder_dim)
        support_feature_cross = BN(support_feature_cross)
        support_feature_cross = support_feature_cross.cpu().numpy()
        # print(support_feature_cross.shape)

        query_feature_cross = BN(query_feature_cross)
        query_feature_cross = query_feature_cross.cpu().numpy()

        clf.fit(support_feature_cross, support_label)
        pred_7 = clf.predict_proba(query_feature_cross)

        pred_8 = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) / 5

        if torch.cuda.is_available():
            pred_1 = torch.from_numpy(pred_1).cuda()
            pred_2 = torch.from_numpy(pred_2).cuda()
            pred_3 = torch.from_numpy(pred_3).cuda()
            pred_4 = torch.from_numpy(pred_4).cuda()
            pred_5 = torch.from_numpy(pred_5).cuda()
            pred_6 = torch.from_numpy(pred_6).cuda()
            pred_7 = torch.from_numpy(pred_7).cuda()
            pred_8 = torch.from_numpy(pred_8).cuda()

        preds = torch.cat((pred_1.unsqueeze(-1), pred_2.unsqueeze(-1), pred_3.unsqueeze(-1),
                           pred_4.unsqueeze(-1), pred_5.unsqueeze(-1), pred_6.unsqueeze(-1),
                           pred_7.unsqueeze(-1), pred_8.unsqueeze(-1)), -1)

        return preds, query_label


    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader
        preds = []
        label = []
        with torch.no_grad():
            for [i_batch, (batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_Z)] in enumerate(loader):
                eval_attr = batch_Z.squeeze(-1)


                net = model
                results, truths = equal_mean(batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_Z, net)
                # y_hat_view1 = torch.argmax(results[:, :, 0], dim=1)
                # y_hat_view2 = torch.argmax(results[:, :, 1], dim=1)
                # y_hat_view3 = torch.argmax(results[:, :, 2], dim=1)
                # y_hat_mean  = torch.argmax(results[:, :, 3], dim=1)
                # y_hat_cross = torch.argmax(results[:, :, 4], dim=1)
                # y_hat_co    = torch.argmax(results[:, :, 6], dim=1)
                #
                #
                # Acc_view1  = acc(results[:, :, 0], truths, True)
                # Acc_view2  = acc(results[:, :, 1], truths, True)
                # Acc_view3  = acc(results[:, :, 2], truths, True)
                # Acc_mean   = acc(results[:, :, 3], truths, True)
                # Acc_cross  = acc(results[:, :, 4], truths, True)
                # Acc_co     = acc(results[:, :, 6], truths, True)

                # impro_for_view1 = (Acc_cross - Acc_view1) * 100
                # impro_for_view2 = (Acc_cross - Acc_view2) * 100
                # impro_for_view3 = (Acc_cross - Acc_view3) * 100
                # impro_for_mean  = (Acc_cross - Acc_mean) * 100

                # print('y_hat_view1', y_hat_view1.cpu().numpy())
                # print('y_hat_view2', y_hat_view2.cpu().numpy())
                # print('y_hat_view3', y_hat_view3.cpu().numpy())
                # print('y_hat_mean ', y_hat_mean.cpu().numpy())
                # print('y_hat_cross', y_hat_cross.cpu().numpy())
                # print('y_hat_co   ', y_hat_co.cpu().numpy())
                # print('y_truth    ', truths.cpu().numpy())
                #
                # print(
                #     ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_cross {:5.4f} | Accuracy_co {:5.4f} |'.format(
                #         Acc_view1, Acc_view2, Acc_view3, Acc_mean, Acc_cross, Acc_co))
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
            Acc_view4 = acc(preds[:, :, 3], label, True)
            Acc_view5 = acc(preds[:, :, 4], label, True)

            Acc_mean = acc(preds[:, :, 5], label, True)
            Acc_cross = acc(preds[:, :, 6], label, True)
            Acc_co = acc(preds[:, :, 7], label, True)
            impro_for_view1 = (Acc_cross - Acc_view1) * 100
            impro_for_view2 = (Acc_cross - Acc_view2) * 100
            impro_for_view3 = (Acc_cross - Acc_view3) * 100
            impro_for_view4 = (Acc_cross - Acc_view4) * 100
            impro_for_view5 = (Acc_cross - Acc_view5) * 100
            impro_for_mean = (Acc_cross - Acc_mean) * 100
            print("-" * 100)
            print('all task in test: \n'
                  ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} | Accuracy_view4 {:5.4f} |'
                  '    Accuracy_view5 {:5.4f} |Accuracy_gap {:5.4f} | Accuracy_cross {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1, Acc_view2, Acc_view3, Acc_view4, Acc_view5, Acc_mean, Acc_cross, Acc_co))
            print(
                'improvment: \n'
                ' |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} | view4 {:5.4f} | view5 {:5.4f} |  mean {:5.4f} '.format(
                    impro_for_view1, impro_for_view2, impro_for_view3, impro_for_view4, impro_for_view5,
                    impro_for_mean))


    model = load_model(hyp_params, name=hyp_params.name)

    evaluate(model, criterion, test=True)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

if __name__ == '__main__':
    test_loss = initiate(hyp_params, test_loader)