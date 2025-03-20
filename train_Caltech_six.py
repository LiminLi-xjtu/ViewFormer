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
from Caltech import Caltechdataset
from src.six_view_fsl import feature_proj, MULTModel, ViewFormer
from src.eval_metrics import acc
from src.utils import *

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


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
parser.add_argument('--alpha_6', type=float, default=1,
                    help='alpha_6')
parser.add_argument('--lamda', type=float, default=1,
                    help='regular term')
parser.add_argument('--beta', type=float, default=1,
                    help='regular term for cross')

args = parser.parse_args()

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

multi_view_data = Caltechdataset(args)

train_sampler = Sampler(multi_view_data.label, [0, 12], args.train_num, way, per_class)
val_sampler = Sampler(multi_view_data.label, [12, 16], args.val_num, way, per_class)
test_sampler = Sampler(multi_view_data.label, [16, 20], args.test_num, way, per_class)

train_loader = DataLoader(dataset=multi_view_data,
                          num_workers=0,
                          batch_sampler=train_sampler,
                          pin_memory=False)

val_loader = DataLoader(dataset=multi_view_data,
                        num_workers=0,
                        batch_sampler=val_sampler,
                        pin_memory=False)

test_loader = DataLoader(dataset=multi_view_data,
                         num_workers=0,
                         batch_sampler=test_sampler,
                         pin_memory=False)

hyp_params = args
hyp_params.orig_d_1, hyp_params.orig_d_2, hyp_params.orig_d_3, hyp_params.orig_d_4, hyp_params.orig_d_5, hyp_params.orig_d_6 = 48, 40, 254, 1984, 512, 928
# 48	40	254	1984	512	928
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = args.encoder_dim,args.encoder_dim,args.encoder_dim
hyp_params.layers = args.nlevels
hyp_params.use_cuda = True
hyp_params.when = args.when
hyp_params.model = str.upper(args.model.strip())

criterion = nn.CrossEntropyLoss()

def initiate(hyp_params, train_loader, valid_loader, test_loader):

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
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, train_loader, val_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

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

    def equal_mean(view1, view2, view3, view4, view5, view6, batch_Z, net):

        eval_attr = batch_Z.squeeze(-1)

        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                view1, view2, view3, view4, view5, view6, eval_attr = view1.cuda(), view2.cuda(), view3.cuda(), view4.cuda(), view5.cuda(), view6.cuda(), eval_attr.cuda()

        support_idx, query_idx = (torch.Tensor(np.arange(n_class * shot)).long().view(shot, n_class),
                                  torch.Tensor(np.arange(n_class * shot, n_class * per_class)).long())
        support_view1 = view1[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view1.shape[-1])
        support_view2 = view2[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view2.shape[-1])
        support_view3 = view3[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view3.shape[-1])
        support_view4 = view4[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view4.shape[-1])
        support_view5 = view5[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view5.shape[-1])
        support_view6 = view6[support_idx.contiguous().view(-1)].contiguous().view(shot, n_class, view6.shape[-1])

        query_view1 = view1[query_idx.contiguous().view(-1)].contiguous().view(-1, view1.shape[-1])
        query_view2 = view2[query_idx.contiguous().view(-1)].contiguous().view(-1, view2.shape[-1])
        query_view3 = view3[query_idx.contiguous().view(-1)].contiguous().view(-1, view3.shape[-1])
        query_view4 = view4[query_idx.contiguous().view(-1)].contiguous().view(-1, view4.shape[-1])
        query_view5 = view5[query_idx.contiguous().view(-1)].contiguous().view(-1, view5.shape[-1])
        query_view6 = view6[query_idx.contiguous().view(-1)].contiguous().view(-1, view6.shape[-1])

        support_feature_view1, support_feature_view2, support_feature_view3, support_feature_view4, support_feature_view5, support_feature_view6, support_feature_mean, support_feature_cross = \
            net(support_view1, support_view2, support_view3, support_view4, support_view5, support_view6)
        query_feature_view1, query_feature_view2, query_feature_view3, query_feature_view4, query_feature_view5, query_feature_view6, query_feature_mean, query_feature_cross = \
            net(query_view1, query_view2, query_view3, query_view4, query_view5, query_view6)


        support_feature = torch.cat((support_feature_view1.unsqueeze(-2), support_feature_view2.unsqueeze(-2), support_feature_view3.unsqueeze(-2),
                                     support_feature_view4.unsqueeze(-2), support_feature_view5.unsqueeze(-2), support_feature_view6.unsqueeze(-2), support_feature_mean.unsqueeze(-2)), -2) #(shot,way,view+1,dim)
        query_feature = torch.cat((query_feature_view1.unsqueeze(-2), query_feature_view2.unsqueeze(-2), query_feature_view3.unsqueeze(-2),
                                   query_feature_view4.unsqueeze(-2), query_feature_view5.unsqueeze(-2), query_feature_view6.unsqueeze(-2), query_feature_mean.unsqueeze(-2)), -2)

        prototype = support_feature.mean(dim=0)#(way,view+1,dim)
        # print(prototype.shape)  #(way,view,768)
        # print(query_feature.shape)  #query:(30,view,768)


        #single view + mean view + cross

        proto = support_feature_cross.mean(dim = 0) #(way, 768, 2)

        proto = F.dropout(proto, p=0.2, training=True)
        proto_order1, proto_order2 = proto[:,:,0], proto[:,:,1]
        BN = nn.BatchNorm1d(hyp_params.encoder_dim)
        proto_order1, proto_order2 = BN(proto_order1), BN(proto_order2)
        prototype = torch.cat((prototype, proto_order1.unsqueeze(1), proto_order2.unsqueeze(1)), 1)  # (way,6,768)

        query = F.dropout(query_feature_cross, p=0.2, training=True)
        query = BN(query)
        query_feature = torch.cat((query_feature, query.unsqueeze(1), query.unsqueeze(1)), 1)  # (39,6,768)

        mean_feature = BN(torch.cat((support_feature_mean.view(args.way*args.shot, -1), query_feature_mean), 0))
        cross_feature = torch.cat((BN(support_feature_cross[:, :, :, 0].reshape(-1, hyp_params.encoder_dim)), query), 0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # cross_loss = torch.norm(1 - cos(cross_feature, mean_feature), p=1)
        cross_loss = 0
        for v in range(args.view_num):
            # print(support_feature[:, :, v, :].shape, query_feature[:, v, :].shape)
            # print(torch.cat((support_feature[:, :, v, :].view(args.way * args.shot, -1), query_feature[:, v, :]), 0).shape)
            feature = BN(
                torch.cat((support_feature[:, :, v, :].view(args.way * args.shot, -1), query_feature[:, v, :]), 0))
            cross_loss += torch.norm(1 - cos(cross_feature, feature), p=1)
        cross_loss /= args.view_num

        support_label, query_label = prepare_label()
        temp = query_feature - prototype.repeat(1, n_class * (per_class - shot), 1).view(way,
                                                                                         n_class * (per_class - shot),
                                                                                         args.view_num + 3, -1)  # (3,39,view+3,768)
        distance = torch.norm(temp, dim=-1)  # (3,way*query,view+3)

        # 协同正则
        dis_co = distance[:, :, 0] + distance[:, :, 1] + distance[:, :, 2] + distance[:, :, 3] + distance[:, :, 4] + distance[:, :, 5]
        distance = torch.cat((distance, dis_co.unsqueeze(-1)), -1)  # (way*query,way,7)

        similarity = -distance.transpose(1, 0)  # (way*query,way,view+4)
        preds = F.softmax(similarity.float(), dim=1).type_as(similarity)  # (30,way,view+4)

        return preds, query_label, cross_loss

    def train(model, optimizer, epoch, criterion):
        epoch_loss = 0
        model.train()
        proc_loss, proc_size = 0, 0
        results = []
        truths = []
        start_time = time.time()
        for [i_batch, (batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_X6, batch_Z)] in enumerate(train_loader):
            eval_attr = batch_Z.squeeze(-1)
            # print('meta-train task:',i_batch, eval_attr[0:3])


            model.zero_grad()

            net = nn.DataParallel(model)
            preds, query_label, cross_loss = equal_mean(batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_X6, batch_Z, net)


            pre_epoch = args.num_epochs / 1
            # pre_epoch = 0
            if epoch <= pre_epoch:
                view_1_loss = criterion(preds[:, :, 0], query_label)
                view_2_loss = criterion(preds[:, :, 1], query_label)
                view_3_loss = criterion(preds[:, :, 2], query_label)
                view_4_loss = criterion(preds[:, :, 3], query_label)
                view_5_loss = criterion(preds[:, :, 4], query_label)
                view_6_loss = criterion(preds[:, :, 5], query_label)
                raw_loss = ((hyp_params.alpha_1 + 1) * view_1_loss + (hyp_params.alpha_2 + 1) * view_2_loss + (hyp_params.alpha_3 + 1) *
                            view_3_loss + (hyp_params.alpha_4 + 1) * view_4_loss + (hyp_params.alpha_5 + 1) * view_5_loss + (hyp_params.alpha_6 + 1) * view_6_loss) / (
                        hyp_params.alpha_1 + hyp_params.alpha_2 + hyp_params.alpha_3 + hyp_params.alpha_4 + hyp_params.alpha_5+ hyp_params.alpha_6 + 6)

                combined_loss = raw_loss / (per_class - shot)
            else:
                raw_loss = criterion(preds[:, :, args.view_num + 1], query_label)
                view_1_loss = criterion(preds[:, :, 0], query_label)
                view_2_loss = criterion(preds[:, :, 1], query_label)
                view_3_loss = criterion(preds[:, :, 2], query_label)
                view_4_loss = criterion(preds[:, :, 3], query_label)
                view_5_loss = criterion(preds[:, :, 4], query_label)
                view_6_loss = criterion(preds[:, :, 5], query_label)
                raw_loss = (raw_loss + hyp_params.alpha_1 * view_1_loss + hyp_params.alpha_2 * view_2_loss + hyp_params.alpha_3 * view_3_loss
                            + hyp_params.alpha_4 * view_4_loss + hyp_params.alpha_5 * view_5_loss + hyp_params.alpha_6 * view_6_loss) / (
                                   1 + hyp_params.alpha_1 + hyp_params.alpha_2 + hyp_params.alpha_3 + hyp_params.alpha_4 + hyp_params.alpha_5 + hyp_params.alpha_6)

                KL_loss = F.kl_div(preds[:,:,args.view_num+2].log(), preds[:,:,args.view_num+1], reduction='sum')
                # combined_loss = (raw_loss / epoch + hyp_params.lamda * KL_loss + hyp_params.beta * cross_loss) / (per_class - shot)
                combined_loss = (raw_loss + hyp_params.lamda * KL_loss + hyp_params.beta * cross_loss) / (per_class - shot)
            combined_loss.backward(retain_graph=True)

            # Collect the results into dictionary
            results.append(preds)
            truths.append(query_label)

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += combined_loss.item()


        results = torch.cat(results)
        truths = torch.cat(truths)
        train_num = (i_batch+1) * hyp_params.way
        # print('train', raw_loss, view_1_loss, view_2_loss, (per_class - shot)*hyp_params.train_num * hyp_params.way, (per_class - shot)*(i_batch+1) * hyp_params.way)

        return epoch_loss / train_num, results, truths

    def evaluate(model, criterion, epoch, test=False):
        model.eval()
        loader = test_loader if test else val_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for [i_batch, (batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_X6, batch_Z)] in enumerate(loader):
                eval_attr = batch_Z.squeeze(-1)

                combined_loss = 0
                net = model
                preds, query_label, cross_loss = equal_mean(batch_X1, batch_X2, batch_X3, batch_X4, batch_X5, batch_X6, batch_Z, net)

                # total_loss += criterion(preds[:, :, args.view_num+1], query_label) / epoch

                # pre_epoch = 0
                pre_epoch = args.num_epochs / 1
                if epoch <= pre_epoch:
                    view_1_loss = criterion(preds[:, :, 0], query_label)
                    view_2_loss = criterion(preds[:, :, 1], query_label)
                    view_3_loss = criterion(preds[:, :, 2], query_label)
                    view_4_loss = criterion(preds[:, :, 3], query_label)
                    view_5_loss = criterion(preds[:, :, 4], query_label)
                    view_6_loss = criterion(preds[:, :, 5], query_label)
                    raw_loss = ((hyp_params.alpha_1 + 1) * view_1_loss + (hyp_params.alpha_2 + 1) * view_2_loss + (
                                 hyp_params.alpha_3 + 1) *view_3_loss + (hyp_params.alpha_4 + 1) * view_4_loss + (
                                 hyp_params.alpha_5 + 1) * view_5_loss + (hyp_params.alpha_6 + 1) * view_6_loss) / (
                                       hyp_params.alpha_1 + hyp_params.alpha_2 + hyp_params.alpha_3 + hyp_params.alpha_4 + hyp_params.alpha_5 + hyp_params.alpha_6 + 6)


                    total_loss = raw_loss
                else:
                    raw_loss = criterion(preds[:, :, args.view_num + 1], query_label)
                    view_1_loss = criterion(preds[:, :, 0], query_label)
                    view_2_loss = criterion(preds[:, :, 1], query_label)
                    view_3_loss = criterion(preds[:, :, 2], query_label)
                    view_4_loss = criterion(preds[:, :, 3], query_label)
                    view_5_loss = criterion(preds[:, :, 4], query_label)
                    view_6_loss = criterion(preds[:, :, 5], query_label)
                    raw_loss = (raw_loss + hyp_params.alpha_1 * view_1_loss + hyp_params.alpha_2 * view_2_loss + hyp_params.alpha_3 * view_3_loss
                                + hyp_params.alpha_4 * view_4_loss + hyp_params.alpha_5 * view_5_loss + hyp_params.alpha_6 * view_6_loss) / (
                                 1 + hyp_params.alpha_1 + hyp_params.alpha_2 + hyp_params.alpha_3 + hyp_params.alpha_4 + hyp_params.alpha_5 + hyp_params.alpha_6)

                    KL_loss = F.kl_div(preds[:, :, args.view_num + 2].log(), preds[:, :, args.view_num + 1],
                                       reduction='sum')
                    # combined_loss = (raw_loss / epoch + hyp_params.lamda * KL_loss + hyp_params.beta * cross_loss) / (per_class - shot)
                    total_loss = raw_loss + hyp_params.lamda * KL_loss + hyp_params.beta * cross_loss

                '''raw_loss = criterion(preds[:, :, args.view_num + 1], query_label)
                view_1_loss = criterion(preds[:, :, 0], query_label)
                view_2_loss = criterion(preds[:, :, 1], query_label)

                total_loss += (raw_loss + hyp_params.alpha_1 * view_1_loss + hyp_params.alpha_2 * view_2_loss) / (
                            1 + hyp_params.alpha_1 + hyp_params.alpha_2)
                total_loss += hyp_params.lamda * F.kl_div(preds[:, :, args.view_num+2].log(), preds[:, :, args.view_num+1], reduction='sum')
                total_loss += hyp_params.beta * cross_loss'''


                # Collect the results into dictionary
                results.append(preds)
                truths.append(query_label)

        num = (i_batch + 1) * hyp_params.way * (per_class - shot)
        num = hyp_params.way * (per_class - shot)
        # print(test, raw_loss, view_1_loss, view_2_loss, num)
        avg_loss = total_loss / num

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss.item(), results, truths


    best_valid = 0
    epo = 0

    writer = SummaryWriter('./logs')
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss, results, truths = train(model, optimizer, epoch, criterion)
        Acc_view1_train  = acc(results[:, :, 0], truths, True)
        Acc_view2_train  = acc(results[:, :, 1], truths, True)
        Acc_view3_train = acc(results[:, :, 2], truths, True)
        Acc_view4_train = acc(results[:, :, 3], truths, True)
        Acc_view5_train = acc(results[:, :, 4], truths, True)
        Acc_view6_train = acc(results[:, :, 5], truths, True)


        Acc_gap_train    = acc(results[:, :, args.view_num], truths, True)
        Acc_weight_train = acc(results[:, :, args.view_num+1], truths, True)
        Acc_co_train = acc(results[:, :, args.view_num+3], truths, True)

        Acc_view1_val,  Acc_view2_val,  Acc_view3_val, Acc_view4_val, Acc_view5_val, Acc_gap_val,  Acc_weight_val, Acc_co_val = 0, 0, 0, 0, 0, 0, 0, 0
        Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_view4_test,Acc_view5_test, Acc_gap_test, Acc_weight_test, Acc_co_test = 0, 0, 0, 0, 0, 0, 0, 0
        loss_val, loss_test = 0, 0
        num_val = 1
        num_test = 1
        for epoch_val in range(num_val):
            val_loss, results, truths = evaluate(model, criterion, epoch, test=False)
            Accuracy_view1_val  = acc(results[:, :, 0], truths, True)
            Accuracy_view2_val  = acc(results[:, :, 1], truths, True)
            Accuracy_view3_val = acc(results[:, :, 2], truths, True)
            Accuracy_view4_val = acc(results[:, :, 3], truths, True)
            Accuracy_view5_val = acc(results[:, :, 4], truths, True)


            Accuracy_gap_val    = acc(results[:, :, args.view_num], truths, True)
            Accuracy_weight_val = acc(results[:, :, args.view_num+1], truths, True)
            Accuracy_co_val = acc(results[:, :, args.view_num+3], truths, True)
            loss_val += val_loss
            Acc_view1_val  += Accuracy_view1_val
            Acc_view2_val  += Accuracy_view2_val
            Acc_view3_val += Accuracy_view3_val
            Acc_view4_val += Accuracy_view4_val
            Acc_view5_val += Accuracy_view5_val

            Acc_gap_val    += Accuracy_gap_val
            Acc_weight_val += Accuracy_weight_val
            Acc_co_val     += Accuracy_co_val
        for epoch_test in range(num_test):
            test_loss, results, truths = evaluate(model, criterion, epoch, test=True)
            Accuracy_view1_test  = acc(results[:, :, 0], truths, True)
            Accuracy_view2_test  = acc(results[:, :, 1], truths, True)
            Accuracy_view3_test = acc(results[:, :, 2], truths, True)
            Accuracy_view4_test = acc(results[:, :, 3], truths, True)
            Accuracy_view5_test = acc(results[:, :, 4], truths, True)

            Accuracy_gap_test    = acc(results[:, :, args.view_num], truths, True)
            Accuracy_weight_test = acc(results[:, :, args.view_num+1], truths, True)
            Accuracy_co_test     = acc(results[:, :, args.view_num+3], truths, True)
            loss_test += test_loss
            Acc_view1_test  += Accuracy_view1_test
            Acc_view2_test  += Accuracy_view2_test
            Acc_view3_test += Accuracy_view3_test
            Acc_view4_test += Accuracy_view4_test
            Acc_view5_test += Accuracy_view5_test

            Acc_gap_test    += Accuracy_gap_test
            Acc_weight_test += Accuracy_weight_test
            Acc_co_test += Accuracy_co_test
        Acc_view1_val  /= num_val
        Acc_view2_val  /= num_val
        Acc_view3_val /= num_val
        Acc_view4_val /= num_val
        Acc_view5_val /= num_val

        Acc_gap_val    /= num_val
        Acc_weight_val /= num_val
        Acc_co_val /= num_val

        Acc_view1_test  /= num_test
        Acc_view2_test  /= num_test
        Acc_view3_test /= num_test
        Acc_view4_test /= num_test
        Acc_view5_test /= num_test

        Acc_gap_test    /= num_test
        Acc_weight_test /= num_test
        Acc_co_test /= num_test

        impro_for_view1 = (Acc_weight_test - Acc_view1_test) * 100
        impro_for_view2 = (Acc_weight_test - Acc_view2_test) * 100
        impro_for_view3 = (Acc_weight_test - Acc_view3_test) * 100
        impro_for_view4 = (Acc_weight_test - Acc_view4_test) * 100
        impro_for_view5 = (Acc_weight_test - Acc_view5_test) * 100

        impro_for_mean = (Acc_weight_test - Acc_gap_test) * 100

        val_loss = loss_val / num_val
        test_loss = loss_test / num_test

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 150)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch,
                                                                                                                  duration,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  test_loss))
        print(
            'train:   |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} | Accuracy_view4 {:5.4f} | Accuracy_view4 {:5.4f} |'
            '  Accuracy_gap {:5.4f} | Accuracy_att {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_train,  Acc_view2_train, Acc_view3_train, Acc_view4_train, Acc_view5_train, Acc_gap_train,  Acc_weight_train, Acc_co_train))
        print(
            'val:     |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} | Accuracy_view4 {:5.4f} | Accuracy_view5 {:5.4f} |'
            '  Accuracy_gap {:5.4f} | Accuracy_att {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_val,  Acc_view2_val, Acc_view3_val, Acc_view4_val, Acc_view5_val,  Acc_gap_val,  Acc_weight_val, Acc_co_val))
        print(
            'test:    |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} | Accuracy_view4 {:5.4f} | Accuracy_view5 {:5.4f} |'
            '  Accuracy_gap {:5.4f} | Accuracy_att {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_view4_test,  Acc_view5_test, Acc_gap_test, Acc_weight_test, Acc_co_test))
        print(
            'improvment: \n',
            'valid:    |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} | view4 {:5.4f} | view5 {:5.4f} |  mean {:5.4f} \n'.format(
                (Acc_weight_val - Acc_view1_val) * 100, (Acc_weight_val - Acc_view2_val) * 100, (Acc_weight_val - Acc_view3_val) * 100,
                (Acc_weight_val - Acc_view4_val) * 100, (Acc_weight_val - Acc_view5_val) * 100, (Acc_weight_val - Acc_gap_val) * 100),
            'test:     |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} || view4 {:5.4f}   mean {:5.4f} '.format(
                impro_for_view1, impro_for_view2, impro_for_view3, impro_for_view4, impro_for_view5, impro_for_mean)
        )

        # print("-" * 50)
        writer.add_scalars("train_acc", {"view_1": Acc_view1_train, "view_2": Acc_view2_train, "view_3": Acc_view3_train, "view_mean": Acc_gap_train,
                                    'view_cross': Acc_weight_train}, epoch)
        writer.add_scalars("val_acc", {"view_1": Acc_view1_val, "view_2": Acc_view2_val, "view_3": Acc_view3_val, "view_mean": Acc_gap_val,
                                    'view_cross': Acc_weight_val}, epoch)
        writer.add_scalars("test_acc", {"view_1": Acc_view1_test, "view_2": Acc_view2_test, "view_3": Acc_view3_test, "view_mean": Acc_gap_test,
                                    'view_cross': Acc_weight_test}, epoch)
        writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}, epoch)


        improve = Acc_weight_val - Acc_gap_val
        if (improve > best_valid) or (epoch == 1):
            epo = epoch
            # print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = improve

    writer.close()

    model = load_model(hyp_params, name=hyp_params.name)
    Acc_view1_test, Acc_view2_test,  Acc_view3_test,  Acc_view4_test,  Acc_view5_test,  Acc_view6_test,  Acc_gap_test, Acc_weight_test,Acc_co_test  = 0, 0, 0, 0, 0, 0, 0, 0, 0
    num_test = 100
    for epoch_test in range(num_test):

        test_loss, results, truths = evaluate(model, criterion, epoch, test=True)
        Accuracy_view1_test = acc(results[:, :, 0], truths, True)
        Accuracy_view2_test = acc(results[:, :, 1], truths, True)
        Accuracy_view3_test = acc(results[:, :, 2], truths, True)
        Accuracy_view4_test = acc(results[:, :, 3], truths, True)
        Accuracy_view5_test = acc(results[:, :, 4], truths, True)
        Accuracy_view6_test = acc(results[:, :, 5], truths, True)

        Accuracy_gap_test = acc(results[:, :, args.view_num], truths, True)
        Accuracy_weight_test = acc(results[:, :, args.view_num + 1], truths, True)
        Accuracy_co_test = acc(results[:, :, args.view_num + 3], truths, True)


        Acc_view1_test += Accuracy_view1_test
        Acc_view2_test += Accuracy_view2_test
        Acc_view3_test += Accuracy_view3_test
        Acc_view4_test += Accuracy_view4_test
        Acc_view5_test += Accuracy_view5_test
        Acc_view6_test += Accuracy_view6_test

        Acc_gap_test += Accuracy_gap_test
        Acc_weight_test += Accuracy_weight_test
        Acc_co_test += Accuracy_co_test
    Acc_view1_test /= num_test
    Acc_view2_test /= num_test
    Acc_view3_test /= num_test
    Acc_view4_test /= num_test
    Acc_view5_test /= num_test
    Acc_view6_test /= num_test
    Acc_gap_test /= num_test
    Acc_weight_test /= num_test
    Acc_co_test /= num_test

    impro_for_view1 = (Acc_weight_test - Acc_view1_test) * 100
    impro_for_view2 = (Acc_weight_test - Acc_view2_test) * 100
    impro_for_view3 = (Acc_weight_test - Acc_view3_test) * 100
    impro_for_view4 = (Acc_weight_test - Acc_view4_test) * 100
    impro_for_view5 = (Acc_weight_test - Acc_view5_test) * 100
    impro_for_view6 = (Acc_weight_test - Acc_view6_test) * 100
    impro_for_mean = (Acc_weight_test - Acc_gap_test) * 100

    print("-" * 100)
    print(
        'final test in {:5.4f} epoch:   \n'
        ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} | Accuracy_view4 {:5.4f} | Accuracy_view5 {:5.4f} | Accuracy_view6 {:5.4f} |'
        '  Accuracy_gap {:5.4f} | Accuracy_att {:5.4f} | Accuracy_co {:5.4f} |'.format(
            epo,Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_view4_test,  Acc_view5_test,  Acc_view6_test,  Acc_gap_test, Acc_weight_test, Acc_co_test))
    print(
        'improvment: \n'
        ' |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} | view4 {:5.4f} | view5 {:5.4f} | view6 {:5.4f} |  mean {:5.4f} '.format(
            impro_for_view1, impro_for_view2, impro_for_view3, impro_for_view4,  impro_for_view5,  impro_for_view6, impro_for_mean))

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

if __name__ == '__main__':
    test_loss = initiate(hyp_params, train_loader, val_loader, test_loader)



