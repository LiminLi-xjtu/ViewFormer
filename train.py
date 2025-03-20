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
from MULTfeature import train_feature, val_feature, test_feature
from src import cross_view, cross_view_fsl
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

parser.add_argument('--projection', action='store_true',default=False,
                    help='use the projection layer (default: True)')

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
parser.add_argument('--train_num', type=int, default=15)
parser.add_argument('--val_num', type=int, default=6)
parser.add_argument('--test_num', type=int, default=4)

parser.add_argument('--way', type=int, default=3)
parser.add_argument('--shot', type=int, default=2)

parser.add_argument('--data_view', type=str, default='front')
parser.add_argument('--envs', type=str, default='grass')

parser.add_argument('--cut', type=int, default=800)
parser.add_argument('--train_root', type=str, default='./MULT_dataset/pre_process/train_val',
                    help='train_dataset root')
parser.add_argument('--test_root', type=str, default='./MULT_dataset/pre_process/test',
                    help='test_dataset root')
parser.add_argument('--model_root', type=str, default='./model',
                    help='the root to save model')
parser.add_argument('--lamda', type=float, default=1,
                    help='regular term')
parser.add_argument('--beta', type=float, default=1,
                    help='regular term for cross')





args = parser.parse_args()



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
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

train_embedding = train_feature(args)
val_embedding = val_feature(args)
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



train_sampler = CategoriesSampler(train_embedding.label.cpu(),args.train_num,n_class,per_class)
train_loader = DataLoader(dataset=train_embedding,
                                  num_workers=0,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
val_sampler = CategoriesSampler(val_embedding.label.cpu(),args.val_num,n_class,per_class)
val_loader = DataLoader(dataset=val_embedding,
                                  num_workers=0,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)

test_sampler = CategoriesSampler(test_embedding.label.cpu(),args.test_num,n_class,per_class)
test_loader = DataLoader(dataset=test_embedding,
                                  num_workers=0,
                                  batch_sampler=test_sampler,
                                  pin_memory=True)

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 768,768,768
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = args.encoder_dim,args.encoder_dim,args.encoder_dim
hyp_params.layers = args.nlevels
hyp_params.use_cuda = 1
hyp_params.when = args.when
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_embedding.token_feature), len(val_embedding.token_feature), len(test_embedding.token_feature)
hyp_params.model = str.upper(args.model.strip())


criterion = nn.CrossEntropyLoss()


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    models = cross_view_fsl
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

        #single view
        prototype = support_feature.mean(dim=0)
        # print(prototype.shape)  #(way,view,768)
        # print(query_feature.shape)  #query:(30,view,768)

        #single view + gap view
        gap_prototype = prototype.mean(dim=1)  # (way,768)
        prototype = torch.cat((prototype, gap_prototype.unsqueeze(1)), 1)  # (way,4,768)
        gap_query = query_feature.mean(dim=1)  # (30,768)

        # print('support set',support_feature.shape)
        # print('query   set', query_feature.shape)

        #single view + gap view + cross
        support_feature_cross = net(support_feature) #(shot, way, 768, 2)
        proto = support_feature_cross.mean(dim = 0) #(way, 768, 2)
        proto = F.dropout(proto, p=0.2, training=True)
        proto_order1, proto_order2 = proto[:,:,0], proto[:,:,1]
        BN = nn.BatchNorm1d(hyp_params.l_len)
        proto_order1, proto_order2 = BN(proto_order1), BN(proto_order2)
        prototype = torch.cat((prototype, proto_order1.unsqueeze(1), proto_order2.unsqueeze(1)), 1)  # (way,6,768)
        query = net(query_feature)  # (39,768)
        query = F.dropout(query, p=0.2, training=True)
        query = BN(query)
        query_feature = torch.cat((query_feature, gap_query.unsqueeze(1), query.unsqueeze(1), query.unsqueeze(1)), 1)  # (39,6,768)

        gap_feature = BN(batch_X.mean(dim=1))
        cross_feature = torch.cat((BN(support_feature_cross[:, :, :, 0].reshape(-1, hyp_params.orig_d_l)), query), 0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cross_loss = torch.norm(1 - cos(cross_feature, gap_feature), p=1)


        support_label, query_label = prepare_label()
        temp = query_feature - prototype.repeat(1, n_class * (per_class - shot), 1).view(way,
                                                                                         n_class * (per_class - shot),
                                                                                         6, -1)  # (3,39,view+3,768)
        distance = torch.norm(temp, dim=-1)  # (3,way*query,view+3)

        # 协同正则
        dis_co = distance[:, :, 0] + distance[:, :, 1] + distance[:, :, 2]
        distance = torch.cat((distance, dis_co.unsqueeze(-1)), -1)  # (way*query,way,7)

        similarity = -distance.transpose(1, 0)  # (way*query,way,view+4)
        preds = F.softmax(similarity.float(), dim=1).type_as(similarity)  # (30,way,view+4)

        return preds, query_label, cross_loss

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        proc_loss, proc_size = 0, 0
        results = []
        truths = []
        start_time = time.time()
        for [i_batch, (batch_X, batch_Y)] in enumerate(train_loader):
            eval_attr = batch_Y.squeeze(-1)
            # print('meta-train task:',i_batch, eval_attr[0:3])

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    batch_X, eval_attr = batch_X.cuda(), eval_attr.cuda()

            model.zero_grad()
            combined_loss = 0
            net = nn.DataParallel(model)
            preds, query_label, cross_loss = equal_mean(batch_X, batch_Y, net)

            raw_loss = criterion(preds[:,:,4], query_label)
            KL_loss = F.kl_div(preds[:,:,5].log(), preds[:,:,4], reduction='sum')


            combined_loss = (raw_loss + hyp_params.lamda * KL_loss + hyp_params.beta * cross_loss) / (per_class - shot)
            combined_loss.backward()

            # Collect the results into dictionary
            results.append(preds)
            truths.append(query_label)

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += combined_loss.item()


        results = torch.cat(results)
        truths = torch.cat(truths)
        train_num = hyp_params.train_num * hyp_params.way

        return epoch_loss / train_num, results, truths

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else val_loader
        total_loss = 0.0
        raw_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for [i_batch, (batch_X, batch_Y)] in enumerate(loader):
                # torch.manual_seed(2 + i_batch)

                eval_attr = batch_Y.squeeze(-1)
                # if test:
                #     print('meta-test:', i_batch, eval_attr[0:3])
                # else:
                #     print('meta-val:', i_batch, eval_attr[0:3])


                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        batch_X, eval_attr = batch_X.cuda(), eval_attr.cuda()

                combined_loss = 0
                net = model
                preds, query_label, cross_loss = equal_mean(batch_X, batch_Y, net)

                total_loss += criterion(preds[:, :, 4], query_label)
                total_loss += hyp_params.lamda * F.kl_div(preds[:, :, 5].log(), preds[:, :, 4], reduction='sum')
                total_loss += hyp_params.beta * cross_loss


                # Collect the results into dictionary
                results.append(preds)
                truths.append(query_label)
        num = (hyp_params.test_num if test else hyp_params.val_num) * hyp_params.way * (per_class - shot)
        avg_loss = total_loss / num

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths


    best_valid = -100
    epo = 0

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_loss, results, truths = train(model, optimizer, criterion)
        Acc_view1_train  = acc(results[:, :, 0], truths, True)
        Acc_view2_train  = acc(results[:, :, 1], truths, True)
        Acc_view3_train  = acc(results[:, :, 2], truths, True)
        Acc_gap_train    = acc(results[:, :, 3], truths, True)
        Acc_weight_train = acc(results[:, :, 4], truths, True)
        Acc_co_train = acc(results[:, :, 6], truths, True)

        Acc_view1_val,  Acc_view2_val,  Acc_view3_val,  Acc_gap_val,  Acc_weight_val, Acc_co_val =  0, 0, 0, 0, 0, 0
        Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_gap_test, Acc_weight_test, Acc_co_test = 0, 0, 0, 0, 0, 0
        loss_val, loss_test = 0, 0
        num_val = 1
        num_test = 1
        for epoch_val in range(num_val):
            val_loss, results, truths = evaluate(model, criterion, test=False)
            Accuracy_view1_val  = acc(results[:, :, 0], truths, True)
            Accuracy_view2_val  = acc(results[:, :, 1], truths, True)
            Accuracy_view3_val  = acc(results[:, :, 2], truths, True)
            Accuracy_gap_val    = acc(results[:, :, 3], truths, True)
            Accuracy_weight_val = acc(results[:, :, 4], truths, True)
            Accuracy_co_val = acc(results[:, :, 6], truths, True)
            loss_val += val_loss
            Acc_view1_val  += Accuracy_view1_val
            Acc_view2_val  += Accuracy_view2_val
            Acc_view3_val  += Accuracy_view3_val
            Acc_gap_val    += Accuracy_gap_val
            Acc_weight_val += Accuracy_weight_val
            Acc_co_val     += Accuracy_co_val
        for epoch_test in range(num_test):
            test_loss, results, truths = evaluate(model, criterion, test=True)
            Accuracy_view1_test  = acc(results[:, :, 0], truths, True)
            Accuracy_view2_test  = acc(results[:, :, 1], truths, True)
            Accuracy_view3_test  = acc(results[:, :, 2], truths, True)
            Accuracy_gap_test    = acc(results[:, :, 3], truths, True)
            Accuracy_weight_test = acc(results[:, :, 4], truths, True)
            Accuracy_co_test     = acc(results[:, :, 6], truths, True)
            loss_test += test_loss
            Acc_view1_test  += Accuracy_view1_test
            Acc_view2_test  += Accuracy_view2_test
            Acc_view3_test  += Accuracy_view3_test
            Acc_gap_test    += Accuracy_gap_test
            Acc_weight_test += Accuracy_weight_test
            Acc_co_test += Accuracy_co_test
        Acc_view1_val  /= num_val
        Acc_view2_val  /= num_val
        Acc_view3_val  /= num_val
        Acc_gap_val    /= num_val
        Acc_weight_val /= num_val
        Acc_co_val /= num_val

        Acc_view1_test  /= num_test
        Acc_view2_test  /= num_test
        Acc_view3_test  /= num_test
        Acc_gap_test    /= num_test
        Acc_weight_test /= num_test
        Acc_co_test /= num_test

        impro_for_view1 = (Acc_weight_test - Acc_view1_test) * 100
        impro_for_view2 = (Acc_weight_test - Acc_view2_test) * 100
        impro_for_view3 = (Acc_weight_test - Acc_view3_test) * 100
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
            'train:   |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_weight {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_train,  Acc_view2_train,  Acc_view3_train,  Acc_gap_train,  Acc_weight_train, Acc_co_train))
        print(
            'val:     |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_weight {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_val,  Acc_view2_val,  Acc_view3_val,  Acc_gap_val,  Acc_weight_val, Acc_co_val))
        print(
            'test:    |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_weight {:5.4f} | Accuracy_co {:5.4f} |'.format(
                Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_gap_test, Acc_weight_test, Acc_co_test))
        print(
            'improvment: \n',
            'valid:    |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} |  mean {:5.4f} \n'.format(
                (Acc_weight_val - Acc_view1_val) * 100, (Acc_weight_val - Acc_view2_val) * 100,
                (Acc_weight_val - Acc_view3_val) * 100, (Acc_weight_val - Acc_gap_val) * 100),
            'test:     |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} |  mean {:5.4f} '.format(
                impro_for_view1, impro_for_view2, impro_for_view3, impro_for_mean)
        )

        # print("-" * 50)

        improve = Acc_weight_val - Acc_gap_val
        if improve > best_valid:
            epo = epoch
            # print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = improve

    model = load_model(hyp_params, name=hyp_params.name)
    Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_gap_test, Acc_weight_test,Acc_co_test  = 0, 0, 0, 0, 0, 0
    num_test = 1
    for epoch_test in range(num_test):

        test_loss, results, truths = evaluate(model, criterion, test=True)
        Accuracy_view1_test = acc(results[:, :, 0], truths, True)
        Accuracy_view2_test = acc(results[:, :, 1], truths, True)
        Accuracy_view3_test = acc(results[:, :, 2], truths, True)
        Accuracy_gap_test = acc(results[:, :, 3], truths, True)
        Accuracy_weight_test = acc(results[:, :, 4], truths, True)
        Accuracy_co_test = acc(results[:, :, 6], truths, True)

        Acc_view1_test += Accuracy_view1_test
        Acc_view2_test += Accuracy_view2_test
        Acc_view3_test += Accuracy_view3_test
        Acc_gap_test += Accuracy_gap_test
        Acc_weight_test += Accuracy_weight_test
        Acc_co_test += Accuracy_co_test
    Acc_view1_test /= num_test
    Acc_view2_test /= num_test
    Acc_view3_test /= num_test
    Acc_gap_test /= num_test
    Acc_weight_test /= num_test
    Acc_co_test /= num_test

    impro_for_view1 = (Acc_weight_test - Acc_view1_test) * 100
    impro_for_view2 = (Acc_weight_test - Acc_view2_test) * 100
    impro_for_view3 = (Acc_weight_test - Acc_view3_test) * 100
    impro_for_mean = (Acc_weight_test - Acc_gap_test) * 100

    print("-" * 100)
    print(
        'final test in {:5.4f} epoch:   \n'
        ' |  Accuracy_view1 {:5.4f} | Accuracy_view2 {:5.4f} | Accuracy_view3 {:5.4f} |  Accuracy_gap {:5.4f} | Accuracy_weight {:5.4f} | Accuracy_co {:5.4f} |'.format(
            epo,Acc_view1_test, Acc_view2_test, Acc_view3_test, Acc_gap_test, Acc_weight_test, Acc_co_test))
    print(
        'improvment: \n'
        ' |  view1 {:5.4f} | view2 {:5.4f} | view3 {:5.4f} |  mean {:5.4f} '.format(
            impro_for_view1, impro_for_view2, impro_for_view3, impro_for_mean))

    sys.stdout.flush()
    input('[Press Any Key to start another run]')


if __name__ == '__main__':
    test_loss = initiate(hyp_params, train_loader, val_loader, test_loader)
