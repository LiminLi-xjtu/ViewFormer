import timm
import torch
from torch import nn
import time
from torch.utils.data import Dataset
import numpy as np
from MULTdataset import train_dataset,test_dataset

def _fearture(imgs):
    model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        patch_token = model.patch_embed(imgs)
        cls_token = model.cls_token.expand(patch_token.shape[0], -1, -1)
        embedding = torch.cat((cls_token, patch_token), dim=1)
        x = model.pos_drop(embedding + model.pos_embed)
        x = model.blocks(x)
        #x = x[:, 1:, :]
        token_feature = model.norm(x)
    return token_feature



class train_feature(Dataset):
    def __init__(self,args):
        start_time = time.time()
        train_data = train_dataset(args)
        # print(train_data.label)
        print('loading finish train', time.time() - start_time)

        imgs_front = train_data.imgs_front[0:15*20]
        #print('loading front view',type(imgs_front))
        imgs_side = train_data.imgs_side[0:15*20]
        imgs_top = train_data.imgs_top[0:15*20]
        label = train_data.label[0:15*20]
        # print(imgs_front.shape, imgs_side.shape, imgs_top.shape, label.shape)
        # print(label)
        # print(len(label))

        if torch.cuda.is_available():
            imgs_front, imgs_side, imgs_top, label = imgs_front.cuda(), imgs_side.cuda(), imgs_top.cuda(), label.cuda()

        feature_front = _fearture(imgs_front).unsqueeze(1)
        feature_side = _fearture(imgs_side).unsqueeze(1)
        feature_top = _fearture(imgs_top).unsqueeze(1)

        token_feature = torch.cat((torch.cat((feature_front, feature_side), 1), feature_top), 1)
        #print(feature_front.shape, feature_side.shape, feature_top.shape, token_feature.shape)

        self.label = label
        self.token_feature = token_feature

    def __getitem__(self, i):
        token_feature, label = self.token_feature[i], self.label[i]
        token_feature = token_feature.cpu()
        label = label.cpu()
        return token_feature, label

class val_feature(Dataset):
    def __init__(self,args):
        start_time = time.time()
        train_data = train_dataset(args)
        # print(train_data.label)
        print('loading finish val', time.time() - start_time)
        imgs_front = train_data.imgs_front[15*20:]
        imgs_side = train_data.imgs_side[15*20:]
        imgs_top = train_data.imgs_top[15*20:]
        label = train_data.label[15*20:]-15
        # print(imgs_front.shape,imgs_side.shape,imgs_top.shape,label.shape)
        # print(label)
        # print(len(label))

        if torch.cuda.is_available():
            imgs_front, imgs_side, imgs_top, label = imgs_front.cuda(), imgs_side.cuda(), imgs_top.cuda(), label.cuda()

        feature_front = _fearture(imgs_front).unsqueeze(1)
        feature_side = _fearture(imgs_side).unsqueeze(1)
        feature_top = _fearture(imgs_top).unsqueeze(1)

        token_feature = torch.cat((torch.cat((feature_front, feature_side), 1), feature_top), 1)
        #print(feature_front.shape, feature_side.shape, feature_top.shape, token_feature.shape)

        self.label = label
        self.token_feature = token_feature

    def __getitem__(self, i):
        token_feature, label = self.token_feature[i], self.label[i]
        token_feature = token_feature.cpu()
        label = label.cpu()
        return token_feature, label


class test_feature(Dataset):
    def __init__(self,args):
        start_time = time.time()
        test_data = test_dataset(args)
        print('loading finish test',time.time()-start_time)
        imgs_front = test_data.imgs_front
        #print('loading front view',type(imgs_front))
        imgs_side = test_data.imgs_side
        imgs_top = test_data.imgs_top
        label = test_data.label

        if torch.cuda.is_available():
            imgs_front, imgs_side, imgs_top, label = imgs_front.cuda(), imgs_side.cuda(), imgs_top.cuda(), label.cuda()

        feature_front = _fearture(imgs_front).unsqueeze(1)
        feature_side = _fearture(imgs_side).unsqueeze(1)
        feature_top = _fearture(imgs_top).unsqueeze(1)

        token_feature = torch.cat((torch.cat((feature_front, feature_side), 1), feature_top), 1)
        self.label = label
        self.token_feature = token_feature

    def __getitem__(self, i):
        token_feature, label = self.token_feature[i], self.label[i]
        token_feature = token_feature.cpu()
        label = label.cpu()
        return token_feature, label



class test_fusion(Dataset):
    def __init__(self,args):
        start_time = time.time()
        test_data = test_dataset(args)
        print('loading finish test', time.time() - start_time)
        imgs_front = test_data.imgs_front
        # print('loading front view',type(imgs_front))
        imgs_side = test_data.imgs_side
        imgs_top = test_data.imgs_top
        label = test_data.label

        imgs_front, imgs_side, imgs_top, label = imgs_front.cuda(), imgs_side.cuda(), imgs_top.cuda(), label.cuda()

        feature_front = _fearture(imgs_front)
        feature_side = _fearture(imgs_side)
        feature_top = _fearture(imgs_top)
        feature_fusion = torch.cat((torch.cat((feature_front, feature_side), 0), feature_top), 0)
        self.label_fusion = label.repeat(3)
        self.feature_fusion = feature_fusion

    def __getitem__(self, i):
        feature_fusion, label_fusion = self.feature_fusion[i], self.label_fusion[i]
        feature_fusion = feature_fusion.cpu()
        label_fusion = label_fusion.cpu()
        return feature_fusion, label_fusion



if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
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
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('--cut', type=int, default=800)
    n_batch = 20
    n_class = 3
    per_class = 15
    way = 3
    shot = 2
    args = parser.parse_args()
    #train_embedding = train_feature(args)
    val_embedding = val_feature(args)
    '''train_sampler = CategoriesSampler(train_embedding.label.cpu(), n_batch, n_class, per_class)
    train_loader = DataLoader(dataset=train_embedding,
                              num_workers=0,
                              batch_sampler=train_sampler,
                              pin_memory=True)'''
    print('val_embedding_label:')
    print(val_embedding.label)
    val_sampler = CategoriesSampler(val_embedding.label.cpu(), n_batch, n_class, per_class)
    val_loader = DataLoader(dataset=val_embedding,
                            num_workers=0,
                            batch_sampler=val_sampler,
                            pin_memory=True)
    '''for [i_batch, (batch_X, batch_Y)] in enumerate(train_loader):
        eval_attr = batch_Y.squeeze(-1)
        print(eval_attr)'''
    for [i_batch, (batch_X, batch_Y)] in enumerate(val_loader):
        eval_attr = batch_Y.squeeze(-1)
        print(eval_attr)