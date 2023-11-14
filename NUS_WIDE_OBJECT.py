import torch
import h5py
import os
from torch.utils.data import Dataset



class NUS_WIDE_OBJECTdataset(Dataset):
    # def __init__(self, args, keywords_NETs, actor_NETs):
    def __init__(self, args):
        super(NUS_WIDE_OBJECTdataset, self).__init__()

        root = args.root

        filename = os.path.join(root, 'NUSWIDEOBJ_multi-view.mat')
        data = h5py.File(filename, 'r')
        view1 = torch.tensor(data['view1'][:])
        view2 = torch.tensor(data['view2'][:])
        view3 = torch.tensor(data['view3'][:])
        view4 = torch.tensor(data['view4'][:])
        view5 = torch.tensor(data['view5'][:])
        label = torch.tensor(data['Y'][0]).squeeze(-1)


        self.view1 = view1.t()
        self.view2 = view2.t()
        self.view3 = view3.t()
        self.view4 = view4.t()
        self.view5 = view5.t()
        self.label = label

        if torch.cuda.is_available():
            self.view1 = self.view1.cuda()
            self.view2 = self.view2.cuda()
            self.view3 = self.view3.cuda()
            self.view4 = self.view4.cuda()
            self.view5 = self.view5.cuda()
            self.label = self.label.cuda()


    def __getitem__(self, i):
        view1 = self.view1[i]
        view2 = self.view2[i]
        view3 = self.view3[i]
        view4 = self.view4[i]
        view5 = self.view5[i]
        label = self.label[i]
        label = label.cpu()
        return view1, view2, view3, view4, view5, label

