import torch
import h5py
import os
from torch.utils.data import Dataset



class Caltechdataset(Dataset):
    # def __init__(self, args, keywords_NETs, actor_NETs):
    def __init__(self, args):
        super(Caltechdataset, self).__init__()

        root = args.root

        filename = os.path.join(root, 'Caltech101-20_multi-view.mat')
        data = h5py.File(filename, 'r')
        view1 = torch.tensor(data['view1'][:])
        view2 = torch.tensor(data['view2'][:])
        view3 = torch.tensor(data['view3'][:])
        view4 = torch.tensor(data['view4'][:])
        view5 = torch.tensor(data['view5'][:])
        view6 = torch.tensor(data['view6'][:])
        label = torch.tensor(data['labels'][0]).squeeze(-1)


        self.view1 = view1.t()
        self.view2 = view2.t()
        self.view3 = view3.t()
        self.view4 = view4.t()
        self.view5 = view5.t()
        self.view6 = view6.t()
        self.label = label

        if torch.cuda.is_available():
            self.view1 = self.view1.cuda()
            self.view2 = self.view2.cuda()
            self.view3 = self.view3.cuda()
            self.view4 = self.view4.cuda()
            self.view5 = self.view5.cuda()
            self.view6 = self.view6.cuda()
            self.label = self.label.cuda()


    def __getitem__(self, i):
        view1 = self.view1[i]
        view2 = self.view2[i]
        view3 = self.view3[i]
        view4 = self.view4[i]
        view5 = self.view5[i]
        view6 = self.view6[i]
        label = self.label[i]
        label = label.cpu()
        return view1, view2, view3, view4, view5, view6, label

