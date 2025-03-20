import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from modules.transform_FSL import TransformerEncoder

class feature_proj(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a feature_extractor

        input:[len, num_keywords], [len, num_actor]
        output:[len, encoder_dim], [len, encoder_dim]
        """
        super(feature_proj, self).__init__()
        # self.dim = 1024
        self.orig_d_1, self.orig_d_2, self.orig_d_3, self.orig_d_4, self.orig_d_5 = hyp_params.orig_d_1, hyp_params.orig_d_2, hyp_params.orig_d_3, hyp_params.orig_d_4, hyp_params.orig_d_5
        self.encoder_dim = hyp_params.encoder_dim

        # 1. Temporal convolutional layers

        self.proj_1 = nn.Sequential(
            nn.Linear(self.orig_d_1, self.encoder_dim),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(self.orig_d_2, self.encoder_dim),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.proj_3 = nn.Sequential(
            nn.Linear(self.orig_d_3, self.encoder_dim),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.proj_4 = nn.Sequential(
            nn.Linear(self.orig_d_4, self.encoder_dim),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.proj_5 = nn.Sequential(
            nn.Linear(self.orig_d_5, self.encoder_dim),
            # nn.ReLU(),
            nn.Dropout(0.1),
        )
        # self.proj_2 = nn.Linear(self.orig_d_2, self.encoder_dim)
        # self.proj_3 = nn.Linear(self.orig_d_3, self.encoder_dim)



    def forward(self, x_1, x_2, x_3, x_4, x_5):
        # print(keywords_matrix.shape, actor_matrix.shape)

        # Project the view features
        if self.orig_d_1 == self.encoder_dim:
            proj_x_1, proj_x_2, proj_x_3, proj_x_4, proj_x_5 = x_1, x_2, x_3, x_4, x_5
        else:
            x_1, x_2, x_3, x_4, x_5 = x_1.to(torch.float32), x_2.to(torch.float32), x_3.to(torch.float32), x_4.to(torch.float32), x_5.to(torch.float32)
            proj_x_1, proj_x_2, proj_x_3, proj_x_4, proj_x_5 = self.proj_1(x_1), self.proj_2(x_2), self.proj_3(x_3), self.proj_4(x_4), self.proj_5(x_5)

        return proj_x_1, proj_x_2, proj_x_3, proj_x_4, proj_x_5

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_1, self.orig_d_2, self.orig_d_3, self.orig_d_4, self.orig_d_5 = hyp_params.orig_d_1, hyp_params.orig_d_2, hyp_params.orig_d_3, hyp_params.orig_d_4, hyp_params.orig_d_5
        self.d_l, self.d_a, self.d_v = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len

        self.num_heads = hyp_params.num_heads
        self.view_num = hyp_params.view_num
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.view_encoding = hyp_params.view_encoding
        self.shot_encoding = hyp_params.shot_encoding


        self.sep_embedding = Parameter(torch.rand(1, self.d_l, requires_grad=True))

        if self.shot_encoding:
            self.shot_embedding = Parameter(torch.rand(hyp_params.shot, self.d_l, requires_grad=True))
        else:
            self.shot_embedding = torch.zeros(hyp_params.shot, self.d_l, requires_grad=False)




            # 1. Temporal convolutional layers

        self.proj = nn.Linear(self.orig_d_1, self.d_l)
        self.proj_1 = nn.Linear(self.d_l, self.d_l)
        self.proj_2 = nn.Linear(self.d_l, self.d_l)
        self.proj_3 = nn.Linear(self.d_l, self.d_l)
        self.proj_4 = nn.Linear(self.d_l, self.d_l)
        self.proj_5 = nn.Linear(self.d_l, self.d_l)

        self.pos = nn.Linear(5, self.d_l, bias=False)


        # 2. Crossmodal Attentions

        self.cross_att = self.get_network(self_type='cross')


        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)

        self.self_att = self.get_network(self_type='self', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(self.d_l, self.d_l)
        self.proj2 = nn.Linear(self.d_l, self.d_l)

    def get_network(self, self_type='cross', layers=-1):
        if self_type in ['cross']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['self']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout)



    def forward(self, x):
        """
        input: [view1,view2,view3]   { shot , way, view , dim }
        """
        x = F.dropout(x, p=self.embed_dropout, training=self.training)
        # print(x.shape)
        # print(self.proj)


        if self.view_encoding:
            #print('True')
            mask1 = torch.Tensor((1, 0, 0, 0, 0))
            mask2 = torch.Tensor((0, 1, 0, 0, 0))
            mask3 = torch.Tensor((0, 0, 1, 0, 0))
            mask4 = torch.Tensor((0, 0, 0, 1, 0))
            mask5 = torch.Tensor((0, 0, 0, 0, 1))
            encoding1 = self.pos(mask1)
            encoding2 = self.pos(mask2)
            encoding3 = self.pos(mask3)
            encoding4 = self.pos(mask4)
            encoding5 = self.pos(mask5)
            position = torch.cat([encoding1, encoding2, encoding3, encoding4, encoding5]).view(5,-1)
            # position = self.pos

        else:
            #print('false')
            position = torch.zeros(5, self.d_l, requires_grad=False)


        shape = x.shape
        # LN = nn.LayerNorm(self.d_l)
        if len(shape) == 4:
            # support set (shot,way,3,dim)
            shot,way = shape[0], shape[1]

            proj_x = x
            # position encoding
            # add & norm
            proj_x += position.repeat(shot, way, 1, 1)
            proj_x_order1 = proj_x + self.shot_embedding.repeat(way, self.view_num, 1, 1).permute(2, 0, 1, 3)
            proj_x_order1 = proj_x_order1.permute(1, 0, 2, 3)

            sep = self.sep_embedding.repeat(way, shot, 1, 1)
            proj_x_order1 = torch.cat((proj_x_order1, sep), -2)
            proj_x_order1 = proj_x_order1.reshape(way, (self.view_num + 1) * shot, self.d_l)
            # proj_x = proj_x.reshape(-1, 3, self.d_l)

            last_hs_order1 = self.self_att(proj_x_order1)
            last_hs_order1 = last_hs_order1.reshape(way, shot, -1, self.d_l)
            last_hs_order1 = last_hs_order1.permute(1, 0, 2, 3)

            shot_embedding_shuff = torch.cat((self.shot_embedding[1:,:], self.shot_embedding[0,:].unsqueeze(0)), 0)
            # shot_index = torch.randperm(shot)
            # shot_embedding_shuff = self.shot_embedding[shot_index]
            proj_x_order2 = proj_x + shot_embedding_shuff.repeat(way, self.view_num, 1, 1).permute(2, 0, 1, 3)
            proj_x_order2 = proj_x_order2.permute(1, 0, 2, 3)
            sep = self.sep_embedding.repeat(way, shot, 1, 1)
            proj_x_order2 = torch.cat((proj_x_order2, sep), -2)
            proj_x_order2 = proj_x_order2.reshape(way, (self.view_num + 1) * shot, self.d_l)
            last_hs_order2 = self.self_att(proj_x_order2)
            last_hs_order2 = last_hs_order2.reshape(way, shot, -1, self.d_l)
            last_hs_order2 = last_hs_order2.permute(1, 0, 2, 3)

            last_hs = torch.cat((last_hs_order1.unsqueeze(-1), last_hs_order2.unsqueeze(-1)), -1)



            # last_hs = self.self_att(proj_x)
            cross_embedding = last_hs[:,:,0:self.view_num,:,:].mean(dim=-3)
            cross_embedding = cross_embedding.view(shot, way, self.d_l, 2).permute(0,1,3,2)

            # A residual block
            temp_hs_proj = self.proj1(cross_embedding)
            last_hs_proj = self.proj2(F.dropout(F.relu(temp_hs_proj), p=self.out_dropout, training=self.training)).permute(0,1,3,2)
            last_hs_proj += cross_embedding.permute(0,1,3,2)


        else:
            # query set (way*(per_class-shot),3,dim)

            proj_x = x
            # position encoding
            # add & norm
            proj_x += position.repeat(shape[0], 1, 1)

            sep = self.sep_embedding.repeat(shape[0], 1, 1)
            proj_x = torch.cat((proj_x, sep), -2)

            proj_x = proj_x.reshape(-1, self.view_num + 1, self.d_l)


            last_hs = self.self_att(proj_x)
            last_hs = last_hs.reshape(-1, self.view_num + 1, self.d_l)
            # temp_hs_proj = self.proj1(last_hs)
            # last_hs_proj = self.proj2(F.dropout(F.relu(temp_hs_proj), p=self.out_dropout, training=self.training))
            # last_hs_proj += last_hs

            # last_hs = self.self_att(proj_x)
            cross_embedding = last_hs[:, 0:self.view_num, :].mean(dim=-2)

            # A residual block
            temp_hs_proj = self.proj1(cross_embedding)
            last_hs_proj = self.proj2(F.dropout(F.relu(temp_hs_proj), p=self.out_dropout, training=self.training))
            last_hs_proj += cross_embedding





        # A residual block
        # temp_hs_proj = self.proj1(cross_embedding)
        # last_hs_proj = self.proj2(F.dropout(F.relu(temp_hs_proj), p=self.out_dropout, training=self.training))
        # last_hs_proj += cross_embedding
        # # last_hs_proj = cross_embedding

        return last_hs_proj

class ViewFormer(nn.Module):
    def __init__(self, projetion, fusion_model):
        """
        Construct a feature_projection + fusion_model.

        input:[len, num_keywords], [len, num_actor]
        output:[len, encoder_dim]
        """
        super(ViewFormer, self).__init__()
        self.feature_projection = projetion
        self.fusion_model = fusion_model



    def forward(self, view1, view2, view3, view4, view5):
        view1_embedding, view2_embedding, view3_embedding, view4_embedding, view5_embedding = self.feature_projection(view1, view2, view3, view4, view5)
        # print('************', keywords_embedding.shape, actor_embedding.shape)
        mean_embedding = torch.cat((view1_embedding.unsqueeze(-2), view2_embedding.unsqueeze(-2), view3_embedding.unsqueeze(-2), view4_embedding.unsqueeze(-2), view5_embedding.unsqueeze(-2)), -2).mean(-2)
        x = torch.cat((view1_embedding.unsqueeze(-2), view2_embedding.unsqueeze(-2), view3_embedding.unsqueeze(-2), view4_embedding.unsqueeze(-2), view5_embedding.unsqueeze(-2)), -2)
        fusion_embedding = self.fusion_model(x)
        # print('movie_fsl_2', keywords_embedding.shape, actor_embedding.shape, mean_embedding.shape, fusion_embedding.shape)

        return view1_embedding, view2_embedding, view3_embedding, view4_embedding, view5_embedding, mean_embedding, fusion_embedding

if __name__ == '__main__':
    # encoder = MULTModel()
    x = torch.tensor(torch.rand(20, 10, 300))
    print(MULTModel(x).shape)
    print(MULTModel(x))
