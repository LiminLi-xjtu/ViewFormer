import torch

def save_load_name(args, name=''):
    name = name
    if (args.view_encoding == False) & (args.shot_encoding == False):
        encoding = 'base'
    elif (args.view_encoding == True) & (args.shot_encoding == False):
        encoding = 'view'
    elif (args.view_encoding == False) & (args.shot_encoding == True):
        encoding = 'shot'
    elif (args.view_encoding == True) & (args.shot_encoding == True):
        encoding = 'view_shot'

    return name + '_' + encoding + '_' + str(args.num_heads) + 'heads_' + str(args.nlevels) + 'layer_' + str(args.way) + '-way' + str(args.shot) + '-shot'


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    model_name = args.model_root + f'/{name}.pt'
    torch.save(model, model_name)

def load_model(args, name=''):
    name = save_load_name(args, name)
    model_name = args.model_root + f'/{name}.pt'
    model = torch.load(model_name)
    return model
