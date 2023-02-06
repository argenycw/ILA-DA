'''
attack_def.py
The same as attack.py, but used for attacking defended models
- run ILA-DA directly without running a reference attack
- save the output as images instead of tensor batches
- some default args are different (e.g. epsilon, surrogate model, etc.)
'''

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import numpy as np

from utils import *
from models import *

import timm

name_to_model = {
    "resnet50": models.resnet50(pretrained=True),
    "inception_v3": models.inception_v3(pretrained=True),
    'adv_inception_v3': timm.create_model('adv_inception_v3', pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
}

# augmentation: reverse adversarial update                                
aug_dict = {
    't': [T.RandomAffine(degrees=0, translate=(0.1, 0.1))],
    'r': [T.RandomAffine(90)],
    'c': [T.RandomResizedCrop(299, scale=(0.95, 0.95))],
    'j': [T.ColorJitter(0.2, 0.2, 0.2, 0.1)],
}

def build_foldername(args):    
    conv = {
        'batch_size': 'bs', 
        'epsilon': 'eps',
        'model_type': '',
        'ila_layer': 'l',
        'adv_dir': 'src-',
    }
    skip = ['max_batch', 'step_size', 'agg_iter', 'save_dir', 'clean_dir']
    foldername = ''
    for k, v in args.__dict__.items():
        if k in skip or not v:
            continue
        if k in conv:
            k = conv[k]
        if k == 'src-':
            v = os.path.basename(os.path.normpath(v))
        if type(v) == bool and v:
            foldername += str(k) + '_'
        else:
            foldername += str(k) + str(v) + '_'
    return foldername[:-1] # remove the ending underscore


def ila_forw_by_models(model, model_type, x, ila_layer):
    if model_type in ('resnet50', 'resnet101'):
        return ila_forw_resnet50(model, x, ila_layer)
    elif model_type in ('vgg16', 'vgg19'):
        assert '_' not in ila_layer, 'You should give a number (1-15) to ila_layer for VGG19. ila_layer: {}'.format(ila_layer)
        layer = int(ila_layer)
        return ila_forw_vgg(model, x, layer)
    elif model_type == 'inception_v3':
        return ila_forw_inception_v3(model, x, ila_layer)
    elif model_type == 'adv_inception_v3':
        return ila_forw_adv_inception_v3(model, x, ila_layer)
    raise Exception("Non-supported model type: {}".format(model_type))

if __name__ == '__main__':

    set_seed(0)
    parser = argparse.ArgumentParser(description='For Generation of Transferrable Attack')
    parser.add_argument('--clean_dir', type=str, default='data/defended_val', help='location of the clean examples')
    parser.add_argument('-f', '--adv_dir', type=str, default='', help='location of the reference adversarial examples')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.063, help="The perturbation budget")
    parser.add_argument('--max_batch', type=int, default=-1, help="The number of batch to stop early (default: -1 = not to stop)")
    parser.add_argument('--save_dir', type=str, default="data/for_defended", help="The path to save the output tensors for validation")
    parser.add_argument('-m', '--model_type', type=str, default='inception_v3', help="resnet50/vgg19/inception_v3")
    # options for ILA    
    parser.add_argument('--niters', type=int, default=10, help="The number of attack iteration")
    parser.add_argument('--ila', type=int, default=500, help="The number of ILA iteration after the attack generation")
    parser.add_argument('--ila_layer', type=str, default='6a', help="The layer to perform ILA (note the format is different for different model)")    
    parser.add_argument('--step_size', type=float, default=1./255., help="The step size of both I-FGSM and ILA")    
    parser.add_argument('--pgd', action='store_true', help="to activate random initialization (PGD)")
    
    # options for ILA-DA
    parser.add_argument('-a', '--alpha', type=float, default=-1, help="The alpha value for attack interpolation (< 0 for being adaptive)")    
    parser.add_argument('--aug', type=str, default='atrc', help="t/c/r/j/a (traslation/cropping/rotation/color jitter/adversarial)")        
    
    # options for auto aug
    parser.add_argument('--search', type=str, default='gumbel', help="to activate augmentation searach")
    parser.add_argument('--eta', type=float, default=5e-05, help="The step size of augmentation update")

    args = parser.parse_args()

    save_dir = args.save_dir
    niters = args.niters
    epsilon = args.epsilon
    step_size = args.step_size
    batch_size = args.batch_size
    alpha = args.alpha
    ila_layer = args.ila_layer
    model_type = args.model_type

    assert args.adv_dir != '', 'The argument (-f, --adv_dir) cannot be empty'

    if (step_size < epsilon/niters):
        print("Step size is smaller than epsilon / niters, the attack will not be the strongest.")
        step_size = epsilon / niters
        print("Replacing step size to be epsilon / niter = {:.4f}...".format(step_size))

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')    

    tform = T.Compose([
        #T.Resize((256,256)),
        #T.CenterCrop((224,224)),
        T.Resize((299)),
        T.ToTensor()
    ])

    #selected_data = 'data/selected_data_full.csv'
    clean_dir = args.clean_dir    

    #dataset = SelectedImagenet(val_datapath, selected_data, tform)
    dataset = DuoImageFolders(tform, args.clean_dir, args.adv_dir, label_dir='data/val_rs.csv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = name_to_model[model_type] #models.resnet50(pretrained=True)
    model = nn.Sequential(Normalize(), model).to(device)
    model.eval()

    # create the directory for the specific setup
    folder_name = build_foldername(args)
    save_dir = os.path.join(save_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # iterate every characters in args.aug to append every augmentation
    aug_list = []
    rev_update = False
    for t in args.aug:
        if t in aug_dict:
            aug_list += aug_dict[t]
        elif t == 'a':
            rev_update = True
        elif t == '_':
            pass
        else:
            raise Exception('Undefined augmentation method: {}'.format(args.aug))

    if args.search == 'gumbel':
        # initialize p
        p = torch.tensor([1.0]*len(aug_list), requires_grad=True)
        optimizer = torch.optim.SGD([p], args.eta, momentum=0.9, 
                                    weight_decay=5e-4, nesterov=True) 
        p_hist = []
        print('Initialized augmentation probability for gumbel softmax')
    else:
        t_aug = T.Compose(aug_list)

    epsilon_warning = False
    label_list = []
    for i, (x, x_adv, filenames, label) in enumerate(dataloader):
        if args.max_batch > 0 and i >= args.max_batch: 
            print('Early stopping at batch {}'.format(args.max_batch))
            break

        label_list.append(label)
        x = x.to(device)        
        x_adv = x_adv.to(device)
        label = label.to(device)
        #x_adv = x.clone()

        # clip epsilon for the current adv. examples
        linf_norm = (x_adv - x).abs().max().item()
        x_adv = clip_epsilon(x_adv, x, epsilon)
        linf_norm_new = (x_adv - x).abs().max().item()
        if not epsilon_warning:
            print('L-inf Norm: {:.3f} => {:.3f}'.format(linf_norm, linf_norm_new))
            epsilon_warning = True

        # Inject ILA
        x_a = x.clone()
        if args.ila > 0:
            ila_niters = args.ila
            attack_img = x_adv.detach().clone()
            img = x.clone().to(device)
            
            if args.search == 'gumbel':
                batch_p_hist = []
                # initialize p for batch update
                # p = torch.tensor([1.0]*len(aug_list), requires_grad=True)
                # optimizer = torch.optim.SGD([p], args.eta, momentum=0.9, 
                #                             weight_decay=5e-4, nesterov=True)

            for ila_i in range(ila_niters):
                img.requires_grad_(True)
                model.zero_grad()

                if rev_update:
                    # the following also works, but we prefer consideration of more attacks
                    # x_a = 2.0 * x - x_advs[-1]
                    #x_a = 2.0 * x - x_advs[int(5 - ila_i / 5)]
                    x_a = 2.0 * x - x_adv

                x_combined = torch.cat([x_a, attack_img, img], dim=0)
                if args.search == 'gumbel':
                    batch_p_hist.append(p.clone())
                    optimizer.zero_grad()
                    gp = F.gumbel_softmax(p, tau=1, hard=True)
                    aug_id = torch.argmax(gp, dim=0)
                    x_aug = sum(w * T.Compose([op])(x_combined) if i == aug_id else 
                                (w * x_combined) for i, (w, op) in enumerate(zip(gp, aug_list)))
                elif args.search == 'random':
                    aug_id = random.randint(0, len(aug_list)-1)
                    x_aug = T.Compose([aug_list[aug_id]])(x_combined)
                else:
                    x_aug = t_aug(x_combined)
                with torch.no_grad():
                    x_mid = ila_forw_by_models(model, model_type, x_aug[:x.shape[0]], ila_layer)       
                    x_adv_mid = ila_forw_by_models(model, model_type, x_aug[x.shape[0]:x.shape[0]*2], ila_layer)             
                x_adv2_mid = ila_forw_by_models(model, model_type, x_aug[x.shape[0]*2:], ila_layer)

                loss = -ILAProjLoss()(x_adv_mid, x_adv2_mid, x_mid, 0.0) 
                
                loss.backward()
                input_grad = img.grad.data

                img = img.data - step_size * torch.sign(input_grad)
                img = clip_epsilon(img, x, epsilon)
                
                # update augmentation parameters
                if args.search == 'gumbel':
                    optimizer.step()

                # interpolation update on X
                with torch.no_grad():
                    if args.alpha < 0.0: # negative alpha => let the norm determine it
                        x_adv_if = ila_forw_by_models(model, model_type, img, ila_layer)   
                        old_norm = (x_adv_mid - x_mid).norm()
                        new_norm = (x_adv_if - x_mid).norm()
                        alpha = new_norm / (new_norm + old_norm)
                    elif args.alpha == 0.0:
                        continue
                    attack_img = alpha * img + (1 - alpha) * attack_img

            x_adv = img
            x_adv = clip_epsilon(x_adv, x, epsilon)
            
            if args.search == 'gumbel':
                p_hist.append(batch_p_hist)

        # we save the images as JPEG file to be compatible with the defended codes
        save_img(x_adv.detach().cpu(), os.path.join(save_dir, 'images'), filenames)
    torch.save(label_list, os.path.join(save_dir, 'label.pt'))
    if args.search == 'gumbel':
        torch.save(p_hist, os.path.join(save_dir, 'aug_p_history.pt'))
    print('Labels and all batches are saved at: {}'.format(save_dir))
