'''
ILA-DA on CIFAR10 and CIFAR100

Remark:
python attack_cifar.py --batch_size 200 --dataset cifar100 -m vgg19 --ila_layer 9  --aug ac -a -1
'''

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
#import torchvision.models as models
import bearpaw_models.cifar as ex_models
import torchvision.datasets as datasets
import numpy as np

from utils import *
from models import *

# build the model and load any state_dict
def create_model(arch, num_classes, dataset):
    model_state_path = 'data/{}'.format(dataset)
    print("creating model '{}'".format(arch))
    if arch == 'vgg19':
        model = ex_models.vgg19_bn(num_classes=num_classes)
        filepath = 'vgg19_bn/model_best.pth.tar'
    elif arch == 'wrn':
        model = ex_models.wrn(num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
        filepath = 'WRN-28-10-drop/model_best.pth.tar'
    elif arch == 'resnext': # ResNeXt-29, 8x64d
        model = ex_models.resnext(num_classes=num_classes, cardinality=8, depth=29, widen_factor=4, dropRate=0)
        filepath = 'resnext-8x64d/model_best.pth.tar'
    elif arch == 'densenet':
        model = ex_models.densenet(num_classes=num_classes, depth=190, growthRate=40, compressionRate=2, dropRate=0)
        filepath = 'densenet-bc-L190-k40/model_best.pth.tar'
    else:
        raise Exception('Undefined model: {}'.format(arch))
       
    model_dict = torch.load(os.path.join(model_state_path, filepath))
    new_model_dict = {}
    for key, val in model_dict['state_dict'].items():
        if 'module.' in key:
            new_key = key.replace('module.', '')
            new_model_dict[new_key] = val
        else:
            new_model_dict[key] = val
    # epoch, state_dict, acc, best_acc, optimizer
    model.load_state_dict(new_model_dict)
    return model

# augmentation: reverse adversarial update                                
aug_dict = {
    't': T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    'r': T.RandomAffine(90),
    'c': T.RandomResizedCrop(32, scale=(0.95, 0.95)),
    'j': T.ColorJitter(0.2, 0.2, 0.2, 0.1),
}

def build_foldername(args):    
    conv = {
        'batch_size': 'bs', 
        'epsilon': 'eps',
        'model_type': '',
        'ila_layer': 'l',
    }
    skip = ['max_batch', 'step_size', 'agg_iter', 'save_dir']
    foldername = ''
    for k, v in args.__dict__.items():
        if k in skip or not v:
            continue
        if k in conv:
            k = conv[k]        
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
    raise Exception("Non-supported model type: {}".format(model_type))

if __name__ == '__main__':

    set_seed(0)
    parser = argparse.ArgumentParser(description='For Generation of Transferrable Attack')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--epsilon', type=float, default=0.03, help="The perturbation budget")
    parser.add_argument('--max_batch', type=int, default=-1, help="The number of batch to stop early (default: -1 = not to stop)")
    parser.add_argument('--save_dir', type=str, default="data/attack_batches", help="The path to save the output tensors for validation")
    parser.add_argument('-m', '--model_type', type=str, default='vgg19', help="resnet110/vgg19")
    # options for ILA    
    parser.add_argument('--niters', type=int, default=10, help="The number of attack iteration")
    parser.add_argument('--ila', type=int, default=50, help="The number of ILA iteration after the attack generation")
    parser.add_argument('--ila_layer', type=str, default='9', help="The layer to perform ILA (note the format is different for different model)")    
    parser.add_argument('--step_size', type=float, default=1./255., help="The step size of both I-FGSM and ILA")    
    parser.add_argument('--pgd', action='store_true', help="to activate random initialization (PGD)")
    # options for ILA-DA
    parser.add_argument('-a', '--alpha', type=float, default=-1, help="The alpha value in momentum (< 0 for being adaptive)")    
    parser.add_argument('--aug', type=str, default='atrc', help="t/c/r/j/a (traslation/scaling/rotation/color jitter/adversarial)")
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

    if (step_size < epsilon/niters):
        print("Step size is smaller than epsilon / niters, the attack will not be the strongest.")
        step_size = epsilon / niters
        print("Replacing step size to be epsilon / niter = {:.4f}...".format(step_size))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    

    tform = T.Compose([
        T.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        dataset = SelectedCifar(path="data/cifar10_batches.pt", labels="data/cifar10_labels.pt")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_class = 10
    elif args.dataset == 'cifar100':
        dataset = SelectedCifar(path="data/cifar100_batches.pt", labels="data/cifar100_labels.pt")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_class = 100
    else:
        raise Exception("Undefined dataset: {}".format(args.dataset))


    model = create_model(model_type, num_class, args.dataset) #models.resnet50(pretrained=True)
    model = nn.Sequential(Normalize([(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]), model).to(device)
    model.eval()

    # iterate every characters in args.aug to append every augmentation
    aug_list = []
    for t in args.aug:
        if t in aug_dict:
            aug_list.append(aug_dict[t])
        elif t == 'a':
            rev_update = True
        elif t == '_':
            pass
        else:
            raise Exception('Undefined augmentation method: {}'.format(args.aug))
    
    #os.makedirs(save_dir, exist_ok=True)
    if args.search == 'gumbel':
        # initialize p
        p = torch.tensor([1.0]*len(aug_list), requires_grad=True)
        optimizer = torch.optim.SGD([p], args.eta, momentum=0.9, 
                                    weight_decay=5e-4, nesterov=True) 
        p_hist = []
        print('Initialized augmentation probability for gumbel softmax')
    else:
        t_aug = T.Compose(aug_list)

    # create the directory for the specific setup
    foldername = build_foldername(args)    
    save_dir = os.path.join(save_dir, foldername)
    os.makedirs(save_dir, exist_ok=True)

    atk_list = []
    label_list = []
    for i, (x, label) in enumerate(dataloader):
        if args.max_batch > 0 and i >= args.max_batch: 
            print('Early stopping at batch {}'.format(args.max_batch))
            break

        label_list.append(label)
        x = x.to(device)        
        label = label.to(device)
        x_adv = x.clone()

        # generate attack for the current batch
        #x_adv = ilapp_attack(False, None, x_adv, label, device, niters, 'ifgsm', epsilon, model, '2_3', batch_size, step_size)
        x_advs = torch.zeros((niters, x_adv.shape[0], x_adv.shape[1], x_adv.shape[2], x_adv.shape[3])).to(device)
        for atk_i in range(niters): 
            if not args.pgd:           
                x_adv = x_adv # I-FGSM
            else:
                x_adv = x_adv + x_adv.new(x_adv.size()).uniform_(-epsilon, epsilon)
            x_adv.requires_grad_(True)
            x_advs[atk_i] = x_adv.data

            out = model(x_adv)
            pred = torch.argmax(out, dim=1).view(-1)
            loss = nn.CrossEntropyLoss()(out, label)
            model.zero_grad()
            loss.backward()
            x_grad = x_adv.grad.data  
            
            x_adv = x_adv.data + step_size * torch.sign(x_grad)
            x_adv = clip_epsilon(x_adv, x, epsilon)

        # Inject ILA
        rev_update = False
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
                #                      

            for ila_i in range(ila_niters):
                img.requires_grad_(True)
                model.zero_grad()

                if rev_update:
                    # the following also works, but we prefer consideration of more attacks
                    # x_a = 2.0 * x - x_advs[-1]
                    x_a = 2.0 * x - x_advs[int(5 - ila_i / 5)]

                x_combined = torch.cat([x_a, attack_img, img], dim=0)

                if args.search == 'gumbel':
                    batch_p_hist.append(p.clone())
                    optimizer.zero_grad()
                    gp = F.gumbel_softmax(p, tau=1, hard=True)
                    aug_id = torch.argmax(gp, dim=0)
                    x_aug = sum(w * T.Compose([op])(x_combined) if i == aug_id else 
                                w*x_combined for i, (w, op) in enumerate(zip(gp, aug_list)))
                elif args.search == 'random':
                    aug_id = random.randint(0, len(aug_list)-1)
                    x_aug = T.Compose([aug_list[aug_id]])(x_combined)
                else:                
                    x_aug = t_aug(x_combined)
                with torch.no_grad():
                    #x_mid = ila_forw_resnet50(model, x_aug[:x.shape[0]], ila_layer)
                    #x_adv_mid = ila_forw_resnet50(model, x_aug[x.shape[0]:x.shape[0]*2], ila_layer)
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
                        #x_adv_if = ila_forw_resnet50(model, img, ila_layer)
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

        # we save the images as np arrays directly, to avoid any conversion mistakes        
        np.save(save_dir + '/batch_{}.npy'.format(i), torch.round(x_adv.data*255).cpu().numpy())
        print('batch_{}.npy saved'.format(i))
    label_ls = torch.cat(label_list)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('all batches saved in {}'.format(save_dir))

