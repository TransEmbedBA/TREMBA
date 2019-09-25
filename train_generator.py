import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from Normalize import Normalize, Permute
import DataLoader
import numpy as np
from FCN import *
from utils import *
import torchvision.models as models
import copy
from imagenet_model.Resnet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, nargs="+", default=[0,1,2,3])
    parser.add_argument('--config', default='config_joint_train_Imagenet_GAN.json', help='config file')

    args = parser.parse_args()

    with open(args.config) as config_file:
        state = json.load(config_file)
    device = torch.device(args.device[0])

    train_loader, test_loader, nlabels, mean, std = DataLoader.imagenet(state)
    nets = []
    for model_name in state['model_name']:
        print(model_name)
        if model_name == "VGG16":
            pretrained_model = models.vgg16_bn(pretrained=True)
        elif model_name == 'Resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'Squeezenet':
            pretrained_model = models.squeezenet1_1(pretrained=True)
        elif model_name == 'Googlenet':
            pretrained_model = models.googlenet(pretrained=True)
        elif model_name == 'Adv_Denoise_Resnet152':
            pretrained_model = resnet152_denoise()
            loaded_state_dict = torch.load(os.path.join('weight', model_name+".pytorch"))
            pretrained_model.load_state_dict(loaded_state_dict)
        if 'defense' in state and state['defense']:
            net = nn.Sequential(
                Normalize(mean, std),
                Permute([2,1,0]),
                pretrained_model
            )
        else:
            net = nn.Sequential(
                Normalize(mean, std),
                pretrained_model
            )
        nets.append(net)

    model = nn.Sequential(
        Imagenet_Encoder(),
        Imagenet_Decoder()
    )

    for i in range(len(nets)):
        nets[i] = torch.nn.DataParallel(nets[i], args.device)
        nets[i].eval()
        nets[i].to(device)

    model = torch.nn.DataParallel(model, args.device)
    model.to(device)
    print(model)

    optimizer_G = torch.optim.SGD(model.parameters(), state['learning_rate_G'], momentum=state['momentum'],
                                weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state['epochs'] // state['schedule'],
                                                gamma=state['gamma'])

    hingeloss = MarginLoss(margin=state['margin'], target=state['target'])

    if state['target']:
        save_name = "Imagenet_{}{}_target_{}.pytorch".format("_".join(state['model_name']), state['save_suffix'], state['target_class'])
    else:
        save_name = "Imagenet_{}{}_untarget.pytorch".format("_".join(state['model_name']), state['save_suffix'])

    def train():
        model.train()

        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.to(device)
            if state['target']:
                label = state['target_class']
            else:
                label = label.to(device)

            losses_g = []
            optimizer_G.zero_grad()
            for net in nets:
                noise = model(nat)
                adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                losses_g.append("%4.2f" % loss_g.item())
                loss_g.backward()
            optimizer_G.step()

            if (batch_idx + 1) % state['log_interval'] == 0:
                print("batch {}, losses_g {}".format(batch_idx + 1, dict(zip(state['model_name'], losses_g))))

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]

        for batch_idx, (data, label) in enumerate(test_loader):
            nat = data.to(device)
            if state['target']:
                label = state['target_class']
            else:
                label = label.to(device)
            noise = model(nat)
            adv = torch.clamp(noise * state['epsilon'] + nat, 0, 1)
            
            for j in range(len(nets)):
                logits = nets[j](adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                if state['target']:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1) == label).sum())

        state['test_loss'] = [loss_avg[i] / len(test_loader) for i in range(len(loss_avg))]
        state['test_successes'] = [success[i] / len(test_loader.dataset) for i in range(len(success))]
        state['test_success'] = 0.0
        for i in range(len(state['test_successes'])):
            state['test_success'] += state['test_successes'][i]/len(state['test_successes'])

    best_success = 0.0
    for epoch in range(state['epochs']):
        scheduler_G.step()
        state['epoch'] = epoch
        train()
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            with torch.no_grad():
                test()
            print(state)

        torch.save(model.module.state_dict(), os.path.join("G_weight", save_name))
        print("epoch {}, Current success: {}, Best success: {}".format(epoch, state['test_success'], best_success))