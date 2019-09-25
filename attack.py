import argparse
import torchvision.models as models
import os
import json
import DataLoader
from utils import *
from FCN import *
from Normalize import Normalize, Permute
from imagenet_model.Resnet import resnet152_denoise, resnet101_denoise


def EmbedBA(function, encoder, decoder, image, label, config, latent=None):
    device = image.device

    if latent is None:
        latent = encoder(image.unsqueeze(0)).squeeze().view(-1)
    momentum = torch.zeros_like(latent)
    dimension = len(latent)
    noise = torch.empty((dimension, config['sample_size']), device=device)
    origin_image = image.clone()
    last_loss = []
    lr = config['lr']
    for iter in range(config['num_iters']+1):

        perturbation = torch.clamp(decoder(latent.unsqueeze(0)).squeeze(0)*config['epsilon'], -config['epsilon'], config['epsilon'])
        logit, loss = function(torch.clamp(image+perturbation, 0, 1), label)
        if config['target']:
            success = torch.argmax(logit, dim=1) == label
        else:
            success = torch.argmax(logit, dim=1) !=label
        last_loss.append(loss.item())

        if function.current_counts > 50000:
            break
        
        if bool(success.item()):
            return True, torch.clamp(image+perturbation, 0, 1)

        nn.init.normal_(noise)
        noise[:, config['sample_size']//2:] = -noise[:, :config['sample_size']//2]
        latents = latent.repeat(config['sample_size'], 1) + noise.transpose(0, 1)*config['sigma']
        perturbations = torch.clamp(decoder(latents)*config['epsilon'], -config['epsilon'], config['epsilon'])
        _, losses = function(torch.clamp(image.expand_as(perturbations) + perturbations, 0, 1), label)

        grad = torch.mean(losses.expand_as(noise) * noise, dim=1)

        if iter % config['log_interval'] == 0 and config['print_log']:
            print("iteration: {} loss: {}, l2_deviation {}".format(iter, float(loss.item()), float(torch.norm(perturbation))))

        momentum = config['momentum'] * momentum + (1-config['momentum'])*grad

        latent = latent - lr * momentum

        last_loss = last_loss[-config['plateau_length']:]
        if (last_loss[-1] > last_loss[0]+config['plateau_overhead'] or last_loss[-1] > last_loss[0] and last_loss[-1]<0.6) and len(last_loss) == config['plateau_length']:
            if lr > config['lr_min']:
                lr = max(lr / config['lr_decay'], config['lr_min'])
            last_loss = []

    return False, origin_image


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.json', help='config file')
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--model_name', default=None)
args = parser.parse_args()

with open(args.config) as config_file:
    state = json.load(config_file)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
if args.model_name is not None:
    state['model_name'] = args.model_name

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

weight = torch.load(os.path.join("G_weight", state['generator_name']+".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val

_, dataloader, nlabels, mean, std = DataLoader.imagenet(new_state)
if 'OSP' in state:
    if state['source_model_name'] == 'Adv_Denoise_Resnet152':
        s_model = resnet152_denoise()
        loaded_state_dict = torch.load(os.path.join('weight', state['source_model_name']+".pytorch"))
        s_model.load_state_dict(loaded_state_dict)
    if 'defense' in state and state['defense']:
        source_model = nn.Sequential(
            Normalize(mean, std),
            Permute([2,1,0]),
            s_model
        )
    else:
        source_model = nn.Sequential(
            Normalize(mean, std),
            s_model
        )

if state['model_name'] == 'Resnet34':
    pretrained_model = models.resnet34(pretrained=True)
elif state['model_name'] == 'VGG19':
    pretrained_model = models.vgg19_bn(pretrained=True)
elif state['model_name'] == 'Densenet121':
    pretrained_model = models.densenet121(pretrained=True)
elif state['model_name'] == 'Mobilenet':
    pretrained_model = models.mobilenet_v2(pretrained=True)
elif state['model_name'] == 'Adv_Denoise_Resnext101':
    pretrained_model = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', state['model_name']+".pytorch"))
    pretrained_model.load_state_dict(loaded_state_dict, strict=True)
if 'defense' in state and state['defense']:
    model = nn.Sequential(
        Normalize(mean, std),
        Permute([2,1,0]),
        pretrained_model
    )
else:
    model = nn.Sequential(
        Normalize(mean, std),
        pretrained_model
    )

encoder = Imagenet_Encoder()
decoder = Imagenet_Decoder()
encoder.load_state_dict(encoder_weight)
decoder.load_state_dict(decoder_weight)

model.to(device)
model.eval()
encoder.to(device)
encoder.eval()
decoder.to(device)
decoder.eval()

if 'OSP' in state:
    source_model.to(device)
    source_model.eval()

F = Function(model, state['batch_size'], state['margin'], nlabels, state['target'])

count_success = 0
count_total = 0
if not os.path.exists(state['save_path']):
    os.mkdir(state['save_path'])

for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = int(labels)
    logits = model(images)
    correct = torch.argmax(logits, dim=1) == labels
    if correct:
        torch.cuda.empty_cache()
        if state['target']:
            labels = state['target_class']

        if 'OSP' in state:
            hinge_loss = MarginLoss_Single(state['white_box_margin'], state['target'])
            images.requires_grad = True
            latents = encoder(images)
            for k in range(state['white_box_iters']):     
                perturbations = decoder(latents)*state['epsilon']
                logits = source_model(torch.clamp(images+perturbations, 0, 1))
                loss = hinge_loss(logits, labels)
                grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - state['white_box_lr'] * grad

            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state, latents.view(-1))

        else:
            with torch.no_grad():
                success, adv = EmbedBA(F, encoder, decoder, images[0], labels, state)

        count_success += int(success)
        count_total += int(correct)
        print("image: {} eval_count: {} success: {} average_count: {} success_rate: {}".format(i, F.current_counts, success, F.get_average(), float(count_success) / float(count_total)))
        F.new_counter()

success_rate = float(count_success) / float(count_total)
if state['target']:
    np.save(os.path.join(state['save_path'], '{}_class_{}.npy'.format(state['save_prefix'], state['target_class'])), np.array(F.counts))
else:
    np.save(os.path.join(state['save_path'], '{}.npy'.format(state['save_prefix'])), np.array(F.counts))
print("success rate {}".format(success_rate))
print("average eval count {}".format(F.get_average()))
