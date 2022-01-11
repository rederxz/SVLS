import argparse
import os
import numpy as np
import pathlib
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import UNet3D
from datasets import get_datasets_brats, get_datasets_brats_with_skeleton, determinist_collate
from utils import seed_everything, EDiceLoss, DataAugmenter
from svls import CELossWithLS, CELossWithSVLS, CELossWithSVLS_V2, CELossWithSVLS_V3, CELossWithSVLS_V4, CELossWithSVLS_V5, CELossWithSVLS_VE

def step_train(data_loader, model, criterion, metric, optimizer): 
    model.train()   
    data_aug = DataAugmenter(p=0.5).cuda()
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        inputs = data_aug(inputs)
        segs = model(inputs)
        segs = data_aug.reverse(segs)
        if isinstance(criterion, CELossWithSVLS_V2):
            skeletons = batch["skeleton"].cuda()
            loss_ = criterion(segs, targets, skeletons)
        elif isinstance(criterion, CELossWithSVLS_VE):
            inputs = data_aug.reverse_2(inputs)
            loss_ = criterion(segs, targets, inputs)
        else:
            loss_ = criterion(segs, targets)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        
def step_valid(data_loader, model, criterion, metric):
    model.eval()
    losses, metrics = [], []
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        segs = model(inputs)
        if isinstance(criterion, CELossWithSVLS_V2):
            skeletons = batch["skeleton"].cuda()
            loss_ = criterion(segs, targets, skeletons)
        elif isinstance(criterion, CELossWithSVLS_VE):
            loss_ = criterion(segs, targets, inputs)
        else:
            loss_ = criterion(segs, targets)
        segs = segs.data.max(1)[1].squeeze_(1)
        metric_ = metric(segs.detach().cpu(), targets.detach().cpu())
        metrics.extend(metric_)
        losses.append(loss_.item())
    return np.mean(losses), metrics
    
def main():
    parser = argparse.ArgumentParser(description='SVLS Brats Training')
    parser.add_argument('--lr', default=1e-4, type=float,help='initial learning rate')
    parser.add_argument('--weight_decay', '--weight-decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batch_size', default=2, type=int,help='mini-batch size')
    parser.add_argument('--num_classes', default=4, type=int, help="num of classes")
    parser.add_argument('--in_channels', default=4, type=int, help="num of input channels")
    parser.add_argument('--svls_smoothing', default=1.0, type=float, help='SVLS smoothing factor')
    parser.add_argument('--ls_smoothing', default=0.1, type=float, help='LS smoothing factor')
    parser.add_argument('--train_option', default='SVLS', help="options:[SVLS, LS, OH, SVLS_V2]")
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_root', default='MICCAI_BraTS_2019_Data_Training/HGG_LGG', help='data directory')
    parser.add_argument('--ckpt_dir', default='ckpt_brats19', help='ckpt directory')
    parser.add_argument('--start_scale_factor', default=2.2, type=float, help='linear decaying scale factor of SVLS')
    parser.add_argument('--end_scale_factor', default=1.4, type=float, help='linear decaying scale factor of SVLS')
    parser.add_argument('--svls_alpha', default=0.1, type=float, help='alpha parameter of SVLS')
    parser.add_argument('--svls_ratio', default=1.0, type=float, help='ratio parameter of SVLS')
    parser.add_argument('--svls_sigma_diff', default=1.0, type=float, help='sigma_diff parameter of SVLS')
    parser.add_argument('--cosine_annealing_lr', action='store_true', help='whether use cosine annealing lr while training')
    args = parser.parse_args()
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    args.save_folder = pathlib.Path(args.ckpt_dir)
    args.save_folder.mkdir(parents=True, exist_ok=True)
    if args.train_option == 'SVLS_V2':
        train_dataset, val_dataset = get_datasets_brats_with_skeleton(data_root=args.data_root)
    else:
        train_dataset, val_dataset = get_datasets_brats(data_root=args.data_root)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, num_workers=2)

    print('train sample:',len(train_dataset), 'train minibatch:',len(train_loader),\
          'valid sample:',len(val_dataset), 'valid minibatch:',len(val_loader))

    model = UNet3D(inplanes=args.in_channels, num_classes=args.num_classes).cuda()

    model = torch.nn.DataParallel(model)
    criterion_dice = EDiceLoss().cuda()
    
    print('train_option',args.train_option)
    if args.train_option == 'SVLS':
        criterion = CELossWithSVLS(classes=args.num_classes, sigma=args.svls_smoothing).cuda()
        best_ckpt_name = 'model_best_svls.pth.tar'
    elif args.train_option == 'LS':
        criterion = CELossWithLS(classes=args.num_classes, smoothing=args.ls_smoothing).cuda()
        best_ckpt_name = 'model_best_ls{}'.format(args.ls_smoothing)
    elif args.train_option == 'OH':
        args.ls_smoothing = 0.0
        criterion = CELossWithLS(classes=args.num_classes, smoothing=args.ls_smoothing).cuda()
        best_ckpt_name = 'model_best_oh.pth.tar'
    elif args.train_option == 'SVLS_V2':
        criterion = CELossWithSVLS_V2(classes=args.num_classes, sigma=args.svls_smoothing).cuda()
        best_ckpt_name = 'model_best_svls_v2.pth.tar'
    elif args.train_option == 'SVLS_V3':
        criterion = CELossWithSVLS_V3(classes=args.num_classes, sigma=args.svls_smoothing, scale_factor=args.start_scale_factor).cuda()
        best_ckpt_name = 'model_best_svls_v3.pth.tar'
    elif args.train_option == 'SVLS_V4':
        criterion = CELossWithSVLS_V4(classes=args.num_classes, sigma=args.svls_smoothing, alpha=args.svls_alpha).cuda()
        best_ckpt_name = 'model_best_svls_v4.pth.tar'
    elif args.train_option == 'SVLS_V5':
        criterion = CELossWithSVLS_V5(classes=args.num_classes, sigma=args.svls_smoothing, ratio=args.svls_ratio).cuda()
        best_ckpt_name = 'model_best_svls_v5.pth.tar'
    elif args.train_option == 'SVLS_VE':
        criterion = CELossWithSVLS_VE(classes=args.num_classes, sigma_dist=args.svls_smoothing, sigma_diff=args.svls_sigma_diff).cuda()
        best_ckpt_name = 'model_best_svls_ve.pth.tar'
    else:
        raise ValueError(args.train_option)
    
    print('ckpt name:', best_ckpt_name)
    best_ckpt_dir = os.path.join(str(args.save_folder),best_ckpt_name)
    metric = criterion_dice.metric_brats
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
    if args.cosine_annealing_lr:
        print('Using cosine annealing lr...')
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss, best_epoch, best_dices = np.inf, 0, [0,0,0]
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if isinstance(criterion, CELossWithSVLS_V3):  # adjust scale factor
            criterion.scale_factor = args.start_scale_factor - (epoch / args.epochs * (args.start_scale_factor - args.end_scale_factor))
            print(f"Current svls scale factor: {criterion.scale_factor}")
        step_train(train_loader, model, criterion, metric, optimizer)
        with torch.no_grad():
            validation_loss, dice_metrics = step_valid(val_loader, model, criterion, metric)
            dice_metrics = list(zip(*dice_metrics))
            dice_metrics = [torch.tensor(dice, device="cpu").numpy() for dice in dice_metrics]
            avg_dices = np.mean(dice_metrics,1)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_dices = avg_dices
            torch.save(dict(epoch=epoch, arhi='unet',state_dict=model.state_dict()),best_ckpt_dir )
        print('epoch:%d/%d, loss:%.4f, best epoch:%d, best loss:%.4f, dice[ET:%.4f, TC:%.4f, WT:%.4f], best dice[ET:%.4f, TC:%.4f, WT:%.4f]' \
                %(epoch, args.epochs, validation_loss, best_epoch, best_loss, avg_dices[0], avg_dices[1], avg_dices[2], best_dices[0], best_dices[1], best_dices[2]))
        if args.cosine_annealing_lr:
            scheduler.step()

if __name__ == '__main__':
    seed_everything()
    main()
