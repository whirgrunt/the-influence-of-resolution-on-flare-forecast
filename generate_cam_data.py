from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
import load_flare
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
parser.add_argument('--model_name', type=str, default='res18')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--resize_scale', type=int, default=512)
parser.add_argument('--data_aug', type=bool, default=False)
# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
parser.add_argument('--feature_extract', type=bool, default=False)
args = parser.parse_args()


def test_model(model, dataloaders, criterion, num_epochs):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch = num_epochs


    # Each epoch has a training and validation phase
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    run_time = time.time()


    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    y_score = []
    y_test = []
    #y_test = targ_labels.numpy()
    #y_score = pred_scores.numpy()
    #np.save('df', [y_test, y_score])
    for inputs, gt in dataloaders:
        inputs = inputs.cuda()
        gt = np.array(gt)
        labels = torch.from_numpy(gt)
        labels = labels.cuda()

        outputs = model(inputs)
        out = torch.nn.functional.softmax(outputs, dim=1)
        out = np.array(out.cpu().detach().numpy())
        #max_out = np.max(out,axis=1)
        out_right = out[:,1]
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        a = running_corrects.double()
        # TP    predict   label =1
        TP += ((preds == 1) & (labels.data == 1)).cpu().sum().to(a)
        # TN    predict   label =0
        TN += ((preds == 0) & (labels.data == 0)).cpu().sum().to(a)
        # FN    predict 0 label 1
        FN += ((preds == 0) & (labels.data == 1)).cpu().sum().to(a)
        # FP    predict 1 label 0
        FP += ((preds == 1) & (labels.data == 0)).cpu().sum().to(a)
        # p = TP / (TP + FP)
        # r = TP / (TP + FN)
        # F1 = 2 * r * p / (r + p)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)
        acc = (TP + TN) / (TP + TN + FP + FN)

        y_score.extend(out_right)
        y_test.extend(labels.cpu().numpy())

    print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.2f}s'.format('val', epoch_loss, epoch_acc, run_time))
    print('TP: {:4f}, TN: {:4f}, FN: {:4f}, FP: {:4f}'.format(TP, TN, FN, FP))
    print('TPR: {:4f}, TNR: {:4f}, FPR: {:4f}, FNR: {:4f}, acc: {:4f}'.format(TPR, TNR, FPR, FNR, acc))
    print("\n")

    np.save(str(num_epochs) + '_df', [y_test, y_score])







def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    model_path = "/mnt/disk100T/lsx/flare-prediction/rst/exp4/"
    list = os.listdir(model_path)

    data_transforms = {
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.resize_scale, args.resize_scale), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create validation datasets
    image_datasets = {'val': load_flare.flare_test(split='test', transform=data_transforms['val'])}

    dataloaders_dict = torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # evaluate/test
    for epoch in range(args.num_epochs):
        if epoch<10:
            str_num = '0' + str(epoch) + '_'
        str_match = [s for s in list if str_num in s]
        model_ft = torch.load(model_path + str_match[0])
        model_ft = model_ft.to(device)
        test_model(model_ft, dataloaders_dict, criterion, epoch)