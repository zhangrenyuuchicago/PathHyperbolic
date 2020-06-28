import torchvision
import torch
import SlideDataset
import pickle
import ntpath
import os
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from optparse import OptionParser
from datetime import datetime
from Model import MINet
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical

import uuid
uid = uuid.uuid1()

usage = "usage: python main.py "
parser = OptionParser(usage)

parser.add_option("-l", "--learning_rate", dest="learning_rate", type="float", default=0.0001,
                    help="set learning rate for optimizor")
parser.add_option("-m", "--mu", dest="mu", type="float", default=1.0,
                    help="set mu")
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=32,
                    help="batch size")
parser.add_option("-o", "--output", dest="output", type="string", default="specified_format.csv",
                    help="output file")
parser.add_option("-r", "--resume", dest="model_file", type="string", default="",
                    help="resume the file from a model file")
parser.add_option("-f", "--fold", dest="fold", type="int", default=0,
                    help="fold in [0,1,2,3,4]")

(options, args) = parser.parse_args()

fold = options.fold
batch_size = options.batch_size
mu = options.mu
epoch_num = 2000
sample_num_each_epoch = 500
patience = 25
best_epoch = 0
best_val_auc = 0.0
test_times = 20

if options.model_file == "":
    minet = MINet()
    minet = torch.nn.DataParallel(minet).cuda()
else:
    print('not implemented')

learning_rate = options.learning_rate

optimizer = torch.optim.Adam([
            {'params': minet.parameters()}
        ],
        lr=learning_rate)

nn_loss_label = torch.nn.CrossEntropyLoss().cuda()

loss = 0

train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ColorJitter(brightness=64.0/255, contrast=0.75, saturation=0.25, hue=0.04),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor()
            ])

val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor()
            ])

train_image_data = SlideDataset.SlideDataset('train', '../../settings/train_label.csv', 
            '../../gen_tiles_1000/train_20X',
            train_transform, fold)

weight = np.loadtxt('weight.txt')
train_sampler = torch.utils.data.WeightedRandomSampler(weight, num_samples= sample_num_each_epoch, replacement=True)
train_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=batch_size, drop_last=False)
train_data_loader = torch.utils.data.DataLoader(train_image_data, num_workers=5, batch_sampler= train_sampler)
#train_data_loader = torch.utils.data.DataLoader(train_image_data, num_workers=5, batch_size=batch_size)

val_image_data = SlideDataset.SlideDataset('val', '../../settings/train_label.csv', 
            '../../gen_tiles_1000/train_20X',
            val_transform, fold)
val_data_loader = torch.utils.data.DataLoader(val_image_data, num_workers=5, batch_size=5)

from sklearn.metrics import roc_auc_score, accuracy_score

def get_acc(pred, var):
    max_value, index = torch.max(pred, 1)
    index = index.data.cpu().numpy()
    var = var.data.cpu().numpy()
    return np.sum(index == var)*1.0/index.shape[0]

writer = SummaryWriter(logdir="logs/MINet_hashtag_cluster_" + datetime.now().strftime('%b%d_%H-%M-%S'))
step = 0

for epoch in range(epoch_num):
    print("Epoch: " + str(epoch))
    sum_label_acc = 0.0
    sum_loss = 0.0
    count = 0

    gender_true_lt = []
    gender_pred_lt = []

    minet.train()

    for id, (item, label, img_name) in enumerate(train_data_loader):
        item = item.cuda()
        label = label.cuda()
        input_var = torch.autograd.Variable(item, requires_grad=True)
        label_var = torch.autograd.Variable(label.squeeze(dim=1))

        optimizer.zero_grad()
        label_pred = minet(input_var) 
        
        loss_label = nn_loss_label(label_pred, label_var)
        loss = loss_label 
        
        loss.backward()
        optimizer.step()
        
        label_soft_pred = torch.nn.Softmax(dim=1)(label_pred)

        cur_loss_np = loss.data.cpu().numpy()
        writer.add_scalar('step/total_loss', cur_loss_np, step)
        sum_loss += cur_loss_np
        
        #print('age acc')
        cur_label_acc = get_acc(label_pred, label_var)

        print( f"Epoch: {epoch}, id: {id}, loss: {cur_loss_np},\
                label acc: {cur_label_acc}")
        writer.add_scalar('step/label_acc', cur_label_acc, step)
 
        sum_label_acc += cur_label_acc
        step += 1
        count += 1

    print( f"training average label acc: { sum_label_acc / count}")
    writer.add_scalar('epoch/train_label_acc', sum_label_acc / count, epoch)
    
    print( f"training average loss: {sum_loss / count}")
    writer.add_scalar('epoch/train_sum_loss', sum_loss / count, epoch)

    print("Epoch val: " + str(epoch))
    sum_loss = 0.0

    label_true_lt, label_pred_lt, img_name_lt = [], [], []
    minet.eval()
    
    for _ in range(test_times):
        for id, (item, label, img_name) in enumerate(val_data_loader):
            item = item.cuda()
            label = label.cuda()
            input_var = torch.autograd.Variable(item, requires_grad=False)
            label_size = label.size()
            if label_size[0] == 1:
                label = label.view(-1)
                label_var = torch.autograd.Variable(label)
            else:
                label_var = torch.autograd.Variable(label.squeeze(dim=1))
            with torch.no_grad():
                label_pred = minet(input_var) 
        
            loss_label = nn_loss_label(label_pred, label_var)
            loss = loss_label
        
            label_soft_pred = torch.nn.Softmax(dim=1)(label_pred)
            cur_loss_np = loss.data.cpu().numpy()
            #writer.add_scalar('step/total_loss', cur_loss_np, step)
            sum_loss += cur_loss_np
        
            cur_label_acc = get_acc(label_pred, label_var)

            #print( f"Epoch val: {epoch}, id: {id}, loss: {cur_loss_np},\
            #    label acc: {cur_label_acc}")

            label_true_lt += list(label_var.data.cpu().numpy())
            label_pred_lt += list(label_soft_pred.data.cpu().numpy())
            img_name_lt += list(img_name)

    label_true_lt = np.array(label_true_lt)
    label_pred_lt = np.array(label_pred_lt)

    img_pred = {}
    img_ground_truth = {}
    for i in range(len(img_name_lt)):
        img_name = img_name_lt[i]
        if img_name not in img_pred:
            img_pred[img_name] = [label_pred_lt[i]]
        else:
            img_pred[img_name].append(label_pred_lt[i])
        if img_name in img_ground_truth:
            assert img_ground_truth[img_name] == label_true_lt[i]
        else:
            img_ground_truth[img_name] = label_true_lt[i]
    
    for img_name in img_pred:
        img_pred[img_name] = np.mean(np.array(img_pred[img_name]), axis=0)

    ground_truth_lt = []
    pred_lt = []
    for img_name in img_pred:
        pred_lt.append(img_pred[img_name])
        ground_truth_lt.append(img_ground_truth[img_name])
    
    label_true_lt = np.array(ground_truth_lt)
    label_pred_lt = np.array(pred_lt)

    label_true_lt = to_categorical(label_true_lt)
    label_auc = roc_auc_score(label_true_lt, label_pred_lt, average='macro')
    print( f"val average label auc: {label_auc}")
    writer.add_scalar('epoch/val_label_auc', label_auc, epoch)
    
    print( f"val sum loss: {sum_loss}")
    writer.add_scalar('epoch/val_sum_loss', sum_loss, epoch)
    
    if label_auc > best_val_auc:
        print(f'save best checkpoint: {epoch}')
        with open(f"best_checkpoint_{uid}_fold_{fold}.pt", 'wb') as f:
            checkpoint = {'minet': minet.state_dict(),
                    'epoch': epoch,
                    'best_val_auc': label_auc
                    }
            torch.save(checkpoint, f)
        best_val_auc = label_auc
        best_epoch = epoch
    else:
        if epoch - best_epoch > patience:
            print('patience end')
            break 

writer.close()
