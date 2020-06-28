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
import tqdm
import uuid
uid = uuid.uuid1()

usage = "usage: python main.py "
parser = OptionParser(usage)

parser.add_option("-l", "--learning_rate", dest="learning_rate", type="float", default=0.0001,
                    help="set learning rate for optimizor")
parser.add_option("-m", "--mu", dest="mu", type="float", default=1.0,
                    help="set mu")
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=512,
                    help="batch size")
parser.add_option("-o", "--output", dest="output", type="string", default="specified_format.csv",
                    help="output file")
parser.add_option("-r", "--resume", dest="model_file", type="string", default="",
                    help="resume the file from a model file")

(options, args) = parser.parse_args()

batch_size = options.batch_size
mu = options.mu
embed_len = 10
epoch_num = 2000
inst_num = 10
patience = 5
best_epoch = 0
best_val_auc = 0.0

minet = MINet(embed_len, inst_num)
minet = torch.nn.DataParallel(minet).cuda()

learning_rate = options.learning_rate

optimizer = torch.optim.Adam([
            {'params': minet.parameters()}
        ],
        lr=learning_rate)

nn_loss_c9 = torch.nn.CrossEntropyLoss().cuda()

loss = 0

train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
            ])

val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
            ])

train_image_data = SlideDataset.SlideDataset('../../syn/training/bags_wo_9.txt', '../../syn/training/bags_9.txt', train_transform)
train_data_loader = torch.utils.data.DataLoader(train_image_data, shuffle=True, num_workers=10, batch_size=batch_size)

val_image_data = SlideDataset.SlideDataset('../../syn/testing/bags_wo_9.txt', '../../syn/testing/bags_9.txt', val_transform)
val_data_loader = torch.utils.data.DataLoader(val_image_data, shuffle=False, num_workers=10, batch_size=batch_size)

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
    sum_c9_acc = 0.0
    sum_loss = 0.0
    count = 0

    gender_true_lt = []
    gender_pred_lt = []

    minet.train()

    #bar = tqdm.tqdm(enumerate(train_data_loader)) 
    bar = tqdm.tqdm(train_data_loader)

    for (item, c9, img_name) in bar:
        item = item.cuda()
        c9 = c9.cuda()
        input_var = torch.autograd.Variable(item, requires_grad=True)
        c9_var = torch.autograd.Variable(c9.squeeze(dim=1))

        optimizer.zero_grad()

        c9_pred, out_weight = minet(input_var) 
        
        loss_c9 = nn_loss_c9(c9_pred, c9_var)
        loss = loss_c9 
        
        loss.backward()
        
        if torch.isnan(input_var).any():
            print('nan in input')

        flag = False
        for para in minet.parameters():
            if torch.isnan(para.grad).any():
                flag = True
                print('find nan in gradient')
            if torch.isnan(para).any():
                print('nan in para')

        if flag == True:
            print('begin update some nan')
            index = torch.isnan(para.grad)
            para.grad[index] = 0.0
        else:
            optimizer.step()
        
        c9_soft_pred = torch.nn.Softmax(dim=1)(c9_pred)

        cur_loss_np = loss.data.cpu().numpy()
        writer.add_scalar('step/total_loss', cur_loss_np, step)
        sum_loss += cur_loss_np
        
        #print('age acc')
        cur_c9_acc = get_acc(c9_pred, c9_var)

        #print( f"Epoch: {epoch}, id: {id}, loss: {cur_loss_np},\
        #        c9 acc: {cur_c9_acc}")
        bar.set_description(desc=f'acc: {cur_c9_acc:.3f}', refresh=True)
        writer.add_scalar('step/c9_acc', cur_c9_acc, step)
 
        sum_c9_acc += cur_c9_acc
        step += 1
        count += 1

    print( f"training average c9 acc: { sum_c9_acc / count}")
    writer.add_scalar('epoch/train_c9_acc', sum_c9_acc / count, epoch)
    
    print( f"training average loss: {sum_loss / count}")
    writer.add_scalar('epoch/train_sum_loss', sum_loss / count, epoch)

    print("Epoch val: " + str(epoch))
    sum_loss = 0.0

    c9_true_lt, c9_pred_lt, img_name_lt = [], [], []
    minet.eval()
    
    for id, (item, c9, img_name) in enumerate(val_data_loader):
        item = item.cuda()
        c9 = c9.cuda()
        input_var = torch.autograd.Variable(item, requires_grad=False)
        
        c9_var = torch.autograd.Variable(c9.squeeze(dim=1))
        with torch.no_grad():
            c9_pred, out_weight = minet(input_var) 
    
        loss_c9 = nn_loss_c9(c9_pred, c9_var)
        loss = loss_c9
    
        c9_soft_pred = torch.nn.Softmax(dim=1)(c9_pred)
        cur_loss_np = loss.data.cpu().numpy()
        #writer.add_scalar('step/total_loss', cur_loss_np, step)
        sum_loss += cur_loss_np
    
        cur_c9_acc = get_acc(c9_pred, c9_var)

        #print( f"Epoch val: {epoch}, id: {id}, loss: {cur_loss_np},\
        #    c9 acc: {cur_c9_acc}")

        c9_true_lt += list(c9_var.data.cpu().numpy())
        c9_pred_lt += list(c9_soft_pred.data.cpu().numpy())
        img_name_lt += list(img_name)

    c9_true_lt = np.array(c9_true_lt)
    c9_pred_lt = np.array(c9_pred_lt)

    img_pred = {}
    img_ground_truth = {}
    for i in range(len(img_name_lt)):
        img_name = img_name_lt[i]
        if img_name not in img_pred:
            img_pred[img_name] = [c9_pred_lt[i]]
        else:
            img_pred[img_name].append(c9_pred_lt[i])
        if img_name in img_ground_truth:
            assert img_ground_truth[img_name] == c9_true_lt[i]
        else:
            img_ground_truth[img_name] = c9_true_lt[i]
    
    for img_name in img_pred:
        img_pred[img_name] = np.mean(np.array(img_pred[img_name]), axis=0)

    ground_truth_lt = []
    pred_lt = []
    for img_name in img_pred:
        pred_lt.append(img_pred[img_name])
        ground_truth_lt.append(img_ground_truth[img_name])
    
    c9_true_lt = np.array(ground_truth_lt)
    c9_pred_lt = np.array(pred_lt)

    c9_true_lt = to_categorical(c9_true_lt)
    c9_auc = roc_auc_score(c9_true_lt, c9_pred_lt, average='macro')
    print( f"val average c9 auc: {c9_auc}")
    writer.add_scalar('epoch/val_c9_auc', c9_auc, epoch)
    
    print( f"val sum loss: {sum_loss}")
    writer.add_scalar('epoch/val_sum_loss', sum_loss, epoch)
    
    if c9_auc > best_val_auc:
        print(f'save best checkpoint: {epoch}')
        with open(f"best_checkpoint_{uid}.pt", 'wb') as f:
            checkpoint = {'minet': minet.state_dict(),
                    'epoch': epoch,
                    'best_val_auc': c9_auc
                    }
            torch.save(checkpoint, f)
        best_val_auc = c9_auc
        best_epoch = epoch
    else:
        if epoch - best_epoch > patience:
            print('patience end')
            break 

writer.close()
