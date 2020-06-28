import glob
import random
import os

stage = 'training'
sample_num = 60000
bag_size = 10

grid1_samples = [[] for _ in range(10)]
grid4_samples = [[] for _ in range(10)]
grid9_samples = [[] for _ in range(10)]
grid16_samples = [[] for _ in range(10)]
grid25_samples = [[] for _ in range(10)]

for i in range(10):
    grid1_samples[i] = glob.glob(f'/home/zhangr/workspace/HyperbolicAttention4Path/MNIST/syn/1grid/{stage}/{i}/*.png')
    grid4_samples[i] = glob.glob(f'/home/zhangr/workspace/HyperbolicAttention4Path/MNIST/syn/4grid/{stage}/{i}/*.png')
    grid9_samples[i] = glob.glob(f'/home/zhangr/workspace/HyperbolicAttention4Path/MNIST/syn/9grid/{stage}/{i}/*.png')
    grid16_samples[i] = glob.glob(f'/home/zhangr/workspace/HyperbolicAttention4Path/MNIST/syn/16grid/{stage}/{i}/*.png')
    grid25_samples[i] = glob.glob(f'/home/zhangr/workspace/HyperbolicAttention4Path/MNIST/syn/25grid/{stage}/{i}/*.png')

grid_samples = [[] for _ in range(10)]
for i in range(10):
    grid_samples[i] = grid1_samples[i] + grid4_samples[i] + grid9_samples[i] + grid16_samples[i] + grid25_samples[i]  


stage = 'val'
# bags contain 9
fout = open(f'syn/{stage}/bags_9.txt', 'w')
for _ in range(sample_num):
    img_lt = []
    img = random.choice(grid_samples[9])
    img_lt.append(img)
    for __ in range(bag_size - 1):
        index = random.randint(0,8)
        img = random.choice(grid_samples[index])
        img_lt.append(img)
    random.shuffle(img_lt)
    fout.write(' '.join(img_lt) + '\n')

fout.close()

# bags contain no 9
fout = open(f'syn/{stage}/bags_wo_9.txt', 'w')
for _ in range(sample_num):
    img_lt = []
    for __ in range(bag_size):
        index = random.randint(0,8)
        img = random.choice(grid_samples[index])
        img_lt.append(img)

    random.shuffle(img_lt)
    fout.write(' '.join(img_lt) + '\n')

fout.close()




