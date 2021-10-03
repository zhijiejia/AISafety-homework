import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from net.CifarNet import CifarNet, split_weights
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools import scheduler, losses, ext_transforms, dataset, metric, utils

epoch = 200
base_lr = 0.1
os.system('rm ./runs/cifar10/*')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
writer = SummaryWriter('runs/cifar10')
print(22222)
model = CifarNet(num_classes=10).cuda()
print(33333)
utils.setup_seed(2021)

Train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

Test_transform = transforms.Compose([
   transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainDataset = dataset.CifarDataset(train=True, transform=Train_transform)
trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True, num_workers=3, drop_last=True)
testDataset = dataset.CifarDataset(train=False, transform=Test_transform)
testLoader = DataLoader(testDataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True)

optimizer = torch.optim.SGD(
    params=split_weights(model),
    lr=base_lr,
    momentum=0.9,
    weight_decay=5e-4,
)

schedulerTrain = scheduler.PolyLR(optimizer, max_iters=epoch * len(trainLoader), power=0.9, warmUp=True)

# schedulerTrain = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)                           # T_max是周期的1/2

lossfun = nn.CrossEntropyLoss()

print(11111)

@torch.no_grad()
def evaluate(e):
    model.eval()
    testMetric = metric.StreamSegMetrics(n_classes=10)
    labelName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    testBar = tqdm(testLoader)
    testBar.set_description(f'Test Stage ')
    for images, labels in tqdm(testLoader):

        images = images.to(torch.float32)
        labels = labels.to(torch.long)
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        outputs = torch.argmax(outputs, 1)
        testMetric.update(label_trues=labels.cpu().numpy(), label_preds=outputs.cpu().numpy())

    result = testMetric.get_results()
    writer.add_scalar(tag='Acc', scalar_value=result['Acc'], global_step=e)
    print(f'Acc Class:', result['Acc'])

for e in range(epoch):
    trainBar = tqdm(trainLoader)
    iters = 0
    AvgLoss = 0
    trainBar.set_description(f'[ Epoch: {e} / {epoch} ]')
    writer.add_scalar(tag='Learning_Rate/Backbone', scalar_value=schedulerTrain.show_lr()[0], global_step=e)
    writer.add_scalar(tag='Learning_Rate/Fc', scalar_value=schedulerTrain.show_lr()[1], global_step=e)
    for images, labels in trainBar:

        images = images.to(torch.float32)
        labels = labels.to(torch.long)
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)

        loss = lossfun(outputs, labels)

        iters += 1
        AvgLoss += loss.item()
        trainBar.set_postfix(loss=AvgLoss / iters)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedulerTrain.step()

    writer.add_scalar(tag='Train Avg Loss', scalar_value=AvgLoss / iters, global_step=e)

    evaluate(e)
