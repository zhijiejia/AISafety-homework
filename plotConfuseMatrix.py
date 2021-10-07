import os
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from net.CifarNet import CifarNet, split_weights
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools import scheduler, losses, ext_transforms, dataset, metric, utils


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Test_transform = transforms.Compose([
   transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

testDataset = dataset.CifarDataset(train=False, transform=Test_transform)
testLoader = DataLoader(testDataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True)

with torch.no_grad():

    model = CifarNet(num_classes=10).cuda()
    state_dict = torch.load('./weights/best_acc_0.pth')
    model.load_state_dict(state_dict)
    model.eval()
    testMetric = metric.StreamSegMetrics(n_classes=10)
    labelName = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for images, labels in tqdm(testLoader):

        images = images.to(torch.float32)
        labels = labels.to(torch.long)
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        outputs = torch.argmax(outputs, 1)
        testMetric.update(label_trues=labels.cpu().numpy(), label_preds=outputs.cpu().numpy())

    result = testMetric.get_results()
    print(f'Acc Class:', result['Acc'])

    plt.figure(figsize=(10,8))
    f, ax= plt.subplots(figsize = (10, 8))

    sns.set_theme(style="whitegrid", palette="pastel")
    fig = sns.heatmap(testMetric.confusion_matrix, annot=True, fmt='.20g', cmap='summer', ax=ax, linewidths=2, annot_kws={'size':30},cbar=False)   # cmap="binary",
    confuse_matrix = fig.get_figure()
    confuse_matrix.savefig('./imgs/confuse_matrix.png', dpi=400)
    print(testMetric.confusion_matrix)

