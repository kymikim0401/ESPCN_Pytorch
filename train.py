import os
import time
import matplotlib.pyplot as plt

import torch
from torch import nn 
from torch import optim as optim
from torch.utils.data import DataLoader

from ESPCN_model import ESPCN
from dataset import ESPCNTrainDataset, ESPCNValDataset

def psnr(ground_truth, output):
    output = output.detach().cpu()
    ground_truth = ground_truth.detach().cpu()
    return 10. * torch.log10(1. / torch.mean((ground_truth - output) ** 2))

def train(model, data_loader, loss_criteria, optimizer):
    model.train()
    running_train_loss = 0
    running_train_psnr = 0
    for batch, data in enumerate(data_loader):
        input_img = data['lr']
        target = data['truth'].cuda()
        optimizer.zero_grad()
        output = model.forward(input_img.cuda())
        loss = loss_criteria(output, target)
        loss.backward()
        optimizer.step()   

        running_train_loss += loss.item()
        running_train_psnr += psnr(target, output)

    final_loss = running_train_loss / len(data_loader.dataset)
    final_psnr = running_train_psnr / int(len(data_loader.dataset)/data_loader.batch_size)
    return final_loss, final_psnr

def validate(model, data_loader, loss_criteria):
    model.eval()
    running_val_loss = 0
    running_val_psnr = 0
    with torch.no_grad():
        for batch, data in enumerate(data_loader):
            input_img = data['lr']
            target = data['truth'].cuda()

            output = model.forward(input_img.cuda())
            loss = loss_criteria(output, target)

            running_val_loss += loss.item()
            running_val_psnr += psnr(target, output)
    
    final_loss = running_val_loss / len(data_loader.dataset)
    final_psnr = running_val_psnr / int(len(data_loader.dataset)/data_loader.batch_size)
    return final_loss, final_psnr

if __name__ == '__main__':

    Epoch = 20
    Batch_size = 10
    train_data = ESPCNTrainDataset('dataset_img', upscale=3)
    train_loader = DataLoader(dataset=train_data, batch_size=Batch_size)
    val_data = ESPCNValDataset('dataset_img', upscale=3)
    val_loader = DataLoader(dataset=val_data, batch_size=Batch_size)

    model = ESPCN(3).cuda()
    loss_criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()
    for epoch in range(Epoch):
        print('===> Epoch {} of {}'.format(epoch+1, Epoch))
        train_epoch_loss, train_epoch_psnr = train(model, train_loader, loss_criteria, optimizer)
        val_epoch_loss, val_epoch_psnr = validate(model, val_loader, loss_criteria)

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        end = time.time()
        print('Train loss: {:.3f}, Val loss: {:.3f}'.format(train_epoch_loss, val_epoch_loss))
        print('Train psnr: {:.3f}, Val psnr: {:.3f}, time: {:.2f}'.format(train_epoch_psnr, val_epoch_psnr, end-start))
    torch.save(model.state_dict(), 'checkpoints/model_relu1.pth')
    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# psnr plots
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='train PSNR dB')
plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()   