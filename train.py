from glob import glob
import pandas as pd
from model import Net
from data_loader import Train_WrittenDatset,Test_WrittenDatset
from torch.utils.data import DataLoader
import torch
from torch import nn
from numpy import zeros
import numpy as np

def train():
    losses=[]
    acces=[]

    for epoch in range(30):
        train_loss=0
        train_acc=0
        model.train()
        for img,label in train_dataloader:
            img=img.to(device)
            label=label.to(device)
            
            out=model(img)
            loss=criterion(out,label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss+=loss.item()
            
            pred = torch.argmax(out,dim = -1)
            pred1 = torch.argmax(label,dim = -1)

            num_correct=(pred==pred1).sum().item()
            acc=num_correct/img.shape[0]
            train_acc+=acc
        losses.append(train_loss/len(train_dataloader))
        acces.append(train_acc/len(train_dataloader))

        print('epoch:{},train loss:{:.4f},train acc:{:.4f}'.format(epoch,train_loss/len(train_dataloader),train_acc/len(train_dataloader)))


def test():
    model.load_state_dict(torch.load('model_params.pth'))
    model.eval()
    test_lable = []
    with torch.no_grad():
        for test_img in test_dataloader:
            test_img = test_img.to(device)

            pre_out = model(test_img)
            pre_label = torch.argmax(pre_out, -1)
            test_lable.append(pre_label)
    
    test_lable = torch.cat(test_lable, dim=0)  # 将预测结果拼接为一个大张量
    torch.set_printoptions(threshold=np.inf)



if __name__ == '__main__':
    root_dir = '../data'
    test_path = '../data/test_no_lable'
    dataset = Train_WrittenDatset(root_dir)
    test_dataset = Test_WrittenDatset(test_path)

    train_dataloader = DataLoader(dataset, batch_size=256)
    test_dataloader = DataLoader(test_dataset,batch_size=256,shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Building Model...")
    model = Net().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    print("training...")
    train()
    torch.save(model.state_dict(), 'model_params.pth')
    print("testing...")
    test()





















# losses=[]
# acces=[]

# for epoch in range(30):
#     train_loss=0
#     train_acc=0
#     model.train()
#     for img,label in train_loader:
#         img=img.to(device)
#         label=label.to(device)
        
#         out=model(img)
#         loss=criterion(out,label)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss+=loss.item()
        
#         _,pred=out.max(1)
#         l,pred1=label.max(1)
#         num_correct=(pred==pred1).sum().item()
#         acc=num_correct/img.shape[0]
#         train_acc+=acc
#     losses.append(train_loss/len(train_loader))
#     acces.append(train_acc/len(train_loader))

#     print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch,train_loss/len(train_loader),train_acc/len(train_loader)))
#     # torch.save(model.state_dict(), 'model_epoch{}.pth'.format(epoch))
# torch.save(model.state_dict(), 'model_params.pth')
# torch.save(self.model.state_dict(), path)
# # torch.save(model, 'model_params.pth')

# file_list = glob("/home/sdnu/djk/pythoncode/written_digits/data/test_no_label/*.txt")
# alldata = []
# for file_path in file_list:
#     with open(file_path, "r") as f:
#         returnVect = zeros((1, 1024))
#         for i in range(32):
#             lineStr = f.readline()
#             for j in range(32):
#                 returnVect[0, 32 * i + j] = int(lineStr[j])
#         alldata.append(returnVect)
# returntensor = torch.squeeze(torch.tensor(np.array(alldata)))
# returntensor = returntensor.to(torch.float32)
# print(returntensor.shape)

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test_imgs=returntensor.to(device)
# #计算结果
# model.eval()
# model.load_state_dict(torch.load('model_params.pth'))
# _,pre=model(test_imgs).max(1)
# #将结果转为提交kaggle的格式
# res={}
# pre = pre.cpu().numpy()
# pre_size=pre.shape[0]
# num = [i for i in range(1,pre_size+1)]
# res_df=pd.DataFrame({
#     'ImageId':num,
#     'Label':pre
# })

# #d导出为CSV文件
# res_df.to_csv('res.csv',index=False)