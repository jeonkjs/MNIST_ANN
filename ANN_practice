root_dir = "root_dir"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch.optim as optim

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class MNIST(nn.Module):
  def __init__(self, config):
    super(MNIST, self).__init__()

    self.width = config["input_width_size"]
    self.height = config["input_height_size"]

    self.feature_size = config["feature_size"]

    self.num_labels = config["num_labels"]

    self.activation = nn.Sigmoid()

    # Neural Network
    
    self.layer_1 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)
    self.layer_2 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)
    self.layer_3 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_4 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_5 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_6 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_7 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_8 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_9 = nn.Linear(in_features=self.width*self.height, out_features = self.width*self.height)  
    self.layer_10 = nn.Linear(in_features=self.width*self.height, out_features = self.num_labels)  
  
  def forward(self, input_features, labels=None):
  
    layer_1_output = self.layer_1(input_features)
    activated_output_1 = self.activation(layer_1_output)
    
    layer_2_output = self.layer_2(activated_output_1)
    activated_output_2 = self.activation(layer_2_output)
    
    layer_3_output = self.layer_3(activated_output_2)
    activated_output_3 = self.activation(layer_3_output)
    
    layer_4_output = self.layer_4(activated_output_3)
    activated_output_4 = self.activation(layer_4_output)
    
    layer_5_output = self.layer_5(activated_output_4)
    activated_output_5 = self.activation(layer_5_output)
    
    layer_6_output = self.layer_6(activated_output_5)
    activated_output_6 = self.activation(layer_6_output)
    
    layer_7_output = self.layer_7(activated_output_6)
    activated_output_7 = self.activation(layer_7_output)
    
    layer_8_output = self.layer_8(activated_output_7)
    activated_output_8 = self.activation(layer_8_output)
    
    layer_9_output = self.layer_9(activated_output_8)
    activated_output_9 = self.activation(layer_9_output)
    
    layer_10_output = self.layer_10(activated_output_9)
    activated_output_10 = self.activation(layer_10_output)

    # 학습 시
    if labels is not None:
      loss_fnc = nn.CrossEntropyLoss()
      logits = activated_output_10
      loss = loss_fnc(logits, labels)
      return loss
    # 평가 시 가장 확률이 높은 것 선택
    else:
      output = torch.argmax(activated_output_10, -1)
      return output


from keras.datasets import mnist
def load_dataset():
  (train_X, train_y), (test_X, test_y) = mnist.load_data()
  print(train_X.shape) #(60000, 28, 28)
  print(train_y.shape)
  print(test_X.shape) #(10000, 28, 28)
  print(test_y.shape)
  # (60000, 784)
  train_X = train_X.reshape(-1, 28*28)
  test_X  = test_X.reshape(-1, 28*28)

  #tensor로 변경
  train_X = torch.tensor(train_X, dtype=torch.float)
  train_y = torch.tensor(train_y, dtype=torch.long)
  test_X = torch.tensor(test_X, dtype=torch.float)
  test_y = torch.tensor(test_y, dtype=torch.long)
  return (train_X, train_y), (test_X, test_y)
tmp = load_dataset()

#tensor을 리스트로 변경
def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()
    
#평가모드  
def do_test(model, test_dataloader):
  #eval로 변경
  model.eval()
  predicts, answers = [], []
  for step, batch in enumerate(test_dataloader):
    # .cuda()를 통해 메모리에 업로드
    batch = tuple(t.cuda() for t in batch)

    input_features, labels = batch
    output = model(input_features)

    #아웃풋과 라벨이 텐서 형태로 되어 있기 때문에 리스트 형태로 변경
    predicts.extend(tensor2list(output))
    answers.extend(tensor2list(labels))
    
  print("Accuracy : {}".format(accuracy_score(answers, predicts)))
 
 #학습모드
 def train(config):
  model = MNIST(config).cuda()
  
  (train_X, train_y), (test_X, test_y) = load_dataset()
  
  train_features = TensorDataset(train_X, train_y)
  train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])

  test_features = TensorDataset(test_X, test_y)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])

  #아담 옵티마이저 사용
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  
  #train으로 변경
  model.train()
  for epoch in range(config["epoch"]):
    # epoch 마다 평균 loss를 출력하기위한 loss 저장 리스트
    losses = []

    for step, batch in enumerate(train_dataloader):
      # .cuda()를 통해 메모리에 업로드
      batch = tuple(t.cuda() for t in batch)

      # 각 feature 저장
      input_features, labels = batch
      
      # 모델 호출을 통해 loss return
      loss = model(input_features, labels)

      # 역전파 변화도 초기화
      # ==> .backward() 호출 시, 변화도 버퍼에 데이터가 계속 누적
      #     이를 초기화 시켜주는 단계
      optimizer.zero_grad()

      # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
      loss.backward()

      # 모델 내부 각 매개변수 가중치 갱신
      optimizer.step()

      # 1000번째 스텝마다, 스텝과 출력값을 출력. loss값은 loss.data.item()으로 뽑아온다. loss값은 텐서 형태이기 때문에 이렇게 출력해야 한다.
      if (step+1) % 1000 == 0:
        print("{} step processed.. current loss : {}".format(step+1, loss.data.item()))
      losses.append(loss.data.item())
    
    # 한 에폭이 끝날때마다 평균 로스 출력
    print("Average Loss : {}".format(np.mean(losses)))
    # epoch이 끝날 때 마다, 모델 저장
    torch.save(model.state_dict(), os.path.join(config["output_dir_path"], "epoch_{}.pt".format(epoch + 1)))

    # 지금까지 학습한 가중치로 평가 진행
    do_test(model, test_dataloader)
    

def test(config):
  model = MNIST(config).cuda()

  # 저장된 모델 가중치 Load
  model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["trained_model_name"])))

  # 데이터 load
  (_, _), (test_X, test_y) = load_dataset()
  
  test_features = TensorDataset(test_X, test_y)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])
  
  do_test(model, test_dataloader)
    
import os
if(__name__=="__main__"):
    # 학습된 모델을 저장하는 경로
    output_dir = os.path.join(root_dir, "10")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모드 변경 중요
    config = {"mode": "test",
              #매 에폭마다 모델을 저장.
              "trained_model_name":"epoch_{}.pt".format(10),
              "output_dir_path":output_dir,
              "input_width_size":28,
              "input_height_size":28,
              "feature_size": 512,
              "num_labels": 10,
              "batch_size":32,
              "epoch":10,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)
