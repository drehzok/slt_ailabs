import torch as T
import torch.nn as NN
import torch.nn.functional as F
import os
from components import *
from mt5config import CFG


class S3D(NN.Module):
  def __init__(self, num_class):
    super(S3D, self).__init__()
    self.base = NN.Sequential(
      SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
      NN.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
      BasicConv3d(64, 64, kernel_size=1, stride=1),
      SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
      NN.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
      Mixed_3b(),
      Mixed_3c(),
      NN.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
      Mixed_4b(),
      Mixed_4c(),
      Mixed_4d(),
      Mixed_4e(),
      Mixed_4f(),
      NN.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),
      Mixed_5b(),
      Mixed_5c(),
    )
    #self.fc = NN.Sequential(NN.Conv3d(1024, num_class, kernel_size=1, stride=1, bias=True),)
  
  def s3d_load_weights(self,weight_path):
    if os.path.isfile(weight_path):
      print("S3D weigth file loading")
      w_dict = T.load(weight_path)
      m_dict = self.state_dict()
      for name,param in w_dict.items():
        if 'module' in name:
          name = '.'.join(name.split('.')[1:])
        if name in m_dict:
          if param.size() == m_dict[name].size():
            m_dict[name].copy_(param)
          else:
            print("bad size:" + name, param.size(), m_dict[name].size())
        else:
          print("bad name:" + name)
      
      print("loading finished")
    else:
      print("bad weight path")

  def forward(self, x):
    y = self.base(x)
    return F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)

  def extract_features(self, x):
    y = self.base(x)
    return F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)

class Translator_TRF(NN.Module):
  """
  To-Do:
    1. Move transformer part to the config
    2. Hidden_size stuffs must be done through 
  """
  def __init__(self,CFG):
    super().__init__()
    self.transformer = CFG.trf

  def generate(self, embs):
    return self.transformer.generate(inputs_embeds = embs)

  def forward(self, embs, targets):
    return self.transformer(inputs_embeds = embs, labels = targets)

class TemporalConvBlock(NN.Module):
  def __init__(self, in_ch, hid_ch):
    super().__init__()

    self.conv1 = NN.Conv1d(in_ch, hid_ch, kernel_size = 3, stride = 1, padding = 1)
    #self.conv2 = NN.Conv1d(hid_ch, in_ch, kernel_size = 3, stride = 1, padding = 1)
    self.fc = NN.Linear(2048, 512)

  def forward(self, x):
    x = self.conv1(x)
    return F.relu(self.fc(T.permute(x,dims=(0,2,1))))

class Feature2Gloss(NN.Module):
  def __init__(self, gloss_vocab_size):
    """
    takes in T/4 x 843

    This part should be be configurable through CFG class
    """
    super().__init__()
    self.head_mlp = NN.Linear(1024,2048)
    self.mlp_bn = NN.BatchNorm1d(int(150/8))
    self.tcon_bn = NN.BatchNorm1d(int(150/8))

    self.temporal_conv = TemporalConvBlock(2048, 2048)

    self.glosser = NN.Linear(512, gloss_vocab_size)


  def feature_extract(self, x):
    x = T.permute(x,dims=(0,2,1))

    ##### HEAD #####
    x = self.head_mlp(x)
    x = F.relu(self.mlp_bn(x)) 

    x = self.temporal_conv(T.permute(x,dims=(0,2,1)))
    x = self.tcon_bn(x)
    # x at this point must have a shape T/4 x 512 (according to the paper)
    # but x actually has T/8 x 1024

    ###############
    return x
  
  def forward(self, x):

    x = self.feature_extract(x)

    return self.glosser(x)



class Aggregate(NN.Module):
  def __init__(self):
    super().__init__()
    self.video2feature = S3D(400)
    self.feature2gloss = Feature2Gloss(1000)
    self.gloss2text = Translator_TRF()

  def forward(self,x,target):
    x = self.video2feature.extract_features(x)
    x = x.squeeze(-1).squeeze(-1)
    x = self.feature2gloss.feature_extract(x)
    return self.gloss2text(x,target)
  
  def inference_gen(self,x):
    x = self.video2feature.extract_features(x)
    x = x.squeeze(-1).squeeze(-1)
    x = self.feature2gloss.feature_extract(x)
    
    return self.gloss2text.generate(x)
  
  def inf_no_gloss(self,x):
    x = self.video2feature.extract_features(x)
    x = x.squeeze(-1).squeeze(-1)
    x = self.feature2gloss.feature_extract(x)
    return x
 
