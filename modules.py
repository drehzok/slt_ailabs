import torch as T
import torch.nn as NN
import torch.nn.functional as F
import os
from components import *
from mt5config import CFG
from torch_geometric_temporal.nn.recurrent import A3TGCN2


class S3D(NN.Module):
  """
  Taken from kylemin/S3D github
  """
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
  Assumption - CFG.trf is a transformer architecture from huggingface
  """
  def __init__(self,CFG):
    super().__init__()
    self.transformer = CFG.trf
    self.maxlen = CFG.maximum_length_seq

  def generate(self, embs):
    return self.transformer.generate(inputs_embeds = embs, max_new_tokens=self.maxlen)

  def forward(self, embs, targets):
    return self.transformer(inputs_embeds = embs, labels = targets)

class TemporalConvBlock(NN.Module):
  """
  Currently - Conv1d but maybe it is worth trying Conv3D with (kernel_size, 1, 1) filter
  """
  def __init__(self, CFG):
    super().__init__()

    self.conv1 = NN.Conv1d(CFG.vl_mlp_hidden, CFG.vl_tempconv_hidden, kernel_size = 3, stride = 1, padding = 1)
    self.conv2 = NN.Conv1d(CFG.vl_tempconv_hidden, CFG.vl_mlp_hidden, kernel_size = 3, stride = 1, padding = 1)
    self.fc = NN.Linear(CFG.vl_mlp_hidden, CFG.trfconfig.d_model)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    return F.relu(self.fc(T.permute(x,dims=(0,2,1))))

class Feature2Gloss(NN.Module):
  def __init__(self, CFG):
    """
    takes in T/4 x 843

    This part should be be configurable through CFG class
    """
    super().__init__()
    self.head_mlp = NN.Linear(1024,CFG.vl_mlp_hidden)
    self.mlp_bn = NN.BatchNorm1d(int(CFG.maximum_length_vid/8))
    self.tcon_bn = NN.BatchNorm1d(int(CFG.maximum_length_vid/8))

    self.temporal_conv = TemporalConvBlock(CFG)

    self.glosser = NN.Linear(CFG.trfconfig.d_model, CFG.gloss_vocab_size)


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

class TGCN(NN.Module):
  def __init__(self, CFG):
    self.window = CFG.tgcn.window
    self.tgcn = A3TGCN2(in_channels=3, out_channels=3, periods=self.window//2, batch_size=1)
    self.lin = NN.Linear(399, CFG.trfconfig.d_model)
    self.skeleton = CFG.tgcn.skeleton

  def forward(self, x):
    per_inc = window//2
    output = None
    i = 0
    while i<data.shape[-1]//per_inc:
      tempin = data[..., i*per_inc:i*per_inc+window-1]
      temp = self.tgcn(tempin, self.skeleton).unsqueeze(-1)
      if output is None:
        output = temp
      else:
        output = T.cat((output, temp), dim=-1)
      i += 1
    shapes = output.shape
    output = T.permute(output, dims=(0,3,1,2)).view(shapes[0], shapes[1], -1)
    return self.lin(output)



class Aggregate(NN.Module):
  def __init__(self,CFG):
    super().__init__()
    self.video2feature = S3D(400)
    self.feature2gloss = Feature2Gloss(CFG)
    self.gloss2text = Translator_TRF(CFG)

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
 
class TGCNAggregate(NN.Module):
  def __init__(self, CFG):
    super().__init__()
    self.graph2gloss = TGCN(CFG)
    self.gloss2text = Translator_TRF(CFG)

  def forward(self, x, target):
    x = self.graph2gloss = TGCN(x)
    return self.gloss2text(x, target)

  def inference_gen(self, x):
    return self.gloss2text.generate(self.graph2gloss(TGCN(x)))
 
