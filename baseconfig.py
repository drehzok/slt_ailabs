########## Config stuffs #######

import os
import pandas as pd
import numpy as np
import cv2
import torch as T

from glob import glob
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, MT5Config
from pytorch_lightning.loggers import MLFlowLogger

dataroot = "/home/ailabs/Desktop/sungbae/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"

class CFG:
  ###### Data stuffs #######
  maximum_length_vid = 150 #should be 550
  maximum_length_seq = 50  #should be 120
  batch_size = 2
  num_workers = 4
  class train:
    df = pd.read_csv(
      os.path.join(
        dataroot,
        "annotations/manual/PHOENIX-2014-T.train.corpus.csv"
      ),
      delimiter = "|"
    )
    path = os.path.join(dataroot,"features/fullFrame-210x260px/train")
  class valid:
    df = pd.read_csv(
      os.path.join(
        dataroot,
        "annotations/manual/PHOENIX-2014-T.dev.corpus.csv"
      ),
      delimiter = "|"
    )
    path = os.path.join(dataroot,"features/fullFrame-210x260px/dev")

  class test:
    df = pd.read_csv(
      os.path.join(
        dataroot,
        "annotations/manual/PHOENIX-2014-T.test.corpus.csv"
      ),
      delimiter = "|"
    )
    path = os.path.join(dataroot,"features/fullFrame-210x260px/test")

  def inoutput_pairgen(self, df: pd.DataFrame, id : int, mode = "train"):
    temp = df.iloc[id]
    vidtemp = os.path.split(os.path.split(temp.video)[0])[0]
    frame_list = glob(os.path.join(getattr(self,mode).path,vidtemp,"*.png"))
    return (frame_list, temp.translation)
  
  def frames_path_to_input_format(self, frame_list, mode = 'train'):
    """
    From the list of frames, generate appropriate input data format
    """
    # start with datadir prefix
    prefix = getattr(self,mode).path
    frame_list.sort()
    snippet = list()

    for f in frame_list:
      img = cv2.imread(os.path.join(prefix,f))
      img = img[...,::-1]
      snippet.append(img)
    
    snippet = np.concatenate(snippet, axis=-1)
    snippet = T.from_numpy(snippet).permute(2,0,1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)

  ############ Model Stuffs ############
  tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
  trf = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
  trfconfig = MT5Config()
  S3D_weights = "/home/ailabs/Desktop/sungbae/S3D_kinetics400.pt"
  vl_mlp_hidden = 2048
  vl_tempconv_hidden = 1024
  gloss_vocab_size = 2048


  ############ Train Stuffs ############
  lr = 5e-5
  trainer_configs = {
    "devices" : 1,
    "accelerator" : "gpu",
    "max_epochs" : 7,
    "logger" : MLFlowLogger(experiment_name = "s3d_translator") ,
  }
