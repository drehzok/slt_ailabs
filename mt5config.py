########## Config stuffs #######

from glob import glob
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from baseconfig import CFG

  ############ Model Stuffs ############
CFG.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
CFG.trf = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")


  ############ Train Stuffs ############
CFG.lr = 5e-5
CFG.trainer_configs = {
    "devices" : 1,
    "accelerator" : "gpu",
    "max_epochs" : 7,
    "logger" : MLFlowLogger(experiment_name = "s3d_translator") ,
  }
