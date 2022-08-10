########## Config stuffs #######

from transformers import MT5Tokenizer, MT5ForConditionalGeneration, MT5Config
from pytorch_lightning.loggers import MLFlowLogger
from baseconfig import CFG

############ Model Stuffs ############
CFG.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
CFG.trf = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
CFG.trfconfig = MT5Config()
CFG.vl_mlp_hidden = 2048
CFG.vl_tempconv_hidden = 1024
CFG.gloss_vocab_size = 2048


############ Train Stuffs ############
CFG.lr = 5e-5
CFG.trainer_configs = {
  "devices" : 1,
  "accelerator" : "gpu",
  "max_epochs" : 7,
  "logger" : MLFlowLogger(experiment_name = "s3d_translator_mt5") ,
}
