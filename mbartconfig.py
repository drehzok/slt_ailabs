from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig
from pytorch_lightning.loggers import MLFlowLogger
from baseconfig import CFG


########### Model Stuffs ##############
CFG.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt") 
CFG.trf = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
CFG.trfconfig = MBartConfig()

CFG.vl_mlp_hidden = 2048
CFG.vl_tempconv_hidden = 1024
CFG.gloss_vocab_size = 2048

CFG.lr = 5e-5
CFG.trainer_configs = {
  "devices" : 1,
  "accelerator" : "gpu",
  "max_epochs" : 7,
  "logger" : MLFlowLogger(experiment_name = "s3d_translator_mbart")
}
