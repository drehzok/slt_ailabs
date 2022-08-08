from pytorch_lightning import Trainer

from data import PLDataModule
from model import PLAggregate
from train_configs import CFG

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__=="__main__":
  pl_datamodule = PLDataModule(CFG)
  pl_model = PLAggregate(CFG)

  pl_model.model.video2feature.s3d_load_weights(CFG.S3D_weights)
  
  trainer = Trainer(**CFG.trainer_configs)
  trainer.fit(model=pl_model, datamodule=pl_datamodule)
