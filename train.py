from pytorch_lightning import Trainer

from data import PLDataModule
from model import PLAggregate
from train_configs import CFG


if __name__=="__main__":
  pl_datamodule = PLDataModule(CFG)
  pl_model = PLAggregate()

  pl_model.model.video2feature.s3d_load_weights(CFG.S3D_weight_path)
  
  trainer = Trainer(**CFG.trainer_configs)
  trainer.fit(model=pl_model, datamodule=pl_datamodule)
