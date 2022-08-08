from pytorch_lightning import LightningModule
from torchmetrics.functional import word_error_rate as wer
from torch import optim as O

from modules import *


class PLAggregate(LightningModule):
  def __init__(self, CFG = None):
    super().__init__()
    self.model = Aggregate()
    self.CFG = CFG

  def forward(self,x):
    return self.model.inference_gen(x)
  
  def configure_optimizers(self):
    optimizer = O.AdamW(self.parameters(),lr = self.CFG.lr)
    return optimizer
  
  def training_step(self, train_batch, batch_idx):
    x,y = train_batch
    loss = self.model(x,y).loss
    #print("train loss",loss)
    self.log("train_loss", loss)

    return loss
  
  def validation_step(self, val_batch, batch_idx):
    x,y = val_batch
    #print("Validation x shape:",x.shape)
    gloss = self.model.inf_no_gloss(x)
    loss = self.model.gloss2text(gloss, y).loss
    preds = self.model.gloss2text.generate(gloss)


    #print("validation loss",loss)

    self.log("val_loss",loss)
    pred_text = self.CFG.tokenizer.batch_decode(preds, skip_special_tokens=True)
    y_text = self.CFG.tokenizer.batch_decode(y, skip_special_tokens=True)

    self.log("WER", wer(pred_text, y_text))

