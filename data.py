from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from pytorch_lightning import LightningDataModule

class CustomDataset(Dataset):
  def __init__(self, CFG, mode="train"):
    super().__init__()
    self.df = getattr(CFG, mode).df
    self.mode = mode
    self.CFG = CFG
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    x,y = self.CFG.inoutput_pairgen(self.CFG, self.df, idx, mode=self.mode)
    x = self.CFG.frames_path_to_input_format(self.CFG, x)
    x = pad(
      x,
      (0,0,0,0,0,self.CFG.maximum_length_vid - x.shape[1]),
      "constant",
      0
    )
    y = self.CFG.tokenizer(
      y,
      return_tensors="pt",
      padding="max_length",
      max_length=self.CFG.maximum_length_seq,
    ).input_ids

    y = pad(
      y, 
      (0, self.CFG.maximum_length_seq - y.shape[-1]),
      "constant",
      0
    )

    return x, y.view(-1)


class PLDataModule(LightningDataModule):
  def __init__(self, CFG):
    super().__init__()
    self.train_ds = CustomDataset(CFG, "train")
    self.valid_ds = CustomDataset(CFG, "valid")
    self.test_ds  = CustomDataset(CFG, "test")
    self.batch_size = CFG.batch_size
    self.num_workers = CFG.num_workers
    
  def train_dataloader(self):
    return DataLoader(
      dataset = self.train_ds,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
    )

  def val_dataloader(self):
    return DataLoader(
      dataset = self.valid_ds,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
    )

  def test_dataloader(self):
    return DataLoader(
      dataset = self.test_ds,
      batch_size = self.batch_size,
      num_workers = self.num_workers,
    )
 
