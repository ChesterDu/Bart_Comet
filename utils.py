import json
class Pretrain_Setting():
  def __init__(self,config_pth):
    with open(config_pth,'r') as fin:
        config = json.load(fin)

    self.data_name = config["data_name"]
    self.saved_data_pth = None if config["saved_data_pth"] == "None" else config["saved_data_pth"]
    self.raw_data_pth = None if config["raw_data_pth"] == "None" else config["raw_data_pth"]
    self.processed_data_pth = None if config["processed_data_pth"] == "None" else config["processed_data_pth"]
    self.corruption = config["corruption"]
    self.ratio = float(config["ratio"])
    self.num_of_span = int(config["num_of_span"])
    self.lamda = int(config["lamda"])
    self.epoch_num = int(config["epoch_num"])
    self.batch_size  = int(config["batch_size"])
    self.num_accumulation = int(config["num_accumulation"])
    self.log_pth = config["log_pth"]
    self.best_model_pth  = config["best_model_pth"]

class Finetune_Setting():
  def __init__(self,config_pth):
    with open(config_pth,"r") as fin:
        config = json.load(fin)

    self.saved_data_pth = None if config["saved_data_pth"] == "None" else config["saved_data_pth"]
    self.raw_data_pth = None if config["raw_data_pth"] == "None" else config["raw_data_pth"]
    self.processed_data_pth = None if config["processed_data_pth"] == "None" else config["processed_data_pth"]
    self.epoch_num = int(config["epoch_num"])
    self.batch_size  = int(config["batch_size"])
    self.num_accumulation = int(config["num_accumulation"])
    self.log_pth = config["log_pth"]
    self.best_model_pth  = config["best_model_pth"]
    self.load_model_pth = config["load_model_pth"]
    self.gen_pth = config["gen_pth"]



def load_config(config_pth, pretrain = True):
    if pretrain:
      setting = Pretrain_Setting(config_pth)

    else:
      setting = Finetune_Setting(config_pth)

    return setting