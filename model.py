from transformers import BartForConditionalGeneration
import torch


def make_model(model_pth = None):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    if model_pth != None:
        print("model loaded from {}".format(model_pth))
        model.load_state_dict(torch.load(model_pth))
    return model