import time
import torch

class BenchHook:
    def __init__(self):
        self.tm_list = []

    def clearTMList(self):
        self.tm_list.clear()

    def getTMlist(self):
        return self.tm_list

    def newforward(self, model, model_type):
        org_forward = model.forward
        if model_type == "decoder" or model_type == "codegen2" or model_type == "mpt" or model_type == "replit" or model_type == "chatglm" or model_type == "falcon":
            def myforward(input_ids: torch.LongTensor, attention_mask = None, past_key_values = None, **kwargs):
                beg = time.time()
                ret = org_forward(input_ids, attention_mask, past_key_values, **kwargs)
                end = time.time()
                self.tm_list.append(end-beg)
                # print("TM=", end-beg)
                return ret
            model.forward = myforward
        elif model_type == "t5" or model_type == "blenderbot" or model_type == "codet5":
            def myforward(input_ids = None, attention_mask = None, decoder_input_ids = None, encoder_outputs = None, past_key_values = None, **kwargs):
                beg = time.time()
                ret = org_forward(input_ids, attention_mask, decoder_input_ids, encoder_outputs, past_key_values, **kwargs)
                end = time.time()
                self.tm_list.append(end-beg)
                # print("TM=", end-beg)
                return ret
            model.forward = myforward
        else:
            print("model_type:{}, does not support overloaded model forward".format(model_type))