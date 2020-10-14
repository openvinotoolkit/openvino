import torch
from collections import namedtuple

from ..hooks import OpenVINOTensor, forward_hook


def inference(model, func, anchors, pred_logits, pred_anchor_deltas, image_sizes):
    # Convert from lists of OpenVINOTensor to torch.tensor and perform origin run
    pred_logits_t = [v.tensor() for v in pred_logits]
    pred_anchor_deltas_t = [v.tensor() for v in pred_anchor_deltas]
    output = func(anchors, pred_logits_t, pred_anchor_deltas_t, image_sizes)

    # Concatenate the inputs (should be tracked)
    logist = torch.cat(pred_logits, dim=1).view(1, -1).sigmoid()
    deltas = torch.cat(pred_anchor_deltas, dim=1).view(1, -1)
    assert(isinstance(logist, OpenVINOTensor))
    assert(isinstance(deltas, OpenVINOTensor))

    # Create an alias
    class DetectionOutput(torch.nn.Module):
        def __init__(self, anchors):
            super().__init__()
            self.anchors = anchors
            self.variance_encoded_in_target = True
            self.nms_threshold = model.nms_threshold
            self.confidence_threshold = model.score_threshold
            self.top_k = model.topk_candidates * len(model.in_features)
            self.keep_top_k = self.top_k
            self.code_type = 'caffe.PriorBoxParameter.CENTER_SIZE'

        def state_dict(self):
            return {'anchors': anchors}

    outputs = [OpenVINOTensor(output[0].pred_boxes.tensor),
               OpenVINOTensor(output[0].scores),
               OpenVINOTensor(output[0].pred_classes)]
    for out in outputs:
        out.graph = pred_logits[0].graph

    # Concatenate anchors
    anchors = torch.cat([a.tensor for a in anchors]).view(1, 1, -1)

    forward_hook(DetectionOutput(anchors), (deltas, logist), outputs[1])
    return output


def forward(model, forward, batched_inputs):
    output = forward([{'image': batched_inputs}])
    output = OpenVINOTensor(output[0]['instances'].scores)
    output.node_name = 'DetectionOutput_'
    return output


def preprocess_image(model, forward, inp):
    out = namedtuple('ImageList', ['tensor', 'image_sizes'])
    out.tensor = (inp[0]['image'] - model.pixel_mean) / model.pixel_std
    out.image_sizes = [out.tensor.shape[-2:]]
    return out


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py
class RetinaNet(object):
    def __init__(self):
        self.class_name = 'detectron2.modeling.meta_arch.retinanet.RetinaNet'

    def hook(self, new_func, model, old_func):
        return lambda *args: new_func(model, old_func, *args)

    def register_hook(self, model):
        model.inference = self.hook(inference, model, model.inference)
        model.forward = self.hook(forward, model, model.forward)
        model.preprocess_image = self.hook(preprocess_image, model, model.preprocess_image)
