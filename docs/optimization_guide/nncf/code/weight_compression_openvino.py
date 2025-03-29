#! [compression_8bit]
from nncf import compress_weights

...
model = compress_weights(model) # model is openvino.Model object
#! [compression_8bit]

#! [compression_4bit]
from nncf import compress_weights, CompressWeightsMode

...
model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.8) # model is openvino.Model object
#! [compression_4bit]