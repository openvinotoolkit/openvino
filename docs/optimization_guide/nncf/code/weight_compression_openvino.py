#! [compression_8bit]
from nncf import compress_weights

...
model = compress_weights(model) # model is openvino.Model object
#! [compression_8bit]