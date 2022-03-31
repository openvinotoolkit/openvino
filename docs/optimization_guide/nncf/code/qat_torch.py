# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [imports]
import torch
import nncf  # Important - should be imported right after torch
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
#! [imports]

#! [nncf_congig]
nncf_config_dict = {
    "input_info": {"sample_size": [1, 3, 224, 224]}, # image size
    "compression": {
        "algorithm": "quantization",  # 8-bit quantization with default settings
    },
}
nncf_config = NNCFConfig.from_dict(nncf_config_dict)
nncf_config = register_default_init_args(nncf_config, data_loader) # data_loader is an instance of torch.utils.data.DataLoader
#! [nncf_congig]

#! [wrap_model]
model = TorchModel() # instance of torch.nn.Module
compression_ctrl, model = create_compressed_model(model, nncf_config)
#! [wrap_model]

#! [distributed]
compression_ctrl.distributed() # call it before the training loop
#! [distributed]

#! [tune_model]
...
# tune quantized model for 5 epochs as the baseline
for epoch in range(0, 5):
    train(train_loader, model, criterion, optimizer, epoch)
#! [tune_model]

#! [save_checkpoint]
torch.save(compressed_model.state_dict(), "compressed_model.pth")
#! [save_checkpoint]

#! [export]
compression_ctrl.export_model("compressed_model.onnx")
#! [export] 

