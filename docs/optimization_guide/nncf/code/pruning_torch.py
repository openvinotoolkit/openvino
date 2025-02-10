# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [imports]
import torch
import nncf  # Important - should be imported right after torch
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
#! [imports]

#! [nncf_congig]
nncf_config_dict = {
    "input_info": {"sample_size": [1, 3, 224, 224]}, # input shape required for model tracing
    "compression": [
        {
            "algorithm": "filter_pruning",
            "pruning_init": 0.1,
            "params": {
                "pruning_target": 0.4,
                "pruning_steps": 15
            }
        },
        {
            "algorithm": "quantization",  # 8-bit quantization with default settings
        },
    ]
}
nncf_config = NNCFConfig.from_dict(nncf_config_dict)
nncf_config = register_default_init_args(nncf_config, train_loader) # train_loader is an instance of torch.utils.data.DataLoader
#! [nncf_congig]

#! [wrap_model]
model = TorchModel() # instance of torch.nn.Module
compression_ctrl, model = create_compressed_model(model, nncf_config)
#! [wrap_model]

#! [distributed]
compression_ctrl.distributed() # call it before the training loop
#! [distributed]

#! [tune_model]
... # fine-tuning preparations, e.g. dataset, loss, optimization setup, etc.

# tune quantized model for 50 epochs as the baseline
for epoch in range(0, 50):
    compression_ctrl.scheduler.epoch_step() # Epoch control API

    for i, data in enumerate(train_loader):
        compression_ctrl.scheduler.step()   # Training iteration control API
        ... # training loop body
#! [tune_model]

#! [export]
compression_ctrl.export_model("compressed_model.onnx")
#! [export]

#! [save_checkpoint]
checkpoint = {
    'state_dict': model.state_dict(),
    'compression_state': compression_ctrl.get_compression_state(),
    ... # the rest of the user-defined objects to save
}
torch.save(checkpoint, path_to_checkpoint)
#! [save_checkpoint]

#! [load_checkpoint]
resuming_checkpoint = torch.load(path_to_checkpoint)
compression_state = resuming_checkpoint['compression_state']
compression_ctrl, model = create_compressed_model(model, nncf_config, compression_state=compression_state)
state_dict = resuming_checkpoint['state_dict']
model.load_state_dict(state_dict)
#! [load_checkpoint]
