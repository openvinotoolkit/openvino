# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [quantize]
model = TorchModel() # instance of torch.nn.Module
quantized_model = nncf.quantize(model, ...)
#! [quantize]

#! [tune_model]
... # fine-tuning preparations, e.g. dataset, loss, optimization setup, etc.

# tune quantized model for 5 epochs as the baseline
for epoch in range(0, 5):
    for i, data in enumerate(train_loader):
        ... # training loop body
#! [tune_model]

#! [export]
# example_input is an example input to make it possible to trace the model
torch.onnx.export(quantized_model, example_input, './compressed_model.onnx')
#! [export]

#! [save_checkpoint]
checkpoint = {
    'state_dict': model.state_dict(),
    'nncf_config': model.nncf.get_config(),
    ... # the rest of the user-defined objects to save
}
torch.save(checkpoint, path_to_checkpoint)
#! [save_checkpoint]

#! [load_checkpoint]
resuming_checkpoint = torch.load(path_to_checkpoint)
nncf_config = resuming_checkpoint['nncf_config']
quantized_model = nncf.torch.load_from_config(model, nncf_config, example_input)
state_dict = resuming_checkpoint['state_dict']
model.load_state_dict(state_dict)
#! [load_checkpoint]
