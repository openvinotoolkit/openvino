# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [imports]
import tensorflow as tf

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
nncf_config = register_default_init_args(nncf_config, dataset, batch_size=1) # dataset is an instance of tf.data.Dataset
#! [nncf_congig]

#! [wrap_model]
model = KerasModel() # instance of the tensorflow.python.keras.models.Model
compression_ctrl, model = create_compressed_model(model, nncf_config)
#! [wrap_model]

#! [distributed]
compression_ctrl.distributed() # call it before the training
#! [distributed]

#! [tune_model]
...
# tune quantized model for 5 epochs the same way as the baseline
model.fit(train_dataset, epochs=5)
#! [tune_model]

#! [export]
compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph') #export to Frozen Graph
#! [export] 

#! [save_checkpoint]
compression_ctrl.export_model("compressed_model", save_format='saved_model')
#! [save_checkpoint]

#! [load_checkpoint]
compression_ctrl.export_model("compressed_model", save_format='saved_model')
#! [load_checkpoint]