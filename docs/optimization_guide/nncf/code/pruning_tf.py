# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [imports]
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, create_compression_callbacks, \
                            register_default_init_args
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
nncf_config = register_default_init_args(nncf_config, train_dataset, batch_size=1) # train_dataset is an instance of tf.data.Dataset
#! [nncf_congig]

#! [wrap_model]
model = KerasModel() # instance of the tensorflow.keras.Model
compression_ctrl, model = create_compressed_model(model, nncf_config)
#! [wrap_model]

#! [distributed]
compression_ctrl.distributed() # call it before the training
#! [distributed]

#! [tune_model]
... # fine-tuning preparations, e.g. dataset, loss, optimization setup, etc.

# create compression callbacks to control pruning parameters and dump compression statistics
# all the setting are being taked from compression_ctrl, i.e. from NNCF config
compression_callbacks = create_compression_callbacks(compression_ctrl, log_dir="./compression_log")

# tune quantized model for 50 epochs as the baseline
model.fit(train_dataset, epochs=50, callbacks=compression_callbacks)
#! [tune_model]

#! [export]
compression_ctrl.export_model("compressed_model.pb") #export to Frozen Graph
#! [export]

#! [save_checkpoint]
from nncf.tensorflow.utils.state import TFCompressionState
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback

checkpoint = tf.train.Checkpoint(model=model,
                                 compression_state=TFCompressionState(compression_ctrl),
                                 ... # the rest of the user-defined objects to save
                                 )
callbacks = []
callbacks.append(CheckpointManagerCallback(checkpoint, path_to_checkpoint))
...
model.fit(..., callbacks=callbacks)
#! [save_checkpoint]

#! [load_checkpoint]
from nncf.tensorflow.utils.state import TFCompressionStateLoader

checkpoint = tf.train.Checkpoint(compression_state=TFCompressionStateLoader())
checkpoint.restore(path_to_checkpoint)
compression_state = checkpoint.compression_state.state

compression_ctrl, model = create_compressed_model(model, nncf_config, compression_state)
checkpoint = tf.train.Checkpoint(model=model,
                                 ...)
checkpoint.restore(path_to_checkpoint)
#! [load_checkpoint]
