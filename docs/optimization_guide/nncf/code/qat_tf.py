# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [quantize]
model = KerasModel() # instance of the tensorflow.keras.Model
quantized_model = nncf.quantize(model, ...)
#! [quantize]

#! [tune_model]
... # fine-tuning preparations, e.g. dataset, loss, optimization setup, etc.

# tune quantized model for 5 epochs the same way as the baseline
quantized_model.fit(train_dataset, epochs=5)
#! [tune_model]

#! [save_checkpoint]
from nncf.tensorflow import ConfigState
from nncf.tensorflow import get_config
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback

nncf_config = get_config(quantized_model)
checkpoint = tf.train.Checkpoint(model=quantized_model,
                                 nncf_config_state=ConfigState(nncf_config),
                                 ... # the rest of the user-defined objects to save
                                 )
callbacks = []
callbacks.append(CheckpointManagerCallback(checkpoint, path_to_checkpoint))
...
quantized_model.fit(..., callbacks=callbacks)
#! [save_checkpoint]

#! [load_checkpoint]
from nncf.tensorflow import ConfigState
from nncf.tensorflow import load_from_config

checkpoint = tf.train.Checkpoint(nncf_config_state=ConfigState())
checkpoint.restore(path_to_checkpoint)

quantized_model = load_from_config(model, checkpoint.nncf_config_state.config)

checkpoint = tf.train.Checkpoint(model=quantized_model
                                 ... # the rest of the user-defined objects to load
                                 )
checkpoint.restore(path_to_checkpoint)
#! [load_checkpoint]
