# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile

def convert_paddle_to_pdmodel(model, inputs=None, outputs=None):
    '''
        There are three paddle model categories:
        - High Level API: is a wrapper for dynamic or static model, use `self.save` to serialize
        - Dynamic Model: use `paddle.jit.save` to serialize
        - Static Model: use `paddle.static.save_inference_model` to serialize
    '''
    model_name = None
    tmp = tempfile.NamedTemporaryFile(delete=True)
    model_name = tmp.name
    try:
        import paddle
        if isinstance(model, paddle.hapi.model.Model):
            model.save(model_name, False)
        else:
            if inputs is None:
                raise RuntimeError(
                    "Saving inference model needs 'inputs' before saving. Please specify 'example_input'"
                )
            if isinstance(model, paddle.fluid.dygraph.layers.Layer):
                with paddle.fluid.framework._dygraph_guard(None):
                    paddle.jit.save(model, model_name, input_spec=inputs)
            elif isinstance(model, paddle.fluid.executor.Executor):
                if outputs is None:
                    raise RuntimeError(
                        "Model is static. Saving inference model needs 'outputs' before saving. Please specify 'example_output' for this model"
                    )
                paddle.static.save_inference_model(model_name, inputs, outputs, model)
            else:
                raise RuntimeError(
                    "Conversion just support paddle.hapi.model.Model, paddle.fluid.dygraph.layers.Layer and paddle.fluid.executor.Executor"
                )

        model_file = "{}.pdmodel".format(model_name)
        if not os.path.exists(model_file):
            print("Failed generating paddle inference format model")
            sys.exit(1)

        return model_file
    finally:
        if isinstance(tmp, tempfile._TemporaryFileWrapper):
            tmp.close()