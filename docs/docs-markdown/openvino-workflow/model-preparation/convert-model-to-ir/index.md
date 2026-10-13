---
sidebar_position: 1
sidebar_label: "Convert to OpenVINO IR"
description: "Convert models from the original framework to OpenVINO representation."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Convert to OpenVINO IR


[OpenVINO IR](../../../documentation/openvino-ir-format/index.md) is the proprietary model format
used by OpenVINO, typically obtained by converting models of supported frameworks:

<Tabs>
  <TabItem value="torch" label="PyTorch">
  <Tabs>
    <TabItem value="py" label="Python">
    - The `convert_model()` method:
      
      This is the only method applicable to PyTorch models.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Python objects**:
        
        - `torch.nn.Module`
        - `torch.jit.ScriptModule`
        - `torch.jit.ScriptFunction`
        - `torch.export.ExportedProgram`

      </details>
      
      ```python
      import torchvision
      import openvino as ov

      model = torchvision.models.resnet50(weights='DEFAULT')
      ov_model = ov.convert_model(model)
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For more details on conversion, refer to the
      [guide](../convert-model-pytorch.md)
      and an example [tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb)
      on this topic.
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="tf" label="TensorFlow">
  <Tabs>
    <TabItem value="py" label="Python">
    - The `convert_model()` method:
      
      When you use the `convert_model()` method, you have more control and you can
      specify additional adjustments for `ov.Model`. The `read_model()` and
      `compile_model()` methods are easier to use, however, they do not have such
      capabilities. With `ov.Model` you can choose to optimize, compile and run
      inference on it or serialize it into a file for subsequent use.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - SavedModel - `<SAVED_MODEL_DIRECTORY>` or `<INPUT_MODEL>.pb`
        - Checkpoint - `<INFERENCE_GRAPH>.pb` or `<INFERENCE_GRAPH>.pbtxt`
        - MetaGraph - `<INPUT_META_GRAPH>.meta`
      - **Python objects**:
        
        - `tf.keras.Model`
        - `tf.keras.layers.Layer`
        - `tf.Module`
        - `tf.compat.v1.Graph`
        - `tf.compat.v1.GraphDef`
        - `tf.function`
        - `tf.compat.v1.session`
        - `tf.train.checkpoint`

      </details>
      
      ```python
      import openvino as ov

      ov_model = ov.convert_model("saved_model.pb")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For more details on conversion, refer to the
      [guide](../convert-model-tensorflow.md)
      and an example [tutorial](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-classification-to-openvino)
      on this topic.
    - The `read_model()` and `compile_model()` methods:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - SavedModel - `<SAVED_MODEL_DIRECTORY>` or `<INPUT_MODEL>.pb`
        - Checkpoint - `<INFERENCE_GRAPH>.pb` or `<INFERENCE_GRAPH>.pbtxt`
        - MetaGraph - `<INPUT_META_GRAPH>.meta`

      </details>
      
      ```python
      import openvino as ov

      core = ov.Core()
      ov_model = core.read_model("saved_model.pb")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cpp" label="C++">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - SavedModel - `<SAVED_MODEL_DIRECTORY>` or `<INPUT_MODEL>.pb`
        - Checkpoint - `<INFERENCE_GRAPH>.pb` or `<INFERENCE_GRAPH>.pbtxt`
        - MetaGraph - `<INPUT_META_GRAPH>.meta`

      </details>
      
      ```cpp
      ov::CompiledModel compiled_model = core.compile_model("saved_model.pb", "AUTO");
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="c" label="C">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - SavedModel - `<SAVED_MODEL_DIRECTORY>` or `<INPUT_MODEL>.pb`
        - Checkpoint - `<INFERENCE_GRAPH>.pb` or `<INFERENCE_GRAPH>.pbtxt`
        - MetaGraph - `<INPUT_META_GRAPH>.meta`

      </details>
      
      ```c
      ov_compiled_model_t* compiled_model = NULL;
      ov_core_compile_model_from_file(core, "saved_model.pb", "AUTO", 0, &compiled_model);
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cli" label="CLI">
    You can use `ovc` command-line tool to convert a model to IR. The obtained IR can
    then be read by `read_model()` and inferred.

    ```bash
    ovc <INPUT_MODEL>.pb
    ```

    For details on the conversion, refer to the
    [article](../index.md).
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="tflite" label="TensorFlow Lite">
  <Tabs>
    <TabItem value="py" label="Python">
    - The `convert_model()` method:
      
      When you use the `convert_model()` method, you have more control and you can
      specify additional adjustments for `ov.Model`. The `read_model()` and
      `compile_model()` methods are easier to use, however, they do not have such
      capabilities. With `ov.Model` you can choose to optimize, compile and run
      inference on it or serialize it into a file for subsequent use.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```python
      import openvino as ov

      ov_model = ov.convert_model("<INPUT_MODEL>.tflite")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For more details on conversion, refer to the
      [guide](../convert-model-tensorflow-lite.md)
      and an example [tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/tflite-to-openvino/tflite-to-openvino.ipynb)
      on this topic.
    - The `read_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```python
      import openvino as ov

      core = ov.Core()
      ov_model = core.read_model("<INPUT_MODEL>.tflite")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```python
      import openvino as ov

      compiled_model = ov.compile_model("<INPUT_MODEL>.tflite", "AUTO")
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cpp" label="C++">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```cpp
      ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.tflite", "AUTO");
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="c" label="C">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```c
      ov_compiled_model_t* compiled_model = NULL;
      ov_core_compile_model_from_file(core, "<INPUT_MODEL>.tflite", "AUTO", 0, &compiled_model);
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cli" label="CLI">
    - The `convert_model()` method:
      
      You can use `ovc` to convert a model to IR. The obtained IR can
      then be read by `read_model()` and inferred.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.tflite`

      </details>
      
      ```bash
      ovc <INPUT_MODEL>.tflite
      ```
      
      For details on the conversion, refer to the
      [article](../convert-model-tensorflow-lite.md).
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  <Tabs>
    <TabItem value="py" label="Python">
    - The `convert_model()` method:
      
      When you use the `convert_model()` method, you have more control and you can
      specify additional adjustments for `ov.Model`. The `read_model()` and
      `compile_model()` methods are easier to use, however, they do not have such
      capabilities. With `ov.Model` you can choose to optimize, compile and run
      inference on it or serialize it into a file for subsequent use.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```python
      import openvino as ov

      ov_model = ov.convert_model("<INPUT_MODEL>.onnx")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For more details on conversion, refer to the
      [guide](../convert-model-onnx.md)
      and an example [tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb)
      on this topic.
    - The `read_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```python
      import openvino as ov

      core = ov.Core()
      ov_model = core.read_model("<INPUT_MODEL>.onnx")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```python
      import openvino as ov

      compiled_model = ov.compile_model("<INPUT_MODEL>.onnx", "AUTO")
      ```
      
      For a guide on how to run inference, see how to [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cpp" label="C++">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```cpp
      ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.onnx", "AUTO");
      ```
      
      For a guide on how to run inference, see how to [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="c" label="C">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```c
      ov_compiled_model_t* compiled_model = NULL;
      ov_core_compile_model_from_file(core, "<INPUT_MODEL>.onnx", "AUTO", 0, &compiled_model);
      ```
      
      For details on the conversion, refer to the [article](../convert-model-onnx.md)
    </TabItem>
    <TabItem value="cli" label="CLI">
    - The `convert_model()` method:
      
      You can use `ovc` to convert a model to IR. The obtained IR
      can then be read by `read_model()` and inferred.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.onnx`

      </details>
      
      ```bash
      ovc <INPUT_MODEL>.onnx
      ```
      
      For details on the conversion, refer to the
      [article](../convert-model-onnx.md)
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="pdpd" label="PaddlePaddle">
  <Tabs>
    <TabItem value="py" label="Python">
    - The `convert_model()` method:
      
      When you use the `convert_model()` method, you have more control and you can
      specify additional adjustments for `ov.Model`. The `read_model()` and
      `compile_model()` methods are easier to use, however, they do not have such
      capabilities. With `ov.Model` you can choose to optimize, compile and run
      inference on it or serialize it into a file for subsequent use.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`
      - **Python objects**:
        
        - `paddle.hapi.model.Model`
        - `paddle.fluid.dygraph.layers.Layer`
        - `paddle.fluid.executor.Executor`

      </details>
      
      ```python
      import openvino as ov

      ov_model = ov.convert_model("<INPUT_MODEL>.pdmodel")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
      
      For more details on conversion, refer to the
      [guide](../convert-model-paddle.md)
      and an example [tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/paddle-to-openvino/paddle-to-openvino-classification.ipynb)
      on this topic.
    - The `read_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`

      </details>
      
      ```python
      import openvino as ov

      core = ov.Core()
      ov_model = core.read_model("<INPUT_MODEL>.pdmodel")
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`

      </details>
      
      ```python
      import openvino as ov

      compiled_model = ov.compile_model("<INPUT_MODEL>.pdmodel", "AUTO")
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cpp" label="C++">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`

      </details>
      
      ```cpp
      ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.pdmodel", "AUTO");
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="c" label="C">
    - The `compile_model()` method:
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`

      </details>
      
      ```c
      ov_compiled_model_t* compiled_model = NULL;
      ov_core_compile_model_from_file(core, "<INPUT_MODEL>.pdmodel", "AUTO", 0, &compiled_model);
      ```
      
      For a guide on how to run inference, see how to
      [Integrate OpenVINO™ with Your Application](../../running-inference/index.md).
    </TabItem>
    <TabItem value="cli" label="CLI">
    - The `convert_model()` method:
      
      You can use `ovc` to convert a model to IR. The obtained IR
      can then be read by `read_model()` and inferred.
      
      <details>
      <summary>List of supported formats:</summary>

      - **Files**:
        
        - `<INPUT_MODEL>.pdmodel`

      </details>
      
      ```bash
      ovc <INPUT_MODEL>.pdmodel
      ```
      
      For details on the conversion, refer to the
      [article](../convert-model-paddle.md).
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="torch-2" label="JAX/Flax">
  <Tabs>
    <TabItem value="py" label="Python">
    The `convert_model()` method is the only method applicable to JAX/Flax models.

    <details>
    <summary>List of supported formats:</summary>

    - **Python objects**:
      
      - `jax._src.core.ClosedJaxpr`
      - `flax.linen.Module`

    </details>

    - Conversion of the `jax._src.core.ClosedJaxpr` object
      
      ```python
      import jax
      import jax.numpy as jnp
      import openvino as ov

      # let user have some JAX function
      def jax_func(x, y):
          return jax.lax.tanh(jax.lax.max(x, y))

      # use example inputs for creation of ClosedJaxpr object
      x = jnp.array([1.0, 2.0])
      y = jnp.array([-1.0, 10.0])
      jaxpr = jax.make_jaxpr(jax_func)(x, y)

      ov_model = ov.convert_model(jaxpr)
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```
    - Conversion of the `flax.linen.Module` object
      
      ```python
      import flax.linen as nn
      import jax
      import jax.numpy as jnp
      import openvino as ov

      # let user have some Flax module
      class SimpleDenseModule(nn.Module):
          features: int

          @nn.compact
          def __call__(self, x):
              return nn.Dense(features=self.features)(x)

      module = SimpleDenseModule(features=4)

      # create example_input used in training
      example_input = jnp.ones((2, 3))

      # prepare parameters to initialize the module
      # they can be also loaded from a disk
      # using pickle, flax.serialization for deserialization
      key = jax.random.PRNGKey(0)
      params = module.init(key, example_input)
      module = module.bind(params)

      ov_model = ov.convert_model(module, example_input=example_input)
      compiled_model = ov.compile_model(ov_model, "AUTO")
      ```

    For more details on conversion, refer to the [conversion guide](../convert-model-jax.md).
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

These are basic examples, for detailed conversion instructions, see the individual guides on
[PyTorch](../convert-model-pytorch.md), [ONNX](../convert-model-onnx.md),
[TensorFlow](../convert-model-tensorflow.md), [TensorFlow Lite](../convert-model-tensorflow-lite.md),
[PaddlePaddle](../convert-model-paddle.md), and [JAX/Flax](../convert-model-jax.md).

Refer to the list of all supported conversion options in [Conversion Parameters](../conversion-parameters.md).

## IR Conversion Benefits

**Saving to IR to improve first inference latency**  
When first inference latency matters, rather than convert the framework model each time it
is loaded, which may take some time depending on its size, it is better to do it once. Save
the model as an OpenVINO IR with `save_model` and then load it with `read_model` as
needed. This should improve the time it takes the model to make the first inference as it
avoids the conversion step.

**Saving to IR in FP16 to save space**  
Save storage space, even more so if FP16 is used as it may cut the size by about 50%,
especially useful for large models, like Llama2-7B.

**Saving to IR to avoid large dependencies in inference code**  
Frameworks such as TensorFlow, PyTorch, and JAX/Flax tend to be large dependencies for applications
running inference (multiple gigabytes). Converting models to OpenVINO IR removes this
dependency, as OpenVINO can run its inference with no additional components.
This way, much less disk space is needed, while loading and compiling usually takes less
runtime memory than loading the model in the source framework and then converting
and compiling it.

Here is an example of how to benefit from OpenVINO IR, saving a model once and running it
multiple times:

```python
# Run once

import openvino as ov
import tensorflow as tf

# 1. Convert model created with TF code
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
ov_model = ov.convert_model(model)

# 2. Save model as OpenVINO IR
ov.save_model(ov_model, 'model.xml', compress_to_fp16=True) # enabled by default

# Repeat as needed

import openvino as ov

# 3. Load model from file
core = ov.Core()
ov_model = core.read_model("model.xml")

# 4. Compile model from memory
compiled_model = ov.compile_model(ov_model)
```

## Additional Resources

- Learn about the [parameters to adjust model conversion](../conversion-parameters.md).
- [Download models from Hugging Face](https://huggingface.co/models).
