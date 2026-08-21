# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ! [ov:caching:part0]
from utils import get_path_to_model, get_temp_dir
import openvino as ov

import openvino.properties as props

# For example: "CPU", "GPU", "NPU".
device_name = 'CPU'
model_path = get_path_to_model()
path_to_cache_dir = get_temp_dir()

core = ov.Core()
core.set_property({props.cache_dir: path_to_cache_dir})
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device_name)
# ! [ov:caching:part0]

assert compiled_model

# ! [ov:caching:part1]
core = ov.Core()
compiled_model = core.compile_model(model=model_path, device_name=device_name)
# ! [ov:caching:part1]

assert compiled_model

# ! [ov:caching:part2]
core = ov.Core()
core.set_property({props.cache_dir: path_to_cache_dir})
compiled_model = core.compile_model(model=model_path, device_name=device_name)
# ! [ov:caching:part2]

assert compiled_model

# ! [ov:caching:part3]
import openvino.properties.device as device

# Find 'EXPORT_IMPORT' capability in supported capabilities
caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, device.capabilities)
# ! [ov:caching:part3]

# ! [ov:caching:part4]
core = ov.Core()
if "GPU" in core.available_devices:
    core.set_property({props.cache_dir: path_to_cache_dir})
    config_cache = {}
    config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
    # Note: model path needs to point to the *.xml file, not *.bin when using the IR model format.
    compiled_model = core.compile_model(model=model_path, device_name='GPU', config=config_cache)
# ! [ov:caching:part4]

# ! [ov:caching:part5]
import base64

def encrypt_base64(src: bytes):
    return base64.b64encode(src)

def decrypt_base64(src: bytes):
    return base64.b64decode(src)

core = ov.Core()
core.set_property({props.cache_dir: path_to_cache_dir})
config_cache = {}
config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device_name, config=config_cache)
# ! [ov:caching:part5]

# ! [ov:caching:part6]
import base64

def encrypt_base64(src: bytes):
    return base64.b64encode(src)

def decrypt_base64(src: bytes):
    return base64.b64decode(src)

core = ov.Core()
if any("GPU" in device for device in core.available_devices):
    core.set_property({props.cache_dir: path_to_cache_dir})
    config_cache = {}
    config_cache["CACHE_ENCRYPTION_CALLBACKS"] = [encrypt_base64, decrypt_base64]
    config_cache["CACHE_MODE"] = "OPTIMIZE_SIZE"
    compiled_model = core.compile_model(model=model_path, device_name='GPU', config=config_cache)
# ! [ov:caching:part6]

# ! [ov:caching:part7]
import numpy as np
import openvino as ov
from utils import get_path_to_model

# For example: "CPU", "GPU", "NPU".
device_name = 'CPU'
model_path = get_path_to_model()

core = ov.Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name=device_name)

# Export the compiled model to bytes
user_stream = compiled_model.export_model()

# Wrap the exported bytes in an ov.Tensor
compiled_blob = ov.Tensor(np.frombuffer(user_stream.getvalue(), dtype=np.uint8))

# Import the compiled model back from the Tensor
imported_model = core.import_model(compiled_blob, device_name)
# ! [ov:caching:part7]

assert imported_model

# ! [ov:caching:part8]
core = ov.Core()

# props.enable_weightless takes priority over CACHE_MODE when both are set.
compiled_model = core.compile_model(model=model_path, device_name="CPU", config={props.enable_weightless: True})
# ! [ov:caching:part8]

assert compiled_model

# ! [ov:caching:part9]
import numpy as np
import openvino.properties.hint as hints

core = ov.Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU", config={props.enable_weightless: True})

# Export a weightless blob: it does not contain the model weights.
user_stream = compiled_model.export_model()
compiled_blob = ov.Tensor(np.frombuffer(user_stream.getvalue(), dtype=np.uint8))

# The weights must be supplied separately when importing a weightless blob:
# Option 1: a path to the original weights file.
imported_from_path = core.import_model(compiled_blob, "CPU", config={props.weights_path: model_path.replace(".xml", ".bin")})

# Option 2: the original ov.Model object.
imported_from_model = core.import_model(compiled_blob, "CPU", config={hints.model: model})
# ! [ov:caching:part9]

assert imported_from_path
assert imported_from_model
