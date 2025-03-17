# Copyright (C) 2018-2025 Intel Corporation
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
