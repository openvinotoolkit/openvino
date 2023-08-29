# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from utils import get_path_to_model, get_temp_dir
import openvino as ov

import openvino.runtime.properties as props

device_name = 'GNA'
model_path = get_path_to_model()
path_to_cache_dir = get_temp_dir()
# ! [ov:caching:part0]
core = ov.Core()
core.set_property({props.cache_dir(): path_to_cache_dir})
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
core.set_property({props.cache_dir(): path_to_cache_dir})
compiled_model = core.compile_model(model=model_path, device_name=device_name)
# ! [ov:caching:part2]

assert compiled_model

# ! [ov:caching:part3]
# Find 'EXPORT_IMPORT' capability in supported capabilities
caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, props.device.capabilities())
# ! [ov:caching:part3]
