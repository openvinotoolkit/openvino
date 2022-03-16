# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino.runtime import Core

device_name = 'GNA'
xml_path = '/tmp/myModel.xml'
# ! [ov:caching:part0]
core = Core()
core.set_property({'CACHE_DIR': '/path/to/cache/dir'})
model = core.read_model(model=xml_path)
compiled_model = core.compile_model(model=model, device_name=device_name)
# ! [ov:caching:part0]

assert compiled_model

# ! [ov:caching:part1]
core = Core()
compiled_model = core.compile_model(model_path=xml_path, device_name=device_name)
# ! [ov:caching:part1]

assert compiled_model

# ! [ov:caching:part2]
core = Core()
core.set_property({'CACHE_DIR': '/path/to/cache/dir'})
compiled_model = core.compile_model(model_path=xml_path, device_name=device_name)
# ! [ov:caching:part2]

assert compiled_model

# ! [ov:caching:part3]
# Find 'EXPORT_IMPORT' capability in supported capabilities
caching_supported = 'EXPORT_IMPORT' in core.get_property(device_name, 'OPTIMIZATION_CAPABILITIES')
# ! [ov:caching:part3]
