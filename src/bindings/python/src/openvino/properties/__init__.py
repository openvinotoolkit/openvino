# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties import Affinity

# Properties
from openvino._pyopenvino.properties import enable_profiling
from openvino._pyopenvino.properties import cache_dir as cd
from openvino._pyopenvino.properties import auto_batch_timeout
from openvino._pyopenvino.properties import num_streams
from openvino._pyopenvino.properties import inference_num_threads
from openvino._pyopenvino.properties import compilation_num_threads
from openvino._pyopenvino.properties import affinity
from openvino._pyopenvino.properties import force_tbb_terminate
from openvino._pyopenvino.properties import enable_mmap
from openvino._pyopenvino.properties import supported_properties as sp
from openvino._pyopenvino.properties import available_devices
from openvino._pyopenvino.properties import model_name
from openvino._pyopenvino.properties import optimal_number_of_infer_requests
from openvino._pyopenvino.properties import range_for_streams
from openvino._pyopenvino.properties import optimal_batch_size
from openvino._pyopenvino.properties import max_batch_size
from openvino._pyopenvino.properties import range_for_async_infer_requests
from openvino._pyopenvino.properties import execution_devices
from openvino._pyopenvino.properties import loaded_from_cache


# Properties as python properties
from openvino.properties.properties import supported_properties
from openvino.properties.properties import cache_dir

# Submodules
from openvino.runtime.properties import hint
from openvino.runtime.properties import intel_cpu
from openvino.runtime.properties import intel_gpu
from openvino.runtime.properties import intel_auto
from openvino.runtime.properties import device
from openvino.runtime.properties import log
from openvino.runtime.properties import streams
