# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Properties
# Enums
from openvino._pyopenvino.properties import (
    Affinity,
    CacheMode,
    affinity,
    auto_batch_timeout,
    available_devices,
    cache_dir,
    cache_mode,
    compilation_num_threads,
    device,
    enable_mmap,
    enable_profiling,
    execution_devices,
    force_tbb_terminate,
    inference_num_threads,
    intel_auto,
    intel_cpu,
    intel_gpu,
    loaded_from_cache,
    log,
    max_batch_size,
    model_name,
    num_streams,
    optimal_batch_size,
    optimal_number_of_infer_requests,
    range_for_async_infer_requests,
    range_for_streams,
    streams,
    supported_properties,
)

# Submodules
from openvino.runtime.properties import hint
