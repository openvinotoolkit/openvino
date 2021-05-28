// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "cldnn_custom_layer.h"

#include <api/network.hpp>

namespace CLDNNPlugin {

struct Config {
    Config() : throughput_streams(1),
               useProfiling(false),
               dumpCustomKernels(false),
               exclusiveAsyncRequests(false),
               memory_pool_on(true),
               enableDynamicBatch(false),
               enableInt8(true),
               nv12_two_inputs(false),
               enable_fp16_for_quantized_models(true),
               queuePriority(cldnn::priority_mode_types::disabled),
               queueThrottle(cldnn::throttle_mode_types::disabled),
               max_dynamic_batch(1),
               customLayers({}),
               tuningConfig(),
               graph_dumps_dir(""),
               sources_dumps_dir(""),
               device_id(""),
               kernels_cache_dir(""),
               n_threads(std::max(static_cast<unsigned int>(1), std::thread::hardware_concurrency())),
               enable_loop_unrolling(true) {
        adjustKeyMapValues();
    }

    void UpdateFromMap(const std::map<std::string, std::string>& configMap);
    void adjustKeyMapValues();

    uint16_t throughput_streams;
    bool useProfiling;
    bool dumpCustomKernels;
    bool exclusiveAsyncRequests;
    bool memory_pool_on;
    bool enableDynamicBatch;
    bool enableInt8;
    bool nv12_two_inputs;
    bool enable_fp16_for_quantized_models;
    cldnn::priority_mode_types queuePriority;
    cldnn::throttle_mode_types queueThrottle;
    int max_dynamic_batch;
    CLDNNCustomLayerMap customLayers;
    cldnn::tuning_config_options tuningConfig;
    std::string graph_dumps_dir;
    std::string sources_dumps_dir;
    std::string device_id;
    std::string kernels_cache_dir;
    size_t n_threads;
    bool enable_loop_unrolling;

    std::map<std::string, std::string> key_config_map;
};

}  // namespace CLDNNPlugin
