// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_istreams_executor.hpp"

#include <algorithm>
#include <openvino/util/log.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_parallel.hpp"
#include "ie_parallel_custom_arena.hpp"
#include "ie_parameter.hpp"
#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

namespace InferenceEngine {
IStreamsExecutor::~IStreamsExecutor() {}

std::vector<std::string> IStreamsExecutor::Config::SupportedKeys() const {
    return get_property(ov::supported_properties.name()).as<std::vector<std::string>>();
}
int IStreamsExecutor::Config::GetDefaultNumStreams(const bool enable_hyper_thread) {
    return get_default_num_streams(enable_hyper_thread);
}

int IStreamsExecutor::Config::GetHybridNumStreams(std::map<std::string, std::string>& config, const int stream_mode) {
    return get_hybrid_num_streams(config, stream_mode);
}

void IStreamsExecutor::Config::SetConfig(const std::string& key, const std::string& value) {
    set_property(key, value);
}

Parameter IStreamsExecutor::Config::GetConfig(const std::string& key) const {
    return get_property(key);
}

void IStreamsExecutor::Config::UpdateHybridCustomThreads(Config& config) {
    return update_hybrid_custom_threads(config);
}

IStreamsExecutor::Config IStreamsExecutor::Config::MakeDefaultMultiThreaded(const IStreamsExecutor::Config& initial,
                                                                            const bool fp_intesive) {
    return make_default_multi_threaded(initial);
}

IStreamsExecutor::Config IStreamsExecutor::Config::ReserveCpuThreads(const Config& initial) {
    return reserve_cpu_threads(initial);
}

}  //  namespace InferenceEngine
