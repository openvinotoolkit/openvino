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
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/util/common_util.hpp"

namespace InferenceEngine {
IStreamsExecutor::~IStreamsExecutor() {}

IStreamsExecutor::Config::Config(std::string name,
                                 int streams,
                                 int threadsPerStream,
                                 ThreadBindingType threadBindingType,
                                 int threadBindingStep,
                                 int threadBindingOffset,
                                 int threads,
                                 PreferredCoreType threadPreferredCoreType)
    : ov::threading::IStreamsExecutor::Config(
          name,
          {{ov::threading::IStreamsExecutor::Config::streams.name(), streams},
           {ov::threading::IStreamsExecutor::Config::threads_per_stream.name(), threadsPerStream},
           {ov::threading::IStreamsExecutor::Config::thread_binding_type.name(), threadBindingType},
           {ov::threading::IStreamsExecutor::Config::thread_binding_step.name(), threadBindingStep},
           {ov::threading::IStreamsExecutor::Config::thread_binding_offset.name(), threadBindingOffset},
           {ov::threading::IStreamsExecutor::Config::threads.name(), threads},
           {ov::threading::IStreamsExecutor::Config::thread_preferred_core_type.name(), threadPreferredCoreType}}) {}

IStreamsExecutor::Config::Config(const ov::threading::IStreamsExecutor::Config& config)
    : ov::threading::IStreamsExecutor::Config(config) {}

std::vector<std::string> IStreamsExecutor::Config::SupportedKeys() const {
    auto property_names = get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    std::vector<std::string> res;
    res.reserve(property_names.size());
    for (const auto& property : property_names)
        res.emplace_back(property);
    return res;
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

}  //  namespace InferenceEngine
