// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cldnn/runtime/engine.hpp>
#include "cldnn/runtime/device_query.hpp"

namespace CLDNNPlugin {
class clDNNEngineFactory {
public:
    static std::shared_ptr<cldnn::engine> create(const Config& config, const cldnn::device::ptr& dev,
                                        InferenceEngine::gpu_handle_param external_queue = nullptr, bool is_dummy = false) {
        auto engine_type = cldnn::engine_types::ocl;
        auto runtime_type = cldnn::runtime_types::ocl;
        bool enable_profiling = (config.useProfiling ||
                (config.tuningConfig.mode == cldnn::tuning_mode::tuning_tune_and_cache) ||
                (config.tuningConfig.mode == cldnn::tuning_mode::tuning_retune_and_cache));
        cldnn::queue_types queue_type;
        if (external_queue) {
            queue_type = cldnn::stream::detect_queue_type(engine_type, external_queue);
        } else if (dev->get_info().supports_immad) {
            queue_type = cldnn::queue_types::in_order;
        } else {
            queue_type = cldnn::queue_types::out_of_order;
        }
        bool use_unified_shared_memory = true;
        InferenceEngine::ITaskExecutor::Ptr task_executor = std::make_shared<InferenceEngine::CPUStreamsExecutor>(config.task_exec_config);
        return cldnn::engine::create(engine_type, runtime_type, dev, cldnn::engine_configuration(enable_profiling,
                    queue_type,
                    !is_dummy ? config.sources_dumps_dir : std::string(),
                    config.queuePriority,
                    config.queueThrottle,
                    config.memory_pool_on,
                    use_unified_shared_memory,
                    !is_dummy ? config.kernels_cache_dir : std::string(),
                    config.throughput_streams),
                    task_executor);
    }

private:
    clDNNEngineFactory() = delete;
    ~clDNNEngineFactory() = delete;
};
}; // namespace CLDNNPlugin
