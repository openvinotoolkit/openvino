// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "zero_executor.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_utils.hpp"
#include "zero_wrappers.hpp"

namespace {
constexpr std::size_t DEFAULT_BATCH_SIZE = 1;
}  // namespace

namespace intel_npu {

class ZeroInferRequest final : public SyncInferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& backendPtr,
                              const std::shared_ptr<const ICompiledModel>& compiledModel,
                              const std::shared_ptr<const IExecutor>& executor,
                              const Config& config);

    void infer() override;
    void infer_async() override;

    void get_result() override;

private:
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<uint8_t> get_raw_profiling_data() const;

    void check_network_precision(const ov::element::Type_t precision) override;

    const std::shared_ptr<const IExecutor> _executorPtr;
    const ZeroExecutor* _executor;
    const Config _config;
    Logger _logger;

    zeroProfiling::ProfilingPool _profilingPool;
    zeroProfiling::ProfilingQuery _profilingQuery;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npuProfiling;
    std::unique_ptr<Pipeline> _pipeline;

    // If batching is handled on the compiler side then batching on the plugin shall be set to 1, we don't do any
    // specific operations on the plugin in this case.
    size_t _batchSize = DEFAULT_BATCH_SIZE;
};

}  //  namespace intel_npu
