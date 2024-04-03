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

    zeroProfiling::ProfilingPool _profiling_pool;
    zeroProfiling::ProfilingQuery _profiling_query;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npu_profiling;
    std::unique_ptr<Pipeline> _pipeline;
};

}  //  namespace intel_npu
