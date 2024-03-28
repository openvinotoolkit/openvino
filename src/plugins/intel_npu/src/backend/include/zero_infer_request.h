// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <mutex>

#include "npu/utils/logger/logger.hpp"
#include "vpux.hpp"
#include "zero_executor.h"
#include "zero_pipeline.h"
#include "zero_profiling.h"
#include "zero_utils.h"
#include "zero_wrappers.h"

namespace vpux {

class ZeroInferRequest final : public SyncInferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& backendPtr,
                              const std::shared_ptr<const vpux::ICompiledModel>& compiledModel,
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

    vpux::zeroProfiling::ProfilingPool _profiling_pool;
    vpux::zeroProfiling::ProfilingQuery _profiling_query;
    std::shared_ptr<vpux::zeroProfiling::VpuInferProfiling> _vpu_profiling;
    std::unique_ptr<Pipeline> _pipeline;
};

}  //  namespace vpux
