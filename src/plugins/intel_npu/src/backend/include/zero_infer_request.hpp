// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "zero_executor.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_remote_tensor.hpp"
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

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    void infer() override;
    void infer_async() override;

    void get_result() override;

private:
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<uint8_t> get_raw_profiling_data() const;

    /**
     * @brief Check the received tensor and set the Level Zero tensor accordingly
     * @param tensor Reference to a tensor.
     * @param name Friendly name of the tensor.
     * @param isParameter True if tensor is a parameter.
     */
    void set_tensor_data(std::shared_ptr<ov::ITensor> tensor, const std::string& name, bool isParameter);

    /**
     * @brief Check the received remote tensor and copy it to the Level Zero tensor
     * @param tensor Reference to a tensor.
     * @param name Friendly name of the tensor.
     */
    void set_remote_tensor_data(std::shared_ptr<ZeroRemoteTensor> tensor, const std::string& name);

    void check_network_precision(const ov::element::Type_t precision) const override;
    void create_pipeline();

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;
    const std::shared_ptr<const IExecutor> _executorPtr;
    const ZeroExecutor* _executor;
    const Config _config;
    Logger _logger;

    ze_device_properties_t _properties = {};

    zeroProfiling::ProfilingPool _profilingPool;
    zeroProfiling::ProfilingQuery _profilingQuery;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npuProfiling;
    std::unique_ptr<Pipeline> _pipeline;
    mutable std::unordered_map<std::string, TensorData> _tensorsData;

    // If batching is handled on the compiler side then batching on the plugin shall be set to 1, we don't do any
    // specific operations on the plugin in this case.
    size_t _batchSize = DEFAULT_BATCH_SIZE;

    bool _createPipeline = true;
    bool _updateCommandList = false;
    bool _levelZeroTensorCreatedLocally = true;
};

}  //  namespace intel_npu
