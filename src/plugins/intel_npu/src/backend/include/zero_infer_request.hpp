// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/sync_infer_request.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_remote_tensor.hpp"

namespace intel_npu {

class ZeroInferRequest final : public SyncInferRequest {
public:
    explicit ZeroInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                              const std::shared_ptr<const ICompiledModel>& compiledModel,
                              const Config& config);

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void infer() override;
    void infer_async() override;

    void get_result() override;

private:
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    std::vector<uint8_t> get_raw_profiling_data() const;

    /**
     * @brief Check the received tensor and set the Level Zero tensor accordingly
     * @param tensor Reference to a tensor.
     * @param index The index corresponding to the position of the tensor inside the I/O structures.
     * @param isInput Used for identifying the structures to which the tensor belongs.
     */
    void set_tensor_data(const std::shared_ptr<ov::ITensor>& tensor, const size_t index, const bool isInput);

    /**
     * @brief Check the received remote tensor and copy it to the Level Zero tensor
     * @param tensor Reference to a tensor.
     * @param index The index corresponding to the position of the tensor inside the I/O structures.
     * @param isInput Used for identifying the structures to which the tensor belongs.
     */
    void set_remote_tensor_data(const std::shared_ptr<ZeroRemoteTensor>& tensor,
                                const size_t index,
                                const bool isInput);

    void check_network_precision(const ov::element::Type_t precision) const override;
    void create_pipeline();

    std::shared_ptr<ov::ITensor>& get_level_zero_input(size_t index, size_t tensorNo = 0) const;
    std::vector<std::shared_ptr<ov::ITensor>>& get_level_zero_inputs(size_t index) const;

    std::optional<TensorData>& get_input_tensor_data(size_t index, size_t tensorNo = 0) const;
    std::vector<std::optional<TensorData>>& get_input_tensors_data(size_t index) const;

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;
    const std::shared_ptr<IGraph> _graph;
    const Config _config;
    Logger _logger;

    // A copy of each tensor is needed to maintain the original L0 memory allocation in case the user provides another
    // memory area for the tensor.
    mutable std::vector<std::vector<std::shared_ptr<ov::ITensor>>> _levelZeroInputTensors;
    mutable std::vector<std::shared_ptr<ov::ITensor>> _levelZeroOutputTensors;

    mutable std::vector<std::vector<std::optional<TensorData>>> _inputTensorsData;
    mutable std::vector<std::optional<TensorData>> _outputTensorsData;

    ze_device_properties_t _properties = {};
    std::shared_ptr<const zeroMemory::HostMemAllocator> _inputAllocator;
    std::shared_ptr<const zeroMemory::HostMemAllocator> _outputAllocator;

    zeroProfiling::ProfilingPool _profilingPool;
    zeroProfiling::ProfilingQuery _profilingQuery;
    std::shared_ptr<zeroProfiling::NpuInferProfiling> _npuProfiling;
    std::unique_ptr<Pipeline> _pipeline;

    bool _pipelineIsCreated = false;
};

}  //  namespace intel_npu
