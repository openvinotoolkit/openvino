// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/sync_infer_request.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_pipeline.hpp"
#include "zero_tensor.hpp"

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

    void check_network_precision(const ov::element::Type_t precision) const override;
    void create_pipeline();

    std::shared_ptr<ZeroTensor>& get_level_zero_input(size_t index, size_t tensorNo = 0) const;
    std::vector<std::shared_ptr<ZeroTensor>>& get_level_zero_inputs(size_t index) const;

    /**
     * @brief Allocates a tensor on host and stores the reference inside multiple attributes.
     * @param index The index which the allocated tensor shall use.
     * @param isInput Determines the containers in which the newly allocated tensors will be stored.
     * @param allocator If provided, the tensor uses the custom allocator instead of using the default one.
     * @param batchSize If provided, the value of the shape on the 0th axis is overriden with this value.
     * @return Pointer towards the allocated tensor
     */
    std::shared_ptr<ZeroTensor> allocate_tensor(const size_t index,
                                                const bool isInput,
                                                const std::optional<std::size_t> batchSize = std::nullopt) const;

    void add_state(const IODescriptor& descriptor, size_t tensorIndex) const override;

    void update_pipeline_if_memory_changed();
    void update_states_if_memory_changed();

    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;
    const std::shared_ptr<IGraph> _graph;
    const Config _config;
    Logger _logger;

    const std::vector<ArgumentDescriptor>& _graphInputDescriptors;
    const std::vector<ArgumentDescriptor>& _graphOutputDescriptors;

    // A copy of each tensor is needed to maintain the original L0 memory allocation in case the user provides another
    // memory area for the tensor.
    mutable std::vector<std::vector<std::shared_ptr<ZeroTensor>>> _levelZeroInputTensors;
    mutable std::vector<std::shared_ptr<ZeroTensor>> _levelZeroOutputTensors;

    mutable std::vector<bool> _levelZeroInputTensorsSharedWithUser;
    mutable std::vector<bool> _levelZeroOutputTensorsSharedWithUser;

    std::unique_ptr<Pipeline> _pipeline;

    bool _pipelineIsCreated = false;
    bool _dynamicBatchValueChanged = false;
    bool _externalMemoryStandardAllocationSupported = false;
};

}  //  namespace intel_npu
