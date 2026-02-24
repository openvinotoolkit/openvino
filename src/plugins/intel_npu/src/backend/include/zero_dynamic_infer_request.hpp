// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_dynamic_pipeline.hpp"
#include "zero_infer_request.hpp"

namespace intel_npu {

class ZeroDynamicInferRequest final : public ZeroInferRequest {
public:
    explicit ZeroDynamicInferRequest(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                     const std::shared_ptr<const ICompiledModel>& compiledModel,
                                     const Config& config);

    void infer_async() override;

protected:
    void construct_pipeline() override;

    /**
     * @brief Allocates a tensor on host and stores the reference inside multiple attributes.
     * @param index The index which the allocated tensor shall use.
     * @param isInput Determines the containers in which the newly allocated tensors will be stored.
     * @param batchSize If provided, the value of the shape on the 0th axis is overriden with this value.
     * @return Pointer towards the allocated tensor
     */
    std::shared_ptr<ZeroTensor> allocate_tensor(const size_t index,
                                                const bool isInput,
                                                const std::optional<std::size_t> batchSize = std::nullopt) const;

    void update_command_list_for_tensor(SyncInferRequest::FoundPort& foundPort,
                                        const ov::SoPtr<ov::ITensor>& tensor) override;

    void update_command_list_for_tensors(SyncInferRequest::FoundPort& foundPort,
                                         const std::vector<ov::SoPtr<ov::ITensor>>& tensors,
                                         std::optional<size_t> batchSizeCandidate = std::nullopt) override;
    void predict_shapes(std::vector<IDynamicGraph::MemRefType>& outputProps);

    void check_tensor_and_predicted_shapes(const std::vector<IDynamicGraph::MemRefType>& outputProps);

    void update_tensor(const std::vector<IDynamicGraph::MemRefType>& outputProps);

    bool _isTensorChanged = false;
};

}  //  namespace intel_npu
