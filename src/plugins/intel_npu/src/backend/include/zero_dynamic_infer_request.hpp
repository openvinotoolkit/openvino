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

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void infer_async() override;

protected:
    void construct_pipeline() override;

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

    IODescriptor prepare_io_descriptor_with_user_info(const IODescriptor& descriptor, bool isInput);

    bool _isTensorChanged = false;
};

}  //  namespace intel_npu
