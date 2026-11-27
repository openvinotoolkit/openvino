// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/iinfer_request_factory.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

/**
 * @brief Creates the level zero inference request that matches the graph of a compiled model.
 * @details It holds the context, so the driver stays loaded for as long as a compiled model keeps the factory.
 */
class ZeroInferRequestFactory final : public IInferRequestFactory {
public:
    explicit ZeroInferRequestFactory(const std::shared_ptr<ZeroContext>& initStructs);

    std::shared_ptr<InferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                     const Config& config) override;

private:
    std::shared_ptr<ZeroContext> _initStructs;
};

}  // namespace intel_npu
