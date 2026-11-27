// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/icompiled_model.hpp"
#include "intel_npu/common/infer_request.hpp"
#include "intel_npu/config/config.hpp"

namespace intel_npu {

/**
 * @brief Creates the inference request that matches a compiled model.
 * @details The backend decides which request class runs the model, the compiled model only asks for one. Tests give
 * the compiled model a fake factory.
 */
class IInferRequestFactory {
public:
    virtual ~IInferRequestFactory() = default;

    virtual std::shared_ptr<InferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>& compiledModel,
                                                             const Config& config) = 0;
};

}  // namespace intel_npu
