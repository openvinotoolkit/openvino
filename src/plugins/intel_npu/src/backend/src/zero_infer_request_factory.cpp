// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_infer_request_factory.hpp"

#include "intel_npu/common/igraph.hpp"
#include "openvino/core/except.hpp"
#include "zero_dynamic_infer_request.hpp"
#include "zero_infer_request.hpp"

namespace intel_npu {

ZeroInferRequestFactory::ZeroInferRequestFactory(const std::shared_ptr<ZeroContext>& initStructs)
    : _initStructs(initStructs) {
    OPENVINO_ASSERT(_initStructs != nullptr, "The inference request factory needs a level zero context");
}

std::shared_ptr<InferRequest> ZeroInferRequestFactory::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const Config& config) {
    if (compiledModel->get_graph()->get_kind() == GraphKind::Dynamic) {
        return std::make_shared<ZeroDynamicInferRequest>(_initStructs, compiledModel, config);
    }
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, config);
}

}  // namespace intel_npu
