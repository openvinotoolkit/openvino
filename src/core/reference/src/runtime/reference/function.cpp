// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/function.hpp"

#include <cstring>

#include "ngraph/opsets/opset5.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "openvino/core/deprecated.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void function(const std::shared_ptr<ngraph::Function>& function,
              const HostTensorVector& inputs,
              HostTensorVector& outputs) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto& results = function->get_results();
    outputs.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputs.push_back(std::make_shared<HostTensor>());
    }
    function->evaluate(outputs, inputs);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
