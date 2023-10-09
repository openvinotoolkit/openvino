// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/function.hpp"

#include <cstring>

#include "openvino/core/deprecated.hpp"
#include "openvino/core/shape_util.hpp"

namespace ov {
namespace reference {
void function(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    const auto& results = function->get_results();
    outputs.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto shape = results[i]->get_output_partial_shape(0).is_static() ? results[i]->get_output_shape(0)
                                                                         : ov::util::make_dynamic_shape();
        OPENVINO_SUPPRESS_DEPRECATED_END
        outputs.push_back(ov::Tensor(results[i]->get_element_type(), shape));
    }
    function->evaluate(outputs, inputs);
}
}  // namespace reference
}  // namespace ov
