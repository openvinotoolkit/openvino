// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationConvertionFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const std::vector<ov::PartialShape>& input_shapes,
        const element::Type input_type,
        const std::vector<float>& fake_quantize_intervals);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
