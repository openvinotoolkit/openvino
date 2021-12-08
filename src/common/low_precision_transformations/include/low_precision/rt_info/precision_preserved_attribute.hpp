// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ngraph {
class LP_TRANSFORMATIONS_API PrecisionPreservedAttribute : public SharedAttribute<bool> {
public:
    OPENVINO_RTTI("LowPrecision::PrecisionPreserved", "", ov::RuntimeAttribute, 0);

    PrecisionPreservedAttribute() = default;
    PrecisionPreservedAttribute(const bool value);

    std::string to_string() const override;
};

} // namespace ngraph
