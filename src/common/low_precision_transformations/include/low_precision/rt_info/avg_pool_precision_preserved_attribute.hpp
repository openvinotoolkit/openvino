// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "openvino/core/visibility.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

namespace ngraph {
class OPENVINO_API AvgPoolPrecisionPreservedAttribute : public PrecisionPreservedAttribute {
public:
    OPENVINO_RTTI("LowPrecision::AvgPoolPrecisionPreserved", "", ov::RuntimeAttribute, 0);
    using PrecisionPreservedAttribute::PrecisionPreservedAttribute;
    void merge(std::vector<ov::Any>& attributes);
    std::string to_string() const override;
};

} // namespace ngraph
