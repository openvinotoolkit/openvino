// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <openvino/core/visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/layer_transformation.hpp"
#include "attribute_parameters.hpp"

namespace ngraph {
class OPENVINO_API PerTensorQuantizationAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::PerTensorQuantization", "", ov::RuntimeAttribute, 0);
    ~PerTensorQuantizationAttribute();
};
} // namespace ngraph
