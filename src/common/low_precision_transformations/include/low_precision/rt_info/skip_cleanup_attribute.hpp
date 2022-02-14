// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>

#include "low_precision/rt_info/attribute_parameters.hpp"

namespace ngraph {
class LP_TRANSFORMATIONS_API SkipCleanupAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::SkipCleanup", "", ov::RuntimeAttribute, 0);
    static ov::Any create(const std::shared_ptr<ngraph::Node>& node);
};
} // namespace ngraph
