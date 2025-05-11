// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/lpt_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
LP_TRANSFORMATIONS_API void mark_as_bias(const std::shared_ptr<Node>& node);

LP_TRANSFORMATIONS_API bool marked_as_bias(const std::shared_ptr<const Node>& node);

class LP_TRANSFORMATIONS_API BiasAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::Bias", "", ov::RuntimeAttribute);
    bool is_copyable(const std::shared_ptr<Node>& to) const override;
};
} // namespace ov
