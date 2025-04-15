// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void set_original_precision_attribute(const std::shared_ptr<Node>& node,
                                                          const element::Type_t original_precision);
TRANSFORMATIONS_API void reset_original_precision_attribute(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API element::Type_t get_original_precision(const std::shared_ptr<Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief OriginalPrecisionAttribute stores the original precision of the node to pass this information to
 * plugins.
 */
class TRANSFORMATIONS_API OriginalPrecisionAttribute : public RuntimeAttribute {
public:
    OPENVINO_RTTI("original_precision", "0", RuntimeAttribute);
};

}  // namespace ov
