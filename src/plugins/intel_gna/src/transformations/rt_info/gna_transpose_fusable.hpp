// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace rt_info {

void add_transpose_fusable(const std::shared_ptr<Node>& node);

void remove_transpose_fusable(const std::shared_ptr<Node>& node);

bool is_transpose_fusable(const std::shared_ptr<Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief GNATransposeFusable class represents runtime info attribute that marks operation
 * as fusable with functional layer
 */
class GNATransposeFusable : public RuntimeAttribute {
public:
    OPENVINO_RTTI("gna_transpose_fusable", "0");

    GNATransposeFusable() = default;

    bool is_copyable() const override {
        return false;
    }
};
} // namespace rt_info
} // namespace intel_gna
} // namespace ov
