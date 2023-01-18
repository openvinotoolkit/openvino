// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_reduceop_path(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API bool is_reduceop_path(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief ReduceOpPath class represents runtime info attribute that marks path that goes
 * into input of ReduceSum and ReduceMean. It is used to mark paths that goes from Exp to ReduceSum and ReduceMean
 */
class TRANSFORMATIONS_API ReduceOpPath : public RuntimeAttribute {
public:
    OPENVINO_RTTI("reduceop_path", "0");

    ReduceOpPath() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
