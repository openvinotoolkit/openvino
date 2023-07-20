// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void enable_keep_fp16_const(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void disable_keep_fp16_const(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_keep_fp16_const(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief DisableFP16Compression class represents runtime info attribute that marks operation
 * as prohibitted to convert to FP16 as part of Compressed Only format.
 */
class TRANSFORMATIONS_API KeepFP16Const : public RuntimeAttribute {
public:
    OPENVINO_RTTI("keep_fp16_const", "0");

    KeepFP16Const() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
