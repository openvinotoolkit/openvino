// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void disable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool fp16_compression_is_disabled(const std::shared_ptr<const Node>& node);

TRANSFORMATIONS_API void postpone_fp16_compression(RTMap& rt_info);

TRANSFORMATIONS_API bool is_fp16_compression_postponed(const RTMap& rt_info);

TRANSFORMATIONS_API void do_not_postpone_fp16_compression(RTMap& rt_info);

/**
 * @ingroup ie_runtime_attr_api
 * @brief DisableFP16Compression class represents runtime info attribute that marks operation
 * as prohibitted to convert to FP16 as part of Compressed Only format.
 */
class TRANSFORMATIONS_API DisableFP16Compression : public RuntimeAttribute {
public:
    OPENVINO_RTTI("disable_fp16_compression", "0");

    DisableFP16Compression() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
