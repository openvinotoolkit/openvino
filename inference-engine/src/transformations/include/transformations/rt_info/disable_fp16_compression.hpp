// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/variant.hpp"
#include "transformations_visibility.hpp"


namespace ov {

TRANSFORMATIONS_API void disable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void enable_fp16_compression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool fp16_compression_is_disabled(const std::shared_ptr<Node>& node);

class TRANSFORMATIONS_API DisableFP16Compression : public VariantImpl<void> {
public:
    OPENVINO_RTTI("disable_fp16_compression", "0");

    DisableFP16Compression() = default;

    bool is_copyable() const override { return false; }
};

}  // namespace ov
