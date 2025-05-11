// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <functional>
#include <memory>
#include <set>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_as_decompression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void unmark_as_decompression(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_decompression(const std::shared_ptr<Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief Decompression class represents runtime info attribute that marks operation
 * as used as decompression for Compressed Only format.
 */
class TRANSFORMATIONS_API Decompression : public RuntimeAttribute {
public:
    OPENVINO_RTTI("decompression", "0", RuntimeAttribute);

    Decompression() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
