// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/visibility.hpp"


namespace ov {

OPENVINO_API void mark_as_decompression(const std::shared_ptr<Node>& node);

OPENVINO_API void unmark_as_decompression(const std::shared_ptr<Node>& node);

OPENVINO_API bool is_decompression(const std::shared_ptr<Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief Decompression class represents runtime info attribute that marks operation
 * as used as decompression for Compressed Only format.
 */
class OPENVINO_API Decompression : public RuntimeAttribute {
public:
    OPENVINO_RTTI("decompression", "0");

    Decompression() = default;

    bool visit_attributes(AttributeVisitor& visitor) override { return true; }

    bool is_copyable() const override { return false; }
};

}  // namespace ov
