// Copyright (C) 2018-2026 Intel Corporation
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
 * @ingroup ov_runtime_attr_api
 * @brief DisableFP16Compression class represents runtime info attribute that marks operation
 * as prohibited to convert to lower precision (e.g. to FP16) and they should be inferred precisely in the original
 * precision.
 */
class TRANSFORMATIONS_API DisableFP16Compression : public RuntimeAttribute {
public:
    OPENVINO_RTTI("precise", "0", RuntimeAttribute);

    DisableFP16Compression() = default;

    bool visit_attributes(AttributeVisitor& visitor) override {
        return true;
    }

    bool is_copyable() const override {
        return false;
    }
};

TRANSFORMATIONS_API void disable_compression_to(const std::shared_ptr<Node>& node, element::Type to);

TRANSFORMATIONS_API void disable_compression_from_to(const std::shared_ptr<Node>& node,
                                                     element::Type from,
                                                     element::Type to);

TRANSFORMATIONS_API void enable_compression_to(const std::shared_ptr<Node>& node, element::Type to);

TRANSFORMATIONS_API void enable_compression_from_to(const std::shared_ptr<Node>& node,
                                                    element::Type from,
                                                    element::Type to);

TRANSFORMATIONS_API bool is_compression_disabled_to(const std::shared_ptr<Node>& node, element::Type to);

TRANSFORMATIONS_API bool is_compression_disabled_from_to(const std::shared_ptr<Node>& node,
                                                         element::Type from,
                                                         element::Type to);

class TRANSFORMATIONS_API DisablePrecisionConversion : public RuntimeAttribute {
public:
    OPENVINO_RTTI("DisablePrecisionConversion", "0", RuntimeAttribute);

    explicit DisablePrecisionConversion(element::Type from, element::Type to) {
        m_disabled_precisions[from].insert(to);
    }

    bool is_copyable() const override {
        return false;
    }

    std::map<ov::element::Type, std::set<ov::element::Type>> m_disabled_precisions = {
        {element::dynamic, {}}};  // TODO: fix encapsulation

    // fp32 -> {i16, f16, ov::dynamic}
};

}  // namespace ov
