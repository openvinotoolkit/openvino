// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/op/convert.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface ConvertTruncation
 * @brief It's a ordinary Convert op with specific rules for integer conversion.
 *        The implementation "truncation" conversion for integer values.
 *        It means that if there are overflow, the integer values will wrap around.
 *        For example, int_32t ---> int8_t
 *                       129   --->  -127
 * @ingroup snippets
 */
class ConvertTruncation : public ov::op::v0::Convert {
public:
    OPENVINO_OP("ConvertTruncation", "SnippetsOpset", ov::op::v0::Convert);

    ConvertTruncation(const Output<Node>& x, const ov::element::Type& destination_type);
    ConvertTruncation(const Output<Node>& x, const ov::element::Type& destination_type, bool use_rounding);
    ConvertTruncation() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override {
        return false;
    }
};

}  // namespace ov::snippets::op
