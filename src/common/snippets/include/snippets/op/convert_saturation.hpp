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
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {

/**
 * @interface ConvertSaturation
 * @brief It's a ordinary Convert op with specific rules for integer conversion.
 *        The implementation uses "saturation" conversion for integer values.
 *        It means that if the integer values are outside the limits
 *        of the maximum and minimum values of the destination data type, they are clamped.
 *        For example, int_32t ---> int8_t
 *                       129   --->  127
 * @ingroup snippets
 */
class SNIPPETS_API ConvertSaturation : public ov::op::v0::Convert {
public:
    OPENVINO_OP("ConvertSaturation", "SnippetsOpset", ov::op::v0::Convert);

    ConvertSaturation(const Output<Node>& x, const ov::element::Type& destination_type);
    ConvertSaturation() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override {
        return false;
    }
};

}  // namespace ov::snippets::op
