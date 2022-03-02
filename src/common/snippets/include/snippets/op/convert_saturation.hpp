// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/convert.hpp>
#include "ngraph/op/op.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface ConvertSaturation
 * @brief The implementation uses "saturation" conversion.
 *        It means that if the values are outside the limits
 *        of the maximum and minimum values of the data type, they are clamped.
 *        For example, int_32t ---> int8_t
 *                       129   --->  127
 *        Note: It isn't covered by specification of "Convert" op
 *              This op is used for conversion into and from FP32 after the correspoding Load
 *              and before Store to calculate in FP32 inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class ConvertSaturation : public ov::op::v0::Convert {
public:
    OPENVINO_OP("ConvertSaturation", "SnippetsOpset", ov::op::v0::Convert);

    ConvertSaturation(const Output<Node>& x, const ov::element::Type& destination_type);
    ConvertSaturation() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
