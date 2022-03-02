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
 * @interface ConvertTruncation
 * @brief The implementation doesn't "saturation" conversion.
 *        It means that if there are overflow, the values will wrap around.
 *        For example, int_32t ---> int8_t
 *                       129   --->  -127
 *        Note: It is covered by specification of "Convert" op
 *              This op is used for real Convert ops inside subgraph body in CPU Plugin
 * @ingroup snippets
 */
class ConvertTruncation : public ov::op::v0::Convert {
public:
    OPENVINO_OP("ConvertTruncation", "SnippetsOpset", ov::op::v0::Convert);

    ConvertTruncation(const Output<Node>& x, const ov::element::Type& destination_type);
    ConvertTruncation() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
