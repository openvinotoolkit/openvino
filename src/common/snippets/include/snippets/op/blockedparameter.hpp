// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include <ngraph/op/parameter.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface BlockedParameter
 * @brief Represents blocked input (NCHW<X>c) for a subgraph
 * @ingroup snippets
 */
class BlockedParameter : public ngraph::op::Parameter {
public:
    OPENVINO_OP("BlockedParameter", "SnippetsOpset", ngraph::op::Parameter);

    BlockedParameter() = default;
    BlockedParameter(const ngraph::element::Type& element_type, const PartialShape& pshape)
        : Parameter(element_type, pshape) {
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<BlockedParameter>(m_element_type, m_partial_shape);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph