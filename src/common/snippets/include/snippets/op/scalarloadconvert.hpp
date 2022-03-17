// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "loadconvert.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface ScalarLoad
 * @brief Generated for load and convert a scalar value at the same time
 * @ingroup snippets
 */
class ScalarLoadConvert : public LoadConvert {
public:
    OPENVINO_OP("ScalarLoadConvert", "SnippetsOpset", ngraph::snippets::op::LoadConvert);

    ScalarLoadConvert(const Output<Node>& x, const ov::element::Type& destination_type);
    ScalarLoadConvert() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarLoadConvert>(new_args.at(0), m_destination_type);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
