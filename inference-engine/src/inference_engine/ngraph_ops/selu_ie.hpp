// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class SeluIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"SeluIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    SeluIE(const Output<Node> & input,
           const float alpha,
           const float gamma);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    float gamma, alpha;
};

}  // namespace op
}  // namespace ngraph
