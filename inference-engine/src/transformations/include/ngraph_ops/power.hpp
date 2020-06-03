// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API PowerIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"PowerIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    PowerIE(const Output<Node>& data_batch,
            const float power, const float scale, const float shift);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    float scale, power, shift;
};

}  // namespace op
}  // namespace ngraph
