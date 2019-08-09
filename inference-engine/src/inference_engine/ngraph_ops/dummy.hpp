// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/op/op.hpp>


namespace ngraph {
namespace op {

class Dummy : public Op {
public:
    Dummy();

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
};

}  // namespace op
}  // namespace ngraph

