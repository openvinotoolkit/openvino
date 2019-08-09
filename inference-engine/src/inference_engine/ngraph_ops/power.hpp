// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"


namespace ngraph {
namespace op {

class PowerIE : public Op {
public:
    PowerIE(const std::shared_ptr<Node>& data_batch,
            const float power, const float scale, const float shift);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    float scale, power, shift;
};

}  // namespace op
}  // namespace ngraph
