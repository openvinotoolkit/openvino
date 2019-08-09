// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"

enum class ELTWISE_TYPE {Sum, Prod, Max, Sub, Min, Div};

namespace ngraph {
namespace op {

class Eltwise : public Op {
public:
    Eltwise(const std::shared_ptr<Node>& data1,
            const std::shared_ptr<Node>& data2,
            const ELTWISE_TYPE eltwise_type);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    ELTWISE_TYPE eltwise_type;
};

}  // namespace op
}  // namespace ngraph
