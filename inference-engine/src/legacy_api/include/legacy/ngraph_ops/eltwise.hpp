// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

enum class ELTWISE_TYPE {Sum, Prod, Max, Sub, Min, Div};

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(Eltwise) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"Eltwise", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    Eltwise(const Output<Node>& data1,
            const Output<Node>& data2,
            const ELTWISE_TYPE eltwise_type);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    ELTWISE_TYPE eltwise_type;
};

}  // namespace op
}  // namespace ngraph
