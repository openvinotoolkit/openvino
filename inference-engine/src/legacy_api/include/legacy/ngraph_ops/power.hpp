// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(PowerIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"PowerIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    PowerIE(const Output<Node>& data_batch,
            const float power, const float scale, const float shift);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float scale, power, shift;
};

}  // namespace op
}  // namespace ngraph
