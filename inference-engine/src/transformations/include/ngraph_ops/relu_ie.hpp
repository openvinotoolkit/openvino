// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(ReLUIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"ReLUIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    ReLUIE(const Output<Node> & data, const float & negative_slope);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    float get_slope() { return m_negative_slope; }

private:
    float m_negative_slope;
};

}  // namespace op
}  // namespace ngraph
