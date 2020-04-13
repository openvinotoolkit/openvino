// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(GatherIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"GatherIE", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    GatherIE() = default;

    GatherIE(const Output<Node>& params, const Output<Node>& indices, int64_t axis, const Shape & output_shape);

    void validate_and_infer_types() override;

    int64_t get_axis() const { return m_axis; }
    void set_axis(int64_t axis) { m_axis = axis; }
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

protected:
    int64_t m_axis;
    Shape m_output_shape;
};

}  // namespace op
}  // namespace ngraph
