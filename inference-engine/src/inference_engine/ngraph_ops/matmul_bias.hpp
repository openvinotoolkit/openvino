// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/axis_set.hpp>
#include <ngraph/op/op.hpp>
#include <memory>

#ifdef DEBUG
#define IE_DEBUG(x) std::cout << x << std::endl;
#else
#define IE_DEBUG(x)
#endif


namespace ngraph {
namespace op {

class MatmulBias : public Op {
public:
    MatmulBias(std::shared_ptr<Node> W,
            std::shared_ptr<Node> x,
            std::shared_ptr<Node> b,
            Shape shape_w,
            Shape shape_x,
            bool transpose_w,
            bool transpose_x,
            AxisSet axes = AxisSet{});

    void validate_and_infer_types() override;

    bool get_is_a_transposed() const { return m_transpose_w; }
    bool get_is_b_transposed() const { return m_transpose_x; }
    Shape get_a_shape() const { return m_shape_w; }
    Shape get_b_shape() const { return m_shape_x; }
    const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

private:
    Shape m_shape_w;
    Shape m_shape_x;
    bool m_transpose_w;
    bool m_transpose_x;
    AxisSet m_broadcast_axes;
};
}  // namespace op
}  // namespace ngraph
