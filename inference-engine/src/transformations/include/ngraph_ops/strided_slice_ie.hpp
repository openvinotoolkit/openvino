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

class INFERENCE_ENGINE_API_CLASS(StridedSliceIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"StridedSliceIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    StridedSliceIE(const Output<Node>& data,
                 const Output<Node>& begin,
                 const Output<Node>& end,
                 const Output<Node>& strides,
                 const std::vector<int64_t>& begin_mask,
                 const std::vector<int64_t>& end_mask,
                 const std::vector<int64_t>& new_axis_mask,
                 const std::vector<int64_t>& shrink_axis_mask,
                 const std::vector<int64_t>& ellipsis_mask,
                 const Shape& output_shape);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    const std::vector<int64_t>& get_begin_mask() const { return m_begin_mask; }
    const std::vector<int64_t>& get_end_mask() const { return m_end_mask; }
    const std::vector<int64_t>& get_new_axis_mask() const { return m_new_axis_mask; }
    const std::vector<int64_t>& get_shrink_axis_mask() const { return m_shrink_axis_mask; }
    const std::vector<int64_t>& get_ellipsis_mask() const { return m_ellipsis_mask; }

protected:
    const std::vector<int64_t> m_begin_mask;
    const std::vector<int64_t> m_end_mask;
    const std::vector<int64_t> m_new_axis_mask;
    const std::vector<int64_t> m_shrink_axis_mask;
    const std::vector<int64_t> m_ellipsis_mask;
    Shape m_output_shape;
};

}  // namespace op
}  // namespace ngraph
