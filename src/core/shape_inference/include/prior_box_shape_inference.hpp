// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/prior_box.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace prior_box {

template <class TOp, class TShape>
std::vector<TShape> shape_infer(const TOp* const op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    auto out_size_rank = input_shapes[0].rank();
    auto img_size_rank = input_shapes[1].rank();
    NODE_VALIDATION_CHECK(op,
                          out_size_rank.compatible(img_size_rank) && out_size_rank.compatible(1),
                          "output size input rank ",
                          out_size_rank,
                          " must match image shape input rank ",
                          img_size_rank,
                          " and both must be 1-D");

    auto output_shapes = std::vector<TShape>(1, TShape{2});

    if (auto out_size = get_input_const_data_as_shape<TShape>(op, 0, constant_data)) {
        NODE_VALIDATION_CHECK(op, out_size->size() == 2, "Output size must have two elements. Got: ", out_size->size());

        using TDim = typename TShape::value_type;
        const auto num_of_priors = TOp::number_of_priors(op->get_attrs());
        output_shapes.front().push_back((*out_size)[0] * (*out_size)[1] * TDim(num_of_priors) * 4);
    } else {
        output_shapes.front().emplace_back(ov::util::dim::inf_bound);
    }

    return output_shapes;
}
}  // namespace prior_box

namespace v0 {
template <class TShape>
std::vector<TShape> shape_infer(const PriorBox* const op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    return prior_box::shape_infer(op, input_shapes, constant_data);
}
}  // namespace v0

namespace v8 {
template <class TShape>
std::vector<TShape> shape_infer(const PriorBox* const op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    return prior_box::shape_infer(op, input_shapes, constant_data);
}

template <class TShape>
void shape_infer(const PriorBox* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = prior_box::shape_infer(op, input_shapes, constant_data);
}
}  // namespace v8

}  // namespace op
}  // namespace ov
