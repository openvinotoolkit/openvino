// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "dimension_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace prior_box {
constexpr std::array<char const*, 2> input_names{"output size", "image"};

namespace validate {
inline std::vector<PartialShape> inputs_et(const Node* const op) {
    const auto inputs_size = op->get_input_size();
    auto input_shapes = std::vector<PartialShape>();
    input_shapes.reserve(inputs_size);

    for (size_t i = 0; i < inputs_size; ++i) {
        const auto& et = op->get_input_element_type(i);
        NODE_VALIDATION_CHECK(op,
                              et.is_integral_number(),
                              prior_box::input_names[i],
                              " input must be an integral number, but is: ",
                              et);
        input_shapes.push_back(op->get_input_partial_shape(i));
    }
    return input_shapes;
}
}  // namespace validate

template <class TDim,
          class TOp,
          typename std::enable_if<std::is_same<v0::PriorBox, TOp>::value ||
                                  std::is_same<v8::PriorBox, TOp>::value>::type* = nullptr>
TDim number_of_priors(const TOp* const op) {
    return {static_cast<typename TDim::value_type>(TOp::number_of_priors(op->get_attrs()))};
}

template <class TDim>
TDim number_of_priors(const v0::PriorBoxClustered* const op) {
    return {static_cast<typename TDim::value_type>(op->get_attrs().widths.size())};
}

template <class TOp, class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const TOp* const op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
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

    auto output_shapes = std::vector<TRShape>(1, TRShape{2});

    if (auto out_size = get_input_const_data_as_shape<TRShape>(op, 0, ta)) {
        NODE_VALIDATION_CHECK(op, out_size->size() == 2, "Output size must have two elements. Got: ", out_size->size());

        using TDim = typename TRShape::value_type;
        const auto num_of_priors = prior_box::number_of_priors<TDim>(op);
        output_shapes.front().push_back((*out_size)[0] * (*out_size)[1] * num_of_priors * 4);
    } else {
        output_shapes.front().emplace_back(ov::util::dim::inf_bound);
    }

    return output_shapes;
}
}  // namespace prior_box
}  // namespace op
}  // namespace ov
