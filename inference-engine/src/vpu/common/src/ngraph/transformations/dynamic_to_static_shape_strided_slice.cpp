// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_strided_slice.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <algorithm>
#include <memory>
#include <numeric>

namespace vpu {

ngraph::AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) {
    ngraph::AxisSet axis_set{};
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] == 1)
            axis_set.emplace(i);
    return axis_set;
}

std::shared_ptr<ngraph::Node> calculate_output_shape(
        const std::vector<int64_t> & begin,
        const std::vector<int64_t> & end,
        const std::vector<int64_t> & strides,
        const ngraph::AxisSet & begin_mask,
        const ngraph::AxisSet & end_mask,
        const ngraph::Output<ngraph::Node> & input_shape) {
    const auto& shape_type = input_shape.get_element_type();

    VPU_THROW_UNLESS(begin.size() == end.size() && begin.size() == strides.size(),
        "Begin, end and strides inputs must be of the same size, but {}, {} and {} given accordingly", begin.size(), end.size(), strides.size());
    const auto inputShapeRank = input_shape.get_partial_shape()[0].get_length();
    VPU_THROW_UNLESS(inputShapeRank >= begin.size(),
        "Input shape rank must not be less than begin/end/strides size, but {} and {} given accordingly", inputShapeRank, begin.size());

    ngraph::OutputVector output_dimensions;
    for (int64_t axis = 0; axis < begin.size(); ++axis) {
        auto lb = begin[axis], ub = end[axis], stride = strides[axis];

        ngraph::Output<ngraph::Node> lower_bound = ngraph::opset3::Constant::create(shape_type, {1}, {lb});
        ngraph::Output<ngraph::Node> upper_bound = ngraph::opset3::Constant::create(shape_type, {1}, {ub});

        const auto shape_dim = std::make_shared<ngraph::opset3::Gather>(input_shape, ngraph::opset3::Constant::create(shape_type, {1}, {axis}),
                ngraph::opset3::Constant::create(shape_type, {}, {0}));
        const auto shape_dim_minus_one = std::make_shared<ngraph::opset3::Add>(shape_dim, ngraph::opset3::Constant::create(shape_type, {1}, {-1}));


        // convert negative indexes to positive
        // take max for this case: if abs(lb) > input_shape[input_shape_idx],then after
        // conversion lb < 0
        // so according to tensorflow and numpy we just get 0
        if (lb < 0)
            lower_bound = std::make_shared<ngraph::opset3::Maximum>(
                    std::make_shared<ngraph::opset3::Add>(lower_bound, shape_dim),
                    ngraph::opset3::Constant::create(shape_type, {1}, {0}));
        if (ub < 0)
            upper_bound = std::make_shared<ngraph::opset3::Maximum>(
                    std::make_shared<ngraph::opset3::Add>(upper_bound, shape_dim),
                    ngraph::opset3::Constant::create(shape_type, {1}, {0}));

        // apply restrictions when begin or end values more than max possible values.
        lower_bound = std::make_shared<ngraph::opset3::Minimum>(shape_dim, lower_bound);
        upper_bound = std::make_shared<ngraph::opset3::Minimum>(shape_dim, upper_bound);

        ngraph::Output<ngraph::Node> output_dimension = ngraph::opset3::Constant::create(shape_type, {1}, {0});
        if (stride < 0) {
           // apply masks
           if (begin_mask.count(axis))
               lower_bound = shape_dim_minus_one;
           if (end_mask.count(axis))
               upper_bound = ngraph::opset3::Constant::create(shape_type, {1}, {-1});

           lower_bound = std::make_shared<ngraph::opset3::Minimum>(lower_bound, shape_dim_minus_one);
           // we always get 1st element, so we need decrease range
           lower_bound = std::make_shared<ngraph::opset3::Add>(lower_bound, ngraph::opset3::Constant::create(shape_type, {1}, {-1}));
           output_dimension = std::make_shared<ngraph::opset3::Select>(
                   std::make_shared<ngraph::opset3::LessEqual>(upper_bound, lower_bound),
                   std::make_shared<ngraph::opset3::Add>(
                           std::make_shared<ngraph::opset3::Divide>(std::make_shared<ngraph::opset3::Subtract>(upper_bound, lower_bound),
                                                                    ngraph::opset3::Constant::create(shape_type, {1}, {stride})),
                           ngraph::opset3::Constant::create(shape_type, {1}, {1})),
                   output_dimension);
        } else {
            // apply masks
            if (begin_mask.count(axis))
                lower_bound = ngraph::opset3::Constant::create(shape_type, {1}, {0});
            if (end_mask.count(axis))
                upper_bound = shape_dim;
            // we always get 1st element, so we need decrease range
            lower_bound = std::make_shared<ngraph::opset3::Add>(lower_bound, ngraph::opset3::Constant::create(shape_type, {1}, {1}));
            output_dimension = std::make_shared<ngraph::opset3::Select>(
                    std::make_shared<ngraph::opset3::GreaterEqual>(upper_bound, lower_bound),
                    std::make_shared<ngraph::opset3::Add>(
                            std::make_shared<ngraph::opset3::Divide>(std::make_shared<ngraph::opset3::Subtract>(upper_bound, lower_bound),
                                                                     ngraph::opset3::Constant::create(shape_type, {1}, {stride})),
                            ngraph::opset3::Constant::create(shape_type, {1}, {1})),
                    output_dimension);
        }
        output_dimensions.push_back(output_dimension);
    }

    if (output_dimensions.size() < inputShapeRank) {
        std::vector<std::int64_t> indices(inputShapeRank - output_dimensions.size());
        std::iota(indices.begin(), indices.end(), static_cast<std::int64_t>(output_dimensions.size()));

        const auto tail = std::make_shared<ngraph::opset3::Gather>(
            input_shape,
            ngraph::opset3::Constant::create(ngraph::element::i64, {indices.size()}, indices),
            ngraph::opset3::Constant::create(shape_type, {}, {0}));
        output_dimensions.push_back(tail);
    }

    VPU_THROW_UNLESS(output_dimensions.size() == inputShapeRank,
        "output shape rank {} must be equal to input shape rank {} for DTS of StridedSlice",
        output_dimensions.size(), inputShapeRank);

    const auto output_shape = std::make_shared<ngraph::opset3::Concat>(output_dimensions, 0);
    return output_shape;
}

void dynamicToStaticShapeStridedSlice(std::shared_ptr<ngraph::Node> target) {
    const auto dsr = target->input_value(0).get_node_shared_ptr();
    VPU_THROW_UNLESS(ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr),
        "DynamicToStaticShape transformation for {} of type {} expects {} as input with index {}",
        target->get_friendly_name(), target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info, 0);

    const auto stridedSlice = ngraph::as_type_ptr<ngraph::opset3::StridedSlice>(target);
    VPU_THROW_UNLESS(stridedSlice, "dynamicToStaticShapeStridedSlice transformation is not applicable for {}", target);

    const auto all_zero = [](const std::vector<int64_t> & v) {return std::all_of(v.cbegin(), v.cend(), [](const int64_t & i){return i == 0;});};
    VPU_THROW_UNLESS(all_zero(stridedSlice->get_new_axis_mask()),
            "dynamicToStaticShapeStridedSlice transformation is not applicable for {}, new_axis_mask expected to be zeros", target);
    VPU_THROW_UNLESS(all_zero(stridedSlice->get_shrink_axis_mask()),
                "dynamicToStaticShapeStridedSlice transformation is not applicable for {}, shrink_axis_mask expected to be zeros", target);
    VPU_THROW_UNLESS(all_zero(stridedSlice->get_ellipsis_mask()),
                "dynamicToStaticShapeStridedSlice transformation is not applicable for {}, ellipsis_mask expected to be zeros", target);

    const auto get_i64_vector_from_const = [&stridedSlice](std::shared_ptr<ngraph::Node> node_ptr) {
        const auto constant = ngraph::as_type_ptr<ngraph::opset3::Constant>(node_ptr);
        VPU_THROW_UNLESS(constant,
                "dynamicToStaticShapeStridedSlice transformation is not applicable for {}, begin, end and stride inputs are expected to be constants",
                stridedSlice);
        return constant->cast_vector<int64_t>();
    };

    const auto input_shape = dsr->input_value(1);
    const auto output_shape = calculate_output_shape(
            get_i64_vector_from_const(stridedSlice->input_value(1).get_node_shared_ptr()),
            get_i64_vector_from_const(stridedSlice->input_value(2).get_node_shared_ptr()),
            get_i64_vector_from_const(stridedSlice->input_value(3).get_node_shared_ptr()),
            convert_mask_to_axis_set(stridedSlice->get_begin_mask()),
            convert_mask_to_axis_set(stridedSlice->get_end_mask()),
            input_shape);

    const auto copied = stridedSlice->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, output_shape);
    outDSR->set_friendly_name(stridedSlice->get_friendly_name());
    ngraph::replace_node(std::move(target), std::move(outDSR));
}

}  // namespace vpu
