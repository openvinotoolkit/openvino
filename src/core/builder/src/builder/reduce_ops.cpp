// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/reduce_ops.hpp"

#include <numeric>

#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/util.hpp"

namespace ngraph {
namespace builder {
namespace {
size_t get_num_elements(const Shape& shape, const AxisSet& reduction_axes) {
    size_t N = 1;
    for (auto a : reduction_axes) {
        N *= shape[a];
    }
    return N;
}

std::shared_ptr<Node> get_num_elements(const Output<Node>& value, const Output<Node>& reduction_axes) {
    const auto value_shape = std::make_shared<ngraph::opset1::ShapeOf>(value);
    const auto dim_values =
        std::make_shared<ngraph::opset1::Gather>(value_shape,
                                                 reduction_axes,
                                                 ngraph::opset1::Constant::create(element::i64, {}, {0}));

    return std::make_shared<ngraph::opset1::ReduceProd>(dim_values,
                                                        ngraph::opset1::Constant::create(element::i64, {}, {0}));
}

}  // namespace

std::shared_ptr<Node> builder::opset1::mean(const Output<Node>& value, const AxisSet& reduction_axes, bool keep_dims) {
    std::shared_ptr<Node> elems_number;
    const auto value_elem_type = value.get_element_type();
    const auto reduction_axes_const =
        ngraph::opset1::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector());
    const auto value_elems_sum = std::make_shared<ngraph::opset1::ReduceSum>(value, reduction_axes_const, keep_dims);
    if (value.get_partial_shape().is_static()) {
        const auto elems_number_value = get_num_elements(value.get_partial_shape().to_shape(), reduction_axes);
        elems_number = ngraph::opset1::Constant::create(value_elem_type, Shape{}, {elems_number_value});
    } else {
        elems_number = get_num_elements(value, reduction_axes_const);
        elems_number = std::make_shared<ngraph::opset1::Convert>(elems_number, value_elem_type);
    }

    return std::make_shared<ngraph::opset1::Divide>(value_elems_sum, elems_number);
}

std::shared_ptr<Node> builder::opset1::mean(const Output<Node>& value,
                                            const Output<Node>& reduction_axes,
                                            bool keep_dims) {
    std::shared_ptr<Node> elems_number;
    const auto value_elem_type = value.get_element_type();
    const auto value_elems_sum = std::make_shared<ngraph::opset1::ReduceSum>(value, reduction_axes, keep_dims);
    elems_number = get_num_elements(value, reduction_axes);
    elems_number = std::make_shared<ngraph::opset1::Convert>(elems_number, value_elem_type);

    return std::make_shared<ngraph::opset1::Divide>(value_elems_sum, elems_number);
}

std::shared_ptr<Node> builder::opset1::variance(const Output<Node>& value,
                                                const AxisSet& reduction_axes,
                                                const bool bessel_correction) {
    const bool keep_dims = true;
    std::shared_ptr<Node> mu = opset1::mean(value, reduction_axes, keep_dims);

    Output<Node> diff = std::make_shared<ngraph::opset1::Subtract>(value, mu);

    diff = std::make_shared<ngraph::opset1::ReduceSum>(
        std::make_shared<ngraph::opset1::Multiply>(diff, diff),
        ngraph::opset1::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector()),
        false);

    const auto& et = value.get_element_type();
    const auto N = get_num_elements(value.get_partial_shape().to_shape(), reduction_axes);

    std::shared_ptr<Node> result;
    if (bessel_correction) {
        const auto N1const = ngraph::opset1::Constant::create(et, Shape{}, {N - 1});
        result = std::make_shared<ngraph::opset1::Divide>(diff, N1const);
    } else {
        const auto Nconst = ngraph::opset1::Constant::create(et, Shape{}, {N});
        result = std::make_shared<ngraph::opset1::Divide>(diff, Nconst);
    }
    return result;
}

std::shared_ptr<Node> builder::opset1::variance(const Output<Node>& value,
                                                const Output<Node>& reduction_axes,
                                                bool keep_dims,
                                                bool bessel_correction) {
    std::shared_ptr<Node> mu = opset1::mean(value, reduction_axes, keep_dims);

    Output<Node> diff = std::make_shared<ngraph::opset1::Subtract>(value, mu);

    diff = std::make_shared<ngraph::opset1::ReduceSum>(std::make_shared<ngraph::opset1::Multiply>(diff, diff),
                                                       reduction_axes,
                                                       keep_dims);

    const auto& et = value.get_element_type();
    auto N = get_num_elements(value, reduction_axes);
    N = std::make_shared<ngraph::opset1::Convert>(N, et);

    std::shared_ptr<Node> result;
    if (bessel_correction) {
        const auto one = std::make_shared<ngraph::opset1::Constant>(et, Shape{}, 1);
        N = std::make_shared<ngraph::opset1::Subtract>(N, one);
    }

    result = std::make_shared<ngraph::opset1::Divide>(diff, N);
    return result;
}

}  // namespace builder
}  // namespace ngraph
