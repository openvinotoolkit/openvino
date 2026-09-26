// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/dynamic_same_padding_fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {
using namespace ov;

// A single element of a small shape tensor. Follow only operations that rearrange
// elements; arithmetic is checked separately, without evaluating sample input sizes.
struct Scalar {
    Output<Node> value;
    size_t index = 0;
};

size_t small_tensor_size(const Output<Node>& value) {
    const auto& shape = value.get_partial_shape();
    if (shape.is_dynamic())
        return 0;
    size_t count = 1;
    for (const auto& dim : shape) {
        const auto length = static_cast<size_t>(dim.get_length());
        if (length == 0 || length > 64 / count)
            return 0;
        count *= length;
    }
    return count;
}

bool resolve(Scalar& scalar) {
    for (size_t depth = 0; depth < 64; ++depth) {
        if (scalar.index >= small_tensor_size(scalar.value))
            return false;
        const auto node = scalar.value.get_node_shared_ptr();
        if (is_type<op::v1::Reshape>(node) || is_type<op::v0::Squeeze>(node) || is_type<op::v0::Unsqueeze>(node)) {
            if (small_tensor_size(node->input_value(0)) != small_tensor_size(scalar.value))
                return false;
            scalar.value = node->input_value(0);
        } else if (const auto concat = as_type_ptr<op::v0::Concat>(node)) {
            if (scalar.value.get_shape().size() != 1 || (concat->get_axis() != 0 && concat->get_axis() != -1))
                return false;
            bool found = false;
            for (const auto& input : concat->input_values()) {
                const auto size = small_tensor_size(input);
                if (scalar.index < size) {
                    scalar.value = input;
                    found = true;
                    break;
                }
                scalar.index -= size;
            }
            if (!found)
                return false;
        } else if (const auto gather = as_type_ptr<op::util::GatherBase>(node)) {
            const auto indices = as_type_ptr<op::v0::Constant>(gather->input_value(1).get_node_shared_ptr());
            const auto axis = as_type_ptr<op::v0::Constant>(gather->input_value(2).get_node_shared_ptr());
            const auto& input = gather->input_value(0);
            const auto size = small_tensor_size(input);
            if (!indices || !axis || shape_size(axis->get_shape()) != 1 || size == 0 || input.get_shape().size() != 1 ||
                gather->get_batch_dims() != 0)
                return false;
            const auto axis_value = axis->cast_vector<int64_t>()[0];
            if (axis_value != 0 && axis_value != -1)
                return false;
            const auto values = indices->cast_vector<int64_t>();
            if (scalar.index >= values.size())
                return false;
            auto index = values[scalar.index];
            if (index < 0)
                index += static_cast<int64_t>(size);
            if (index < 0 || static_cast<size_t>(index) >= size)
                return false;
            scalar = {input, static_cast<size_t>(index)};
        } else if (const auto split = as_type_ptr<op::v1::Split>(node)) {
            const auto axis = as_type_ptr<op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
            const auto& input = split->input_value(0);
            if (!axis || shape_size(axis->get_shape()) != 1 || small_tensor_size(input) == 0)
                return false;
            const auto shape = input.get_shape();
            auto axis_value = axis->cast_vector<int64_t>()[0];
            if (axis_value < 0)
                axis_value += static_cast<int64_t>(shape.size());
            if (axis_value < 0 || static_cast<size_t>(axis_value) >= shape.size())
                return false;
            const auto inner =
                std::accumulate(shape.begin() + axis_value + 1, shape.end(), size_t{1}, std::multiplies<size_t>());
            const auto chunk = scalar.value.get_shape()[axis_value] * inner;
            scalar.index = scalar.index / chunk * shape[axis_value] * inner + scalar.value.get_index() * chunk +
                           scalar.index % chunk;
            scalar.value = input;
        } else {
            return true;
        }
    }
    return false;
}

bool constant(Scalar scalar, double& value) {
    if (!resolve(scalar))
        return false;
    const auto c = as_type_ptr<op::v0::Constant>(scalar.value.get_node_shared_ptr());
    if (!c)
        return false;
    value = c->cast_vector<double>()[scalar.index];
    return std::isfinite(value);
}

bool constant_is(const Scalar& scalar, double expected) {
    double value;
    return constant(scalar, value) && value == expected;
}

bool input_scalar(const Scalar& scalar, size_t port, Scalar& input) {
    const auto& value = scalar.value.get_node_shared_ptr()->input_value(port);
    const auto size = small_tensor_size(value);
    if (size == 0 || (size != 1 && value.get_shape() != scalar.value.get_shape()))
        return false;
    input = {value, size == 1 ? 0 : scalar.index};
    return resolve(input);
}

bool is_shape_float(const element::Type& type) {
    return type == element::f32 || type == element::f64;
}

bool unwrap_integer_convert(Scalar& scalar) {
    if (!resolve(scalar))
        return false;
    if (const auto convert = as_type_ptr<op::v0::Convert>(scalar.value.get_node_shared_ptr())) {
        const auto type = convert->get_destination_type();
        if ((type != element::i32 && type != element::i64) || !is_shape_float(convert->get_input_element_type(0)))
            return false;
        return input_scalar(scalar, 0, scalar);
    }
    return true;
}

bool is_dimension(Scalar scalar, const Output<Node>& data, size_t axis) {
    if (!resolve(scalar))
        return false;
    if (const auto convert = as_type_ptr<op::v0::Convert>(scalar.value.get_node_shared_ptr())) {
        if (!is_shape_float(convert->get_destination_type()) ||
            (convert->get_input_element_type(0) != element::i32 &&
             convert->get_input_element_type(0) != element::i64) ||
            !input_scalar(scalar, 0, scalar))
            return false;
    }
    const auto shape = as_type_ptr<op::util::ShapeOfBase>(scalar.value.get_node_shared_ptr());
    return shape && shape->input_value(0) == data && scalar.index == axis;
}

bool divided_by(Scalar scalar, double divisor, Scalar& numerator) {
    if (!resolve(scalar) || !is_shape_float(scalar.value.get_element_type()))
        return false;
    Scalar a, b;
    const auto node = scalar.value.get_node_shared_ptr();
    if ((!is_type<op::v1::Divide>(node) && !is_type<op::v1::Multiply>(node)) || !input_scalar(scalar, 0, a) ||
        !input_scalar(scalar, 1, b))
        return false;
    if (is_type<op::v1::Divide>(node)) {
        if (!constant_is(b, divisor))
            return false;
        numerator = a;
        return true;
    }
    // Only accept an exact reciprocal (in particular, division by two in the
    // padding split). Do not approximate a precision-sensitive shape division.
    int exponent;
    if (std::frexp(divisor, &exponent) != 0.5)
        return false;
    if (constant_is(a, 1.0 / divisor)) {
        numerator = b;
        return true;
    }
    if (constant_is(b, 1.0 / divisor)) {
        numerator = a;
        return true;
    }
    return false;
}

using Coefficients = std::array<double, 3>;
using Atom = std::function<bool(const Scalar&, Coefficients&)>;

// Recognize affine combinations of two explicitly checked atoms and a constant.
// This handles both (ceil(I/S)-1)*S+Keff-I and constant-folded/reassociated exports.
bool affine(Scalar scalar, const Atom& atom, Coefficients& result, size_t& budget) {
    if (budget == 0 || !resolve(scalar))
        return false;
    --budget;
    if (atom(scalar, result))
        return true;
    double value;
    if (constant(scalar, value)) {
        result = {0, 0, value};
        return true;
    }
    Scalar a, b;
    const auto node = scalar.value.get_node_shared_ptr();
    if ((!is_type<op::v1::Add>(node) && !is_type<op::v1::Subtract>(node) && !is_type<op::v1::Multiply>(node)) ||
        !input_scalar(scalar, 0, a) || !input_scalar(scalar, 1, b))
        return false;
    if (is_type<op::v1::Multiply>(node)) {
        if (constant(a, value))
            std::swap(a, b);
        else if (!constant(b, value))
            return false;
        if (!affine(a, atom, result, budget))
            return false;
        for (auto& coefficient : result)
            coefficient *= value;
    } else {
        Coefficients lhs, rhs;
        if (!affine(a, atom, lhs, budget) || !affine(b, atom, rhs, budget))
            return false;
        const auto sign = is_type<op::v1::Add>(node) ? 1 : -1;
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = lhs[i] + sign * rhs[i];
    }
    return true;
}

bool matches_affine(const Scalar& scalar, const Atom& atom, const Coefficients& expected) {
    size_t budget = 128;
    Coefficients result;
    return affine(scalar, atom, result, budget) && result == expected;
}

class SamePadding {
public:
    SamePadding(const Output<Node>& data, size_t axis, size_t stride, size_t effective_kernel)
        : m_data(data),
          m_axis(axis),
          m_stride(static_cast<double>(stride)),
          m_kernel(static_cast<double>(effective_kernel)) {}

    bool begin(Scalar scalar) const {
        return unwrap_integer_convert(scalar) && half(scalar);
    }

    bool end(Scalar scalar) const {
        if (!unwrap_integer_convert(scalar))
            return false;
        return matches_affine(scalar,
                              [this](const Scalar& s, Coefficients& c) {
                                  if (total(s)) {
                                      c = {1, 0, 0};
                                      return true;
                                  }
                                  if (half(s)) {
                                      c = {0, 1, 0};
                                      return true;
                                  }
                                  return false;
                              },
                              {1, -1, 0});
    }

private:
    bool total(Scalar scalar) const {
        if (!resolve(scalar) || !is_type<op::v1::Maximum>(scalar.value.get_node()))
            return false;
        Scalar a, b;
        if (!input_scalar(scalar, 0, a) || !input_scalar(scalar, 1, b))
            return false;
        if (constant_is(a, 0))
            std::swap(a, b);
        if (!constant_is(b, 0))
            return false;
        return matches_affine(a,
                              [this](Scalar s, Coefficients& c) {
                                  if (is_dimension(s, m_data, m_axis)) {
                                      c = {1, 0, 0};
                                      return true;
                                  }
                                  Scalar numerator;
                                  if (is_type<op::v0::Ceiling>(s.value.get_node()) && input_scalar(s, 0, s) &&
                                      divided_by(s, m_stride, numerator) && is_dimension(numerator, m_data, m_axis)) {
                                      c = {0, 1, 0};
                                      return true;
                                  }
                                  return false;
                              },
                              {-1, m_stride, m_kernel - m_stride});
    }

    bool half(Scalar scalar) const {
        Scalar numerator;
        return resolve(scalar) && is_type<op::v0::Floor>(scalar.value.get_node()) && input_scalar(scalar, 0, scalar) &&
               divided_by(scalar, 2, numerator) && total(numerator);
    }

    Output<Node> m_data;
    size_t m_axis;
    double m_stride;
    double m_kernel;
};
}  // namespace

ov::pass::DynamicSamePaddingFusion::DynamicSamePaddingFusion() {
    MATCHER_SCOPE(DynamicSamePaddingFusion);
    const auto root_pattern = pattern::wrap_type<op::v1::Convolution, op::v1::GroupConvolution>();
    matcher_pass_callback callback = [this](pattern::Matcher& matcher) {
        const auto conv = as_type_ptr<op::util::ConvolutionFwdPropBase>(matcher.get_match_root());
        if (transformation_callback(conv))
            return false;
        const auto pad = as_type_ptr<op::util::PadBase>(conv->input_value(0).get_node_shared_ptr());
        if (!pad || pad->get_pad_mode() != op::PadMode::CONSTANT ||
            (pad->get_input_size() == 4 && !constant_is({pad->input_value(3)}, 0)))
            return false;
        if (conv->get_auto_pad() != op::PadType::EXPLICIT && conv->get_auto_pad() != op::PadType::VALID)
            return false;
        const auto is_zero = [](ptrdiff_t value) {
            return value == 0;
        };
        if (!std::all_of(conv->get_pads_begin().begin(), conv->get_pads_begin().end(), is_zero) ||
            !std::all_of(conv->get_pads_end().begin(), conv->get_pads_end().end(), is_zero))
            return false;
        const auto& data = pad->input_value(0);
        const auto& shape = data.get_partial_shape();
        const auto& weights = conv->get_input_partial_shape(1);
        if (shape.rank().is_dynamic() || weights.rank().is_dynamic())
            return false;
        const auto rank = static_cast<size_t>(shape.rank().get_length());
        const auto offset = is_type<op::v1::GroupConvolution>(conv) ? 3u : 2u;
        if (rank < 3 || rank > 5 || weights.size() != rank + offset - 2 || conv->get_strides().size() != rank - 2 ||
            conv->get_dilations().size() != rank - 2 ||
            pad->get_input_partial_shape(1) != PartialShape{static_cast<int64_t>(rank)} ||
            pad->get_input_partial_shape(2) != PartialShape{static_cast<int64_t>(rank)})
            return false;
        for (size_t axis = 0; axis < rank; ++axis) {
            const Scalar begin{pad->input_value(1), axis};
            const Scalar end{pad->input_value(2), axis};
            if (axis < 2) {
                if (!constant_is(begin, 0) || !constant_is(end, 0))
                    return false;
                continue;
            }
            const auto& kernel = weights[axis - 2 + offset];
            if (kernel.is_dynamic() || kernel.get_length() <= 0)
                return false;
            const auto stride = conv->get_strides()[axis - 2];
            const auto dilation = conv->get_dilations()[axis - 2];
            if (stride == 0 || dilation == 0 ||
                static_cast<size_t>(kernel.get_length()) - 1 > (std::numeric_limits<size_t>::max() - 1) / dilation)
                return false;
            const auto effective_kernel = (static_cast<size_t>(kernel.get_length()) - 1) * dilation + 1;
            const SamePadding same(data, axis, stride, effective_kernel);
            if (!same.begin(begin) || !same.end(end))
                return false;
        }
        // Clone first, so Pad users with different convolution attributes or other
        // consumers of the shape arithmetic retain their original inputs.
        const auto replacement =
            as_type_ptr<op::util::ConvolutionFwdPropBase>(conv->clone_with_new_inputs(conv->input_values()));
        replacement->set_argument(0, data);
        replacement->set_auto_pad(op::PadType::SAME_UPPER);
        replacement->validate_and_infer_types();
        replacement->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({pad, conv}, replacement);
        replace_node(conv, replacement);
        return true;
    };
    register_matcher(std::make_shared<pattern::Matcher>(root_pattern, matcher_name), callback);
}
