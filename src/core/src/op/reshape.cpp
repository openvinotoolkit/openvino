// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <algorithm>
#include <vector>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "itt.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/util.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/reshape.hpp"

using namespace std;
using namespace ov;

namespace reshapeop {
namespace {

template <element::Type_t ET>
void compute_output_shape(const ov::Tensor& shape_pattern, std::vector<int64_t>& output_shape) {
    size_t output_rank;
    if (shape_pattern.get_size() != 0) {
        output_rank = shape_pattern.get_shape().empty() ? 0 : shape_pattern.get_shape()[0];
    } else {
        // Can be dynamic during shape infer as conversion result from empty ov::Tensor
        output_rank = 0;
    }

    for (size_t i = 0; i < output_rank; i++) {
        output_shape.push_back(shape_pattern.data<typename ov::element_type_traits<ET>::value_type>()[i]);
    }
}
}  // namespace
}  // namespace reshapeop

op::v1::Reshape::Reshape(const Output<Node>& arg, const Output<Node>& shape_pattern, bool zero_flag)
    : Op({arg, shape_pattern}),
      m_special_zero(zero_flag) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

bool op::v1::Reshape::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Reshape_visit_attributes);
    visitor.on_attribute("special_zero", m_special_zero);
    return true;
}
void op::v1::Reshape::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Reshape_validate_and_infer_types);
    auto shape_pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(this,
                          shape_pattern_et.is_integral_number(),
                          "PartialShape pattern must be an integral number.");

    // check shapes
    const ov::PartialShape& input_pshape = get_input_partial_shape(0);
    const ov::PartialShape& shape_pattern_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          shape_pattern_shape.rank().compatible(1) ||
                              (shape_pattern_shape.rank().is_static() && shape_pattern_shape.rank().get_length() == 0),
                          "Pattern shape must have rank 1 or be empty, got ",
                          shape_pattern_shape.rank(),
                          ".");
    Rank output_rank = shape_pattern_shape.rank().is_dynamic()
                           ? Rank::dynamic()
                           : shape_pattern_shape.rank().get_length() == 0 ? 0 : shape_pattern_shape[0];
    set_output_type(0, get_input_element_type(0), ov::PartialShape::dynamic(output_rank));
    set_input_is_relevant_to_shape(1);

    std::vector<Dimension> reshape_pattern;
    bool shape_can_be_calculated = false;
    int64_t minus_one_idx = -1;

    ov::Tensor lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(get_input_source_output(1));
    if (lb && ub) {
        const auto lower_bound = std::make_shared<op::v0::Constant>(lb.get_element_type(), lb.get_shape(), lb.data())
                                     ->cast_vector<int64_t>();
        auto upper_bound = std::make_shared<op::v0::Constant>(ub.get_element_type(), ub.get_shape(), ub.data())
                               ->cast_vector<int64_t>();
        shape_can_be_calculated = true;
        OPENVINO_ASSERT(lower_bound.size() == upper_bound.size());
        const TensorLabel& labels = get_input_source_output(1).get_tensor().get_value_label();
        OPENVINO_ASSERT(labels.empty() || lower_bound.size() == labels.size());

        for (size_t i = 0; i < lower_bound.size(); ++i) {
            NODE_VALIDATION_CHECK(this,
                                  lower_bound[i] >= -1 && upper_bound[i] >= -1,
                                  "Dim size cannot be less than -1");

            if (lower_bound[i] == -1 &&
                upper_bound[i] == -1) {  // ctor of Dimension(-1) would turn input Dimension(0, max_int)
                NODE_VALIDATION_CHECK(this, minus_one_idx == -1, "More than one dimension has size of -1");
                minus_one_idx = static_cast<int64_t>(i);
            }

            // We must handle i32 fully dynamic dimension in a special way
            if (get_input_element_type(1) == element::i32 &&
                upper_bound[i] == std::numeric_limits<std::int32_t>::max()) {
                upper_bound[i] = std::numeric_limits<std::int64_t>::max();
            }
            auto d = Dimension(lower_bound[i], upper_bound[i]);
            if (!labels.empty() && labels[i])
                ov::DimensionTracker::set_label(d, labels[i]);
            reshape_pattern.emplace_back(d);
        }
        // For scalar case reshape_patter should be empty but scalar reshape pattern should be empty
        // or equal to 1
        if (output_rank.is_static() && output_rank.get_length() == 0 && !lower_bound.empty()) {
            reshape_pattern.clear();
            OPENVINO_ASSERT(lower_bound.size() == 1);
            NODE_VALIDATION_CHECK(this,
                                  lower_bound[0] == 1 && upper_bound[0] == 1,
                                  "The value of scalar shape pattern should be equal to 1!");
        }
    }

    if (shape_can_be_calculated) {
        std::vector<Dimension> output_shape(output_rank.get_length());
        calculate_output_shape(reshape_pattern, minus_one_idx, input_pshape, output_shape);
        set_output_type(0, get_input_element_type(0), output_shape);
    }
}

shared_ptr<Node> op::v1::Reshape::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Reshape_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

#define COMPUTE_OUT_SHAPE_CASE(a, ...)                                    \
    case element::Type_t::a: {                                            \
        OV_OP_SCOPE(OV_PP_CAT3(compute_reshape_out_shape, _, a));         \
        reshapeop::compute_output_shape<element::Type_t::a>(__VA_ARGS__); \
    } break;

bool op::v1::Reshape::evaluate_reshape(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    // infer and set output shape if the output shape contain -1
    // and zero value dimension
    std::vector<int64_t> out_shape_val;

    switch (inputs[1].get_element_type()) {
        COMPUTE_OUT_SHAPE_CASE(i8, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i16, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i32, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(i64, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u8, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u16, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u32, inputs[1], out_shape_val);
        COMPUTE_OUT_SHAPE_CASE(u64, inputs[1], out_shape_val);
    default:
        OPENVINO_THROW("shape_pattern element type is not integral data type");
    }

    std::vector<Dimension> reshape_pattern;
    int64_t minus_one_idx = -1;
    for (size_t i = 0; i < out_shape_val.size(); ++i) {
        NODE_VALIDATION_CHECK(this, out_shape_val[i] >= -1, "Dim size cannot be less than -1");
        if (out_shape_val[i] == -1) {  // ctor of Dimension(-1) would turn input Dimension(0, max_int)
            NODE_VALIDATION_CHECK(this, minus_one_idx == -1, "More than one dimension has size of -1");
            minus_one_idx = static_cast<int64_t>(i);
        }
        reshape_pattern.emplace_back(out_shape_val[i]);
    }

    std::vector<Dimension> output_shape(out_shape_val.size());
    calculate_output_shape(reshape_pattern, minus_one_idx, inputs[0].get_shape(), output_shape);
    OPENVINO_ASSERT(ov::PartialShape(output_shape).is_static());
    outputs[0].set_shape(ov::PartialShape(output_shape).to_shape());

    OPENVINO_SUPPRESS_DEPRECATED_START
    const AxisVector order = ngraph::get_default_order(inputs[0].get_shape());
    OPENVINO_SUPPRESS_DEPRECATED_END
    ngraph::runtime::opt_kernel::reshape(static_cast<char*>(inputs[0].data()),
                                         static_cast<char*>(outputs[0].data()),
                                         inputs[0].get_shape(),
                                         order,
                                         outputs[0].get_shape(),
                                         inputs[0].get_element_type().size());
    return true;
}

bool op::v1::Reshape::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Reshape_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2);
    if (outputs.empty())
        outputs.emplace_back(ov::Tensor(inputs[0].get_element_type(), {0}));
    else
        OPENVINO_ASSERT(outputs.size() == 1);
    return evaluate_reshape(outputs, inputs);
}

bool op::v1::Reshape::has_evaluate() const {
    OV_OP_SCOPE(v1_Reshape_has_evaluate);
    switch (get_input_element_type(1)) {
    case ov::element::i8:
    case ov::element::i16:
    case ov::element::i32:
    case ov::element::i64:
    case ov::element::u8:
    case ov::element::u16:
    case ov::element::u32:
    case ov::element::u64:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v1::Reshape::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Reshape::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool op::v1::Reshape::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return ov::default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool op::v1::Reshape::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}

namespace {
bool fully_eq(const Dimension& rhs, const Dimension& lhs) {
    return rhs == lhs && ov::DimensionTracker::get_label(rhs) == ov::DimensionTracker::get_label(lhs) &&
           (ov::DimensionTracker::get_label(rhs) || rhs.is_static());
}

Dimension resolve_minus_one(const Node* reshape_node,
                            vector<Dimension>& input_product,
                            vector<Dimension>& output_product) {
    std::vector<Dimension> to_delete_from_output, to_delete_from_input;
    Dimension input_const_part(1), output_const_part(1);

    for (const auto& dim : output_product)
        if (dim.is_static()) {
            output_const_part *= dim;
            to_delete_from_output.push_back(dim);
        }

    for (const auto& dim : input_product)
        if (dim.is_static()) {
            input_const_part *= dim;
            to_delete_from_input.push_back(dim);
        }

    for (const auto& dim : to_delete_from_input) {
        input_product.erase(std::remove_if(input_product.begin(),
                                           input_product.end(),
                                           [=](const Dimension& d) {
                                               return fully_eq(dim, d);
                                           }),
                            input_product.end());
    }
    for (const auto& dim : to_delete_from_output) {
        output_product.erase(std::remove_if(output_product.begin(),
                                            output_product.end(),
                                            [=](const Dimension& d) {
                                                return fully_eq(dim, d);
                                            }),
                             output_product.end());
    }

    to_delete_from_input.clear();
    to_delete_from_output.clear();

    if (input_const_part != output_const_part) {
        input_product.push_back(input_const_part);
        output_product.push_back(output_const_part);
    }

    for (const auto& out_dim : output_product) {
        const auto& it = std::find_if(input_product.begin(), input_product.end(), [out_dim](const Dimension& in_dim) {
            return fully_eq(out_dim, in_dim);
        });
        if (it != input_product.end()) {
            to_delete_from_output.push_back(out_dim);
            to_delete_from_input.push_back(out_dim);
        }
    }
    for (const auto& dim : to_delete_from_input) {
        input_product.erase(std::remove_if(input_product.begin(),
                                           input_product.end(),
                                           [=](const Dimension& d) {
                                               return fully_eq(dim, d);
                                           }),
                            input_product.end());
    }
    for (const auto& dim : to_delete_from_output) {
        output_product.erase(std::remove_if(output_product.begin(),
                                            output_product.end(),
                                            [=](const Dimension& d) {
                                                return fully_eq(dim, d);
                                            }),
                             output_product.end());
    }

    if (output_product.empty() && input_product.size() == 1)
        return input_product[0];

    Dimension input_dim(1), output_dim(1);
    for (const auto& i : input_product) {
        input_dim *= i;
    }
    for (const auto& i : output_product) {
        output_dim *= i;
    }

    if (output_dim == 0) {
        NODE_VALIDATION_CHECK(reshape_node,
                              input_dim == 0,
                              "Cannot infer '-1' dimension with zero-size output "
                              "dimension unless at least one input dimension is "
                              "also zero-size");
        return Dimension(0);
    } else {
        if (input_dim.is_static() && output_dim.is_static()) {
            NODE_VALIDATION_CHECK(reshape_node,
                                  input_dim.get_length() % output_dim.get_length() == 0,
                                  "Non-'-1' output dimensions do not evenly divide the input dimensions");
        }

        if (output_dim == Dimension() || input_dim == Dimension()) {
            return Dimension::dynamic();
        } else {
            auto in_min = input_dim.get_min_length(), in_max = input_dim.get_max_length();
            auto out_min = output_dim.get_min_length(), out_max = output_dim.get_max_length();

            Dimension::value_type lower;
            if (in_min == -1 || out_max == -1)
                lower = -1;  // dynamic
            else
                lower = static_cast<Dimension::value_type>(ceil(static_cast<double>(in_min) / (out_max ? out_max : 1)));

            Dimension::value_type upper;
            if (in_max == -1 || out_min == -1)
                upper = -1;  // dynamic
            else
                upper =
                    static_cast<Dimension::value_type>(floor(static_cast<double>(in_max) / (out_min ? out_min : 1)));

            if (lower == -1 || (lower > upper && upper > -1))
                return Dimension::dynamic();
            else
                return {lower, upper};
        }
    }
}
}  // namespace

void op::v1::Reshape::calculate_output_shape(vector<Dimension>& reshape_pattern,
                                             const int64_t& minus_one_idx,
                                             const ov::PartialShape& input_pshape,
                                             vector<Dimension>& output_shape) const {
    std::vector<Dimension> output_product;
    for (int64_t i = 0; i < static_cast<int64_t>(reshape_pattern.size()); ++i) {
        if (i == minus_one_idx)  // resolving everything except -1
            continue;

        auto pattern_dim = reshape_pattern[i];
        if (pattern_dim == 0 && get_special_zero()) {
            if (input_pshape.rank().is_dynamic()) {
                output_shape[i] = Dimension::dynamic();
                output_product.push_back(Dimension::dynamic());
            } else {
                NODE_VALIDATION_CHECK(this, i < input_pshape.rank().get_length(), "'0' dimension is out of range");
                output_shape[i] = input_pshape[i];
                // we do not include dimension to output product here and won't include in input
                // product later because we will divide output_product by input_product. This
                // dimension contributes to both products equally, but in case this dimension
                // is dynamic and others are not we could fully define output dimension that
                // is masked by -1
            }
        } else {
            output_shape[i] = pattern_dim;
            output_product.push_back(pattern_dim);
        }
    }
    std::vector<Dimension> input_product;
    if (input_pshape.rank().is_static())
        for (int64_t i = 0; i < input_pshape.rank().get_length(); ++i) {
            if (i < static_cast<int64_t>(reshape_pattern.size()) && reshape_pattern[i].get_min_length() == 0 &&
                reshape_pattern[i].get_max_length() == 0)
                continue;
            input_product.push_back(input_pshape[i]);
        }
    else
        input_product.push_back(Dimension::dynamic());

    if (minus_one_idx != -1)  // resolving -1 masked dimension
        output_shape[minus_one_idx] = resolve_minus_one(this, input_product, output_product);

    ov::PartialShape output_pshape(output_shape);
    if (input_pshape.is_static() && output_pshape.is_static()) {
        size_t zero_dims = std::count_if(reshape_pattern.begin(), reshape_pattern.end(), cmp::Equal<Dimension>(0));

        bool backward_compatible_check = (zero_dims && get_special_zero()) || minus_one_idx != -1;
        bool in_out_elements_equal = shape_size(input_pshape.get_shape()) == shape_size(output_pshape.to_shape());

        NODE_VALIDATION_CHECK(this,
                              backward_compatible_check || in_out_elements_equal,
                              "Requested output shape ",
                              output_shape,
                              " is incompatible with input shape ",
                              input_pshape);
    }
}
