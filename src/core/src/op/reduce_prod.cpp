// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/reduce_prod.hpp"

namespace ov {
namespace op {
namespace reduce_prod {
namespace {
bool has_non_negative_bounds_on_data(const Node* const op) {
    const auto& lb = op->get_input_tensor(0).get_lower_value();
    const auto& ub = op->get_input_tensor(0).get_upper_value();

    return lb && ub && ov::util::reduce_and(ov::util::greater_equal(lb, 0)) &&
           ov::util::reduce_and(ov::util::greater_equal(ub, 0));
}
}  // namespace

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::reduce_prod(in0.data<const T>(), out.data<T>(), in0.get_shape(), reduction_axes);
        return true;
    }
};
}  // namespace reduce_prod
namespace v1 {

ReduceProd::ReduceProd(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ReduceProd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceProd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceProd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool ReduceProd::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceProd_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto reduction_axes = ov::util::try_get_normalized_axis_set(inputs[1], inputs[0].get_shape().size(), *this);
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_ReduceProd_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      reduce_prod::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      reduction_axes);
}

bool ReduceProd::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceProd_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

bool ReduceProd::evaluate_lower(ov::TensorVector& output_values) const {
    return reduce_prod::has_non_negative_bounds_on_data(this) && get_input_tensor(1).has_and_set_bound() &&
           default_lower_bound_evaluator(this, output_values);
}

bool ReduceProd::evaluate_upper(ov::TensorVector& output_values) const {
    if (!reduce_prod::has_non_negative_bounds_on_data(this) || !get_input_tensor(1).has_and_set_bound())
        return false;
    // We need to cover a case: if an Upper Bound comes from ShapeOf and contains
    // dynamic dimension (-1) - it has a value max_of_type, which points on
    // a maximum possible value. For example, Upper Bound of shape [-1, 12] is
    // [max_of_type, 12].
    // In such case we shouldn't evaluate a real ReduceProd because it'll cause an
    // overflow and returns wrong value. We should return an Upper Bound as for [-1],
    // which will be evaluated as [max_of_type]
    // In case dimensions has a zero dimension - it should return 0 in any case
    if (tensor_has_max_value(get_input_tensor(0).get_upper_value()) &&
        !tensor_has_zero_value(get_input_tensor(0).get_upper_value())) {
        const auto max_constant = ov::util::make_tensor_of_max_value(get_output_element_type(0));
        OPENVINO_ASSERT(max_constant.get_byte_size() <= output_values[0].get_byte_size());
        std::memcpy(output_values[0].data(), max_constant.data(), max_constant.get_byte_size());
        return true;
    }

    return default_upper_bound_evaluator(this, output_values);
}

bool ReduceProd::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    const auto& input_symbols = get_input_tensor(0).get_value_symbol();
    if (input_symbols.empty())
        return false;

    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.is_dynamic())
        return false;

    const auto& shape = input_shape.to_shape();
    if (input_symbols.size() != ov::shape_size(shape))
        return false;

    // Get reduction axes
    auto axes_node = ov::util::get_constant_from_source(input_value(1));
    if (!axes_node)
        return false;

    const auto rank = shape.size();
    auto axes_vec = axes_node->cast_vector<int64_t>();
    AxisSet reduction_axes;
    for (auto& axis : axes_vec) {
        if (axis < 0)
            axis += static_cast<int64_t>(rank);
        reduction_axes.insert(static_cast<size_t>(axis));
    }

    const auto out_shape = ov::util::reduce(shape, reduction_axes, get_keep_dims());
    const auto out_size = ov::shape_size(out_shape);
    const auto out_rank = out_shape.size();

    // Build mapping: output dim index -> input dim index
    std::vector<size_t> out_to_in;
    if (get_keep_dims()) {
        for (size_t d = 0; d < rank; ++d)
            out_to_in.push_back(d);
    } else {
        for (size_t d = 0; d < rank; ++d)
            if (reduction_axes.count(d) == 0)
                out_to_in.push_back(d);
    }

    // Precompute reduction axes info
    std::vector<size_t> red_axes_vec(reduction_axes.begin(), reduction_axes.end());
    std::vector<size_t> red_sizes;
    size_t red_total = 1;
    for (auto ax : red_axes_vec) {
        red_sizes.push_back(shape[ax]);
        red_total *= shape[ax];
    }

    // Precompute input strides
    std::vector<size_t> input_strides(rank, 1);
    for (size_t i = rank; i > 1; --i)
        input_strides[i - 2] = input_strides[i - 1] * shape[i - 1];

    output_symbols.resize(1);
    auto& out_syms = output_symbols[0];
    out_syms.resize(out_size, nullptr);

    for (size_t flat_out = 0; flat_out < out_size; ++flat_out) {
        // Decompose output flat index to output coords
        std::vector<size_t> out_coords(out_rank);
        size_t rem = flat_out;
        for (size_t d = out_rank; d > 0; --d) {
            out_coords[d - 1] = rem % out_shape[d - 1];
            rem /= out_shape[d - 1];
        }

        // Map output coords to base input coords (reduction dims set to 0)
        std::vector<size_t> base_in_coords(rank, 0);
        for (size_t od = 0; od < out_rank; ++od)
            base_in_coords[out_to_in[od]] = out_coords[od];

        // Iterate over all positions in the reduction slice, multiplying symbols
        std::shared_ptr<ov::Symbol> product = nullptr;
        bool first = true;
        for (size_t ri = 0; ri < red_total; ++ri) {
            auto in_coords = base_in_coords;
            size_t r = ri;
            for (size_t k = red_axes_vec.size(); k > 0; --k) {
                in_coords[red_axes_vec[k - 1]] = r % red_sizes[k - 1];
                r /= red_sizes[k - 1];
            }

            size_t flat_in = 0;
            for (size_t d = 0; d < rank; ++d)
                flat_in += in_coords[d] * input_strides[d];

            const auto& s = (flat_in < input_symbols.size()) ? input_symbols[flat_in] : nullptr;
            if (first) {
                product = s;
                first = false;
            } else {
                product = ov::symbol::mul(product, s);
            }
        }
        out_syms[flat_out] = product;
    }

    for (const auto& s : out_syms)
        if (s != nullptr)
            return true;
    return false;
}

}  // namespace v1
}  // namespace op
}  // namespace ov
