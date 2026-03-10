// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/add.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace add {
struct Evaluate : element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::add(in0.data<const T>(), in1.data<const T>(), out.data<T>(), shape0, shape1, broadcast_spec);
        return true;
    }
};
}  // namespace add

// ------------------------------- v1 ------------------------------------------
namespace v1 {
Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Add::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Add_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v1::Add>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool Add::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Add_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Add_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i16, i32, i64, u8, u16, u32, u64),
                                      add::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool Add::has_evaluate() const {
    OV_OP_SCOPE(v1_Add_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
bool Add::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    const auto& symbols0 = get_input_tensor(0).get_value_symbol();
    const auto& symbols1 = get_input_tensor(1).get_value_symbol();

    if (symbols0.empty() && symbols1.empty())
        return false;

    const auto& pshape0 = get_input_partial_shape(0);
    const auto& pshape1 = get_input_partial_shape(1);
    const auto& out_pshape = get_output_partial_shape(0);

    if (pshape0.is_dynamic() || pshape1.is_dynamic() || out_pshape.is_dynamic())
        return false;

    const auto& shape0 = pshape0.to_shape();
    const auto& shape1 = pshape1.to_shape();
    const auto& out_shape = out_pshape.to_shape();

    const auto out_size = ov::shape_size(out_shape);
    const auto out_rank = out_shape.size();
    const auto rank0 = shape0.size();
    const auto rank1 = shape1.size();

    output_symbols.resize(1);
    auto& out_syms = output_symbols[0];
    out_syms.resize(out_size, nullptr);

    // Compute strides for output
    std::vector<size_t> out_strides(out_rank, 1);
    for (size_t i = out_rank; i > 1; --i)
        out_strides[i - 2] = out_strides[i - 1] * out_shape[i - 1];

    for (size_t flat = 0; flat < out_size; ++flat) {
        // Decompose flat index into output coordinates
        size_t remaining = flat;
        size_t idx0 = 0, idx1 = 0;
        size_t stride0 = 1, stride1 = 1;

        // Compute input flat indices with broadcasting
        // Process from last dimension to first
        for (size_t d = out_rank; d > 0; --d) {
            size_t coord = remaining % out_shape[d - 1];
            remaining /= out_shape[d - 1];

            // Map to input0 index (right-aligned)
            if (d - 1 >= out_rank - rank0) {
                size_t d0 = d - 1 - (out_rank - rank0);
                if (shape0[d0] != 1)
                    idx0 += coord * stride0;
                stride0 *= shape0[d0];
            }
            // Map to input1 index (right-aligned)
            if (d - 1 >= out_rank - rank1) {
                size_t d1 = d - 1 - (out_rank - rank1);
                if (shape1[d1] != 1)
                    idx1 += coord * stride1;
                stride1 *= shape1[d1];
            }
        }

        const auto& s0 = (idx0 < symbols0.size()) ? symbols0[idx0] : nullptr;
        const auto& s1 = (idx1 < symbols1.size()) ? symbols1[idx1] : nullptr;
        out_syms[flat] = ov::symbol::add(s0, s1);
    }

    // Check if at least one output symbol is non-null
    for (const auto& s : out_syms)
        if (s != nullptr)
            return true;
    return false;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
