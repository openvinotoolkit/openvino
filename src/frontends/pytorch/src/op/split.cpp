// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "helper_ops/internal_op.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_list_view_fx(const NodeContext& context) {
    const auto& name = context.get_op_type();
    const auto count = context.get_decoder()->output_list_size();
    const auto data = context.get_input(0);
    const auto zero = v0::Constant::create(element::i64, Shape{1}, {0});
    const auto one = v0::Constant::create(element::i64, Shape{1}, {1});
    const auto dim_index = name == "aten.unbind.int" ? 1 : 2;
    const auto axis = context.const_input<int64_t>(dim_index);
    const auto dim = v0::Constant::create(element::i64, Shape{1}, {axis});
    const auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(data));
    const auto size = context.mark_node(std::make_shared<v8::Gather>(shape, dim, zero));
    Output<Node> step;
    Output<Node> remainder;
    std::vector<int64_t> indices;
    const bool sections = name == "aten.tensor_split.sections";
    if (name == "aten.tensor_split.indices") {
        indices = context.const_input<std::vector<int64_t>>(1);
    } else if (name != "aten.unbind.int") {
        const auto amount = context.const_input<int64_t>(1);
        const auto divisor = v0::Constant::create(element::i64, Shape{1}, {amount});
        if (sections) {
            step = context.mark_node(std::make_shared<v1::Divide>(size, divisor, true));
            remainder = context.mark_node(std::make_shared<v1::Mod>(size, divisor));
        } else if (name == "aten.split.Tensor") {
            step = divisor;
        } else {
            const auto offset = v0::Constant::create(element::i64, Shape{1}, {amount - 1});
            const auto rounded = context.mark_node(std::make_shared<v1::Add>(size, offset));
            step = context.mark_node(std::make_shared<v1::Divide>(rounded, divisor, true));
        }
    }
    OutputVector outputs;
    Output<Node> start = zero;
    for (size_t i = 0; i < count; ++i) {
        if (name == "aten.unbind.int") {
            const auto index = v0::Constant::create(element::i64, Shape{}, {i});
            outputs.push_back(context.mark_node(std::make_shared<v8::Gather>(data, index, dim)));
            continue;
        }
        Output<Node> end;
        if (i + 1 == count) {
            end = size;
        } else if (!indices.empty()) {
            end = v0::Constant::create(element::i64, Shape{1}, {indices[i]});
        } else {
            const auto index = v0::Constant::create(element::i64, Shape{1}, {i + 1});
            end = context.mark_node(std::make_shared<v1::Multiply>(step, index));
            if (sections) {
                const auto extra = context.mark_node(std::make_shared<v1::Minimum>(index, remainder));
                end = context.mark_node(std::make_shared<v1::Add>(end, extra));
            }
        }
        outputs.push_back(context.mark_node(std::make_shared<v8::Slice>(data, start, end, one, dim)));
        start = end;
    }
    return {context.mark_node(make_list_construct(outputs))};
}

OutputVector translate_list_unpack_fx(const NodeContext& context) {
    // Reuse TorchScript's list-unpack normalization for ATen list operations.
    // Export fixes the list length, even when the tensor dimensions are dynamic.
    const auto& fx_name = context.get_op_type();
    const auto overload = fx_name.find('.', 5);
    const auto ts_name = "aten::" + fx_name.substr(5, overload - 5);
    const auto count = context.get_decoder()->output_list_size();
    auto operation = context.mark_node(
        std::make_shared<PtFrameworkNode>(std::make_shared<InternalOpDecoder>(ts_name, 1), context.inputs()));
    auto unpack = context.mark_node(
        std::make_shared<PtFrameworkNode>(std::make_shared<InternalOpDecoder>("prim::ListUnpack", count),
                                          OutputVector{operation}));
    return {context.mark_node(make_list_construct(unpack->outputs()))};
}

OutputVector translate_split_with_sizes(const NodeContext& context) {
    // aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
    num_inputs_check(context, 2, 3, true);
    auto data = context.get_input(0);
    auto split_lengths = get_input_concat_if_list(context, 1);
    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    } else {
        dim = context.get_input(2);
    }

    auto complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());
    bool is_complex = complex != nullptr;
    if (is_complex) {
        if (dim.get_element_type() != element::i32) {
            dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
        }
        auto rank = std::get<1>(get_shape_rank(context, data, true));
        dim = normalize_axis(context, dim, rank);
        data = complex->get_input_source_output(0);
    }

    auto split = context.mark_node(std::make_shared<v1::VariadicSplit>(data, dim, split_lengths));

    auto res = split->outputs();
    if (is_complex) {
        for (auto& output : res) {
            output = context.mark_node(std::make_shared<ComplexTypeMark>(output));
        }
    }
    return {context.mark_node(make_list_construct(res))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
