// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/op/util/arithmetic_reduction.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

template <typename T>
NamedOutputs reduce_ops(const NodeContext& node) {
    auto x = node.get_input("X");
    auto keep_dim = node.get_attribute<bool>("keep_dim");
    auto reduce_all = node.get_attribute<bool>("reduce_all", false);

    PADDLE_OP_CHECK(node, x.get_partial_shape().rank().is_static(), "reduce_ops: X rank must be static!");
    int64_t input_rank = x.get_partial_shape().rank().get_length();
    std::vector<int64_t> dims(input_rank);

    auto any = node.get_attribute_as_any("dim");
    if (any.is<std::vector<int32_t>>()) {
        auto dim = any.as<std::vector<int32_t>>();
        dims.resize(dim.size());
        std::transform(dim.begin(), dim.end(), dims.begin(), [](int32_t value) {
            return static_cast<int64_t>(value);
        });
    } else {
        dims = node.get_attribute<std::vector<int64_t>>("dim");
    }

    std::transform(dims.begin(), dims.end(), dims.begin(), [&input_rank](int64_t value) {
        return value >= 0 ? value : value + input_rank;
    });

    int64_t axis_size = static_cast<int64_t>(dims.size());
    reduce_all = reduce_all || (axis_size == input_rank || axis_size == 0);

    if (reduce_all) {
        dims = std::vector<int64_t>(input_rank);
        std::iota(dims.begin(), dims.end(), 0);
    }

    auto axes_node = default_opset::Constant::create(ov::element::i32, {dims.size()}, dims);
    bool scalar_output = !keep_dim;
    if (scalar_output) {
        for (int32_t i = 0; i < input_rank; i++) {
            if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
                scalar_output = false;
                break;
            }
        }
    }

    // Paddle's reduce_sum/reduce_prod carry an `out_dtype` attribute (a VarType_Type code,
    // -1 == "derive from input"). The arithmetic reductions reject boolean input and, when an
    // explicit out_dtype is requested, must accumulate in that type. Convert the input *before*
    // reducing to match Paddle's cast-before-accumulate semantics (e.g. paddle.sum(fp32 {1.9,1.9},
    // dtype='int64') == 2, not 3). Logical reductions (reduce_all/reduce_any) accept booleans and
    // are left untouched.
    ov::element::Type out_type = element::dynamic;
    if (node.has_attribute("out_dtype")) {
        if (node.get_attribute<int32_t>("out_dtype") != -1) {
            out_type = node.get_attribute<ov::element::Type>("out_dtype");
        }
    }
    constexpr bool is_arithmetic = std::is_base_of<ov::op::util::ArithmeticReduction, T>::value;
    if (is_arithmetic) {
        const auto input_type = x.get_element_type();
        if (input_type == ov::element::boolean) {
            // Paddle promotes sums/prods of boolean tensors to int64 by default.
            x = std::make_shared<default_opset::Convert>(x, out_type.is_dynamic() ? ov::element::i64 : out_type);
        } else if (input_type.is_static() && out_type.is_static() && out_type != input_type) {
            x = std::make_shared<default_opset::Convert>(x, out_type);
        }
    }

    auto reduce_node = std::make_shared<T>(x, axes_node, keep_dim);
    const auto output_info = node.get_output_port_infos("Out");
    size_t output_size = output_info[0].second.size();
    std::shared_ptr<Node> result = reduce_node;
    if (scalar_output && output_size) {
        auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {0});
        result = std::make_shared<default_opset::Unsqueeze>(reduce_node, unsqueeze_scalar);
    }

    return node.default_single_output_mapping({result}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
