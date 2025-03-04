// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/rdft.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_fft_rfftn(const NodeContext& context) {
    // aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    num_inputs_check(context, 1, 4);
    auto input = context.get_input(0);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));

    Output<Node> input_shape;
    Output<Node> input_rank_scalar;
    std::tie(input_shape, input_rank_scalar) = get_shape_rank(context, input, true);

    Output<Node> raw_s;
    // Inputs can be either none or List. Check whether input values should be used or should be set to default values.
    if (!context.input_is_none(1)) {
        // s is provided, load from input.
        raw_s = get_input_concat_if_list(context, 1);
        raw_s = context.mark_node(std::make_shared<v0::Convert>(raw_s, element::i32));
    }
    Output<Node> dim;
    // Handle dim parameter containing vector of integers indicating dimensions to be transformed.
    if (!context.input_is_none(2)) {
        // dim is provided, load from input.
        dim = get_input_concat_if_list(context, 2);
        dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
    } else if (!context.input_is_none(1)) {
        // If dim is default and s is provided, use last s_len dimensions where s_len is length of s.
        auto s_len = context.mark_node(std::make_shared<v3::ShapeOf>(raw_s, element::i32));
        auto slice_start = context.mark_node(std::make_shared<v1::Subtract>(input_rank_scalar, s_len));
        auto slice_start_scalar = context.mark_node(std::make_shared<v0::Squeeze>(slice_start));
        dim = context.mark_node(
            std::make_shared<v4::Range>(slice_start_scalar, input_rank_scalar, const_1, element::i32));
    } else {
        // Dim and s are set to default, use all of dimensions.
        dim = context.mark_node(std::make_shared<v4::Range>(const_0, input_rank_scalar, const_1, element::i32));
    }

    Output<Node> s;
    if (context.input_is_none(1)) {
        // Value for s was set to default, use full size for all dimensions.
        s = context.mark_node(std::make_shared<v8::Gather>(input_shape, dim, const_0));
    } else {
        // Values for s were provided. Replace -1 values with default full size in given dimension.
        auto full_s_cond = context.mark_node(std::make_shared<v1::Equal>(raw_s, const_neg_1));
        auto full_s_values = context.mark_node(std::make_shared<v8::Gather>(input_shape, dim, const_0));
        s = context.mark_node(std::make_shared<v1::Select>(full_s_cond, full_s_values, raw_s));
    }

    // Handle norm parameter indicating normalization mode to use. Defaults to "backward".
    std::string norm = "backward";
    if (!context.input_is_none(3)) {
        norm = context.const_input<std::string>(3);
    }

    auto rdft = context.mark_node(std::make_shared<v9::RDFT>(input, dim, s));

    // Apply normalizations
    auto n_int = context.mark_node(std::make_shared<v1::ReduceProd>(s, const_0));
    auto n = context.mark_node(std::make_shared<v1::ConvertLike>(n_int, rdft));
    Output<Node> normalized_rfftn;
    if (norm == "forward") {
        // Normalize by 1/n
        normalized_rfftn = context.mark_node(std::make_shared<v1::Divide>(rdft, n));
    } else if (norm == "backward") {
        // No normalization
        normalized_rfftn = rdft;
    } else if (norm == "ortho") {
        // Normalize by 1/sqrt(n)
        auto sqrt_n = context.mark_node(std::make_shared<v0::Sqrt>(n));
        normalized_rfftn = context.mark_node(std::make_shared<v1::Divide>(rdft, sqrt_n));
    } else {
        FRONT_END_THROW(
            "aten::fft_rfftn: unrecognized normalization mode. Only forward, backward and ortho are supported.");
    }

    return {std::make_shared<ComplexTypeMark>(normalized_rfftn, normalized_rfftn.get_element_type())};
}

OutputVector translate_fft_irfftn(const NodeContext& context) {
    // aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    num_inputs_check(context, 1, 4, true);
    auto input = context.get_input(0);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    PYTORCH_OP_CONVERSION_CHECK(complex_type_mark, "aten::fft_irfftn operation expects complex type tensor on input.");
    input = complex_type_mark->get_data();

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_scalar_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_scalar_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));

    // Input shape of complex number (excluding dimension created by concatenation of real and imag)
    auto complex_input_shape = get_complex_shape(context, input);
    auto input_rank = context.mark_node(std::make_shared<v3::ShapeOf>(complex_input_shape, element::i32));
    auto input_rank_scalar = context.mark_node(std::make_shared<v0::Squeeze>(input_rank));

    Output<Node> raw_s;
    // Inputs can be either none or List. Check whether input values should be used or should be set to default values.
    if (!context.input_is_none(1)) {
        // s is provided, load from input.
        raw_s = get_input_concat_if_list(context, 1);
        raw_s = context.mark_node(std::make_shared<v0::Convert>(raw_s, element::i32));
    }

    // Handle dim parameter containing vector of integers indicating dimensions to be transformed.
    Output<Node> dim;
    if (!context.input_is_none(2)) {
        // Dim values is provided, load from input.
        dim = get_input_concat_if_list(context, 2);
        dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
    } else if (!context.input_is_none(1)) {
        // If dim is default and s is provided, use last s_len dimensions where s_len is length of s.
        auto s_len = context.mark_node(std::make_shared<v3::ShapeOf>(raw_s, element::i32));
        auto range_start = context.mark_node(std::make_shared<v1::Subtract>(input_rank, s_len));
        auto range_start_scalar = context.mark_node(std::make_shared<v0::Squeeze>(range_start));
        dim = context.mark_node(
            std::make_shared<v4::Range>(range_start_scalar, input_rank_scalar, const_scalar_1, element::i32));
    } else {
        // Dim and s are set to default, use all of dimensions.
        dim = context.mark_node(
            std::make_shared<v4::Range>(const_scalar_0, input_rank_scalar, const_scalar_1, element::i32));
    }

    // Calculate default s values. Use full available size except last element, which is set to even value in last
    // dimension: s[-1] = 2 * (complex_input_shape[dim[-1]])
    auto default_s_raw = context.mark_node(std::make_shared<v8::Gather>(complex_input_shape, dim, const_0));
    auto last_s = context.mark_node(std::make_shared<v8::Gather>(default_s_raw, const_neg_1, const_0));
    auto last_s_m_1 = context.mark_node(std::make_shared<v1::Subtract>(last_s, const_1));
    auto s_upd = context.mark_node(std::make_shared<v1::Multiply>(last_s_m_1, const_2));
    auto s_shape = context.mark_node(std::make_shared<v3::ShapeOf>(default_s_raw, element::i32));
    auto last_s_idx = context.mark_node(std::make_shared<v1::Subtract>(s_shape, const_1));
    auto default_s = context.mark_node(std::make_shared<v3::ScatterUpdate>(default_s_raw, last_s_idx, s_upd, const_0));

    // Handle s parameter containing vector of intigers indicating signal sizes for dimensions.
    Output<Node> s;
    if (!context.input_is_none(1)) {
        // Values for s were provided. Replace -1 values with default full size in given dimension.
        auto full_s_cond = context.mark_node(std::make_shared<v1::Equal>(raw_s, const_neg_1));
        s = context.mark_node(std::make_shared<v1::Select>(full_s_cond, default_s, raw_s));
    } else {
        // Value for s was set to default.
        s = default_s;
    }

    // Handle norm parameter indicating normalization mode to use. Defaults to "backward".
    std::string norm = "backward";
    if (!context.input_is_none(3)) {
        norm = context.const_input<std::string>(3);
    }

    auto irdft = context.mark_node(std::make_shared<v9::IRDFT>(input, dim, s));

    // Apply normalizations.
    auto n_int = context.mark_node(std::make_shared<v1::ReduceProd>(s, const_0));
    auto n = context.mark_node(std::make_shared<v1::ConvertLike>(n_int, irdft));
    Output<Node> normalized_irfftn;
    if (norm == "forward") {
        normalized_irfftn = context.mark_node(std::make_shared<v1::Multiply>(irdft, n));
    } else if (norm == "backward") {
        normalized_irfftn = irdft;
    } else if (norm == "ortho") {
        auto sqrt_n = context.mark_node(std::make_shared<v0::Sqrt>(n));
        normalized_irfftn = context.mark_node(std::make_shared<v1::Multiply>(irdft, sqrt_n));
    } else {
        FRONT_END_THROW(
            "aten::fft_irfftn: unrecognized normalization mode. Only forward, backward and ortho are supported.");
    }
    return {normalized_irfftn};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
