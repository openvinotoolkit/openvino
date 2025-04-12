// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/idft.hpp"
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

namespace {
Output<Node> normalize(const NodeContext& context,
                       const Output<Node>& node,
                       const Output<Node>& s,
                       const std::string& norm,
                       bool inverse) {
    if (norm == "backward") {
        // No normalization
        return node;
    }
    // Apply normalizations
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto n_int = context.mark_node(std::make_shared<v1::ReduceProd>(s, const_0));
    auto n = context.mark_node(std::make_shared<v1::ConvertLike>(n_int, node));
    Output<Node> normalized;
    if (norm == "forward") {
        // Normalize by 1/n
        if (inverse) {
            normalized = context.mark_node(std::make_shared<v1::Multiply>(node, n));
        } else {
            normalized = context.mark_node(std::make_shared<v1::Divide>(node, n));
        }
    } else if (norm == "ortho") {
        // Normalize by 1/sqrt(n)
        auto sqrt_n = context.mark_node(std::make_shared<v0::Sqrt>(n));
        if (inverse) {
            normalized = context.mark_node(std::make_shared<v1::Multiply>(node, sqrt_n));
        } else {
            normalized = context.mark_node(std::make_shared<v1::Divide>(node, sqrt_n));
        }
    } else {
        FRONT_END_THROW("Unrecognized normalization mode " + norm +
                        ". Only forward, backward and ortho are supported.");
    }
    return normalized;
}

std::tuple<Output<Node>, Output<Node>> get_dim_s(const NodeContext& context,
                                                 const Output<Node>& x,
                                                 int size,
                                                 bool is_irfft) {
    Output<Node> input_shape;
    Output<Node> input_rank_scalar;
    std::tie(input_shape, input_rank_scalar) = get_shape_rank(context, x, true);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto const_neg_1_1d = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));

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
        // dim is provided, load from input.
        dim = get_input_concat_if_list(context, 2);
        dim = context.mark_node(std::make_shared<v0::Convert>(dim, element::i32));
    } else if (!context.input_is_none(1)) {
        // If dim is default and s is provided, use last s_len dimensions where s_len is length of s.
        auto s_len = context.mark_node(std::make_shared<v3::ShapeOf>(raw_s, element::i32));
        auto start = context.mark_node(std::make_shared<v1::Subtract>(input_rank_scalar, s_len));
        auto start_scalar = context.mark_node(std::make_shared<v0::Squeeze>(start));
        dim = context.mark_node(std::make_shared<v4::Range>(start_scalar, input_rank_scalar, const_1, element::i32));
    } else {
        // Dim and s are set to default.
        switch (size) {
        case 1:
            dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
            break;
        case 2:
            dim = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-2, -1}));
            break;
        case -1:
            dim = context.mark_node(std::make_shared<v4::Range>(const_0, input_rank_scalar, const_1, element::i32));
            break;
        default:
            FRONT_END_THROW("Invalid FFT size: " + std::to_string(size));
        }
    }
    if (dim.get_partial_shape().rank().is_dynamic() || dim.get_partial_shape().rank().get_length() == 0) {
        dim = context.mark_node(std::make_shared<v1::Reshape>(dim, const_neg_1_1d, false));
    }

    Output<Node> default_s;
    if (is_irfft) {
        // Calculate default s values. Use full available size except last element, which is set to even value in last
        // dimension: s[-1] = 2 * (complex_input_shape[dim[-1]] - 1).
        auto default_s_raw = context.mark_node(std::make_shared<v8::Gather>(input_shape, dim, const_0));
        auto last_s = context.mark_node(std::make_shared<v8::Gather>(default_s_raw, const_neg_1, const_0));
        auto last_s_m_1 = context.mark_node(std::make_shared<v1::Subtract>(last_s, const_1));
        auto s_upd = context.mark_node(std::make_shared<v1::Multiply>(last_s_m_1, const_2));
        auto s_shape = context.mark_node(std::make_shared<v3::ShapeOf>(default_s_raw, element::i32));
        auto last_s_idx = context.mark_node(std::make_shared<v1::Subtract>(s_shape, const_1));
        s_upd = context.mark_node(std::make_shared<v1::Reshape>(s_upd, const_neg_1_1d, false));
        default_s = context.mark_node(std::make_shared<v3::ScatterUpdate>(default_s_raw, last_s_idx, s_upd, const_0));
    } else {
        default_s = context.mark_node(std::make_shared<v8::Gather>(input_shape, dim, const_0));
    }
    Output<Node> s;
    if (context.input_is_none(1)) {
        // Value for s was set to default, use full size for all dimensions.
        s = default_s;
    } else {
        // Values for s were provided. Replace -1 values with default full size in given dimension.
        auto full_s_cond = context.mark_node(std::make_shared<v1::Equal>(raw_s, const_neg_1));
        s = context.mark_node(std::make_shared<v1::Select>(full_s_cond, default_s, raw_s));
    }
    return {dim, s};
}

template <typename T>
OutputVector translate_fft_base(const NodeContext& context,
                                int size,
                                bool complex_input,
                                bool complex_output,
                                bool inverse = false,
                                bool is_irfft = false) {
    num_inputs_check(context, 1, 4, true);
    auto input = context.get_input(0);

    Output<Node> dim;
    Output<Node> s;
    std::tie(dim, s) = get_dim_s(context, input, size, is_irfft);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_mark) {
        PYTORCH_OP_CONVERSION_CHECK(complex_input, "Operation does not support complex type tensor on input.");
        input = complex_type_mark->get_data();
    } else {
        if (complex_input) {
            auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
            const_0 = context.mark_node(std::make_shared<v1::ConvertLike>(const_0, input));
            input = std::make_shared<ComplexTypeMark>(input, const_0)->get_data();
        }
    }

    // Handle norm parameter indicating normalization mode to use. Defaults to "backward".
    std::string norm = "backward";
    if (!context.input_is_none(3)) {
        norm = context.const_input<std::string>(3);
    }

    auto node = context.mark_node(std::make_shared<T>(input, dim, s));

    // Apply normalizations
    Output<Node> normalized = normalize(context, node, s, norm, inverse);
    if (complex_output) {
        normalized = std::make_shared<ComplexTypeMark>(normalized, normalized.get_element_type());
    }
    return {normalized};
}
}  // namespace

OutputVector translate_fft_fft(const NodeContext& context) {
    return translate_fft_base<v7::DFT>(context, 1, true, true);
}

OutputVector translate_fft_fft2(const NodeContext& context) {
    return translate_fft_base<v7::DFT>(context, 2, true, true);
}

OutputVector translate_fft_fftn(const NodeContext& context) {
    return translate_fft_base<v7::DFT>(context, -1, true, true);
}

OutputVector translate_fft_rfft(const NodeContext& context) {
    return translate_fft_base<v9::RDFT>(context, 1, false, true);
}

OutputVector translate_fft_rfft2(const NodeContext& context) {
    return translate_fft_base<v9::RDFT>(context, 2, false, true);
}

OutputVector translate_fft_rfftn(const NodeContext& context) {
    // aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    return translate_fft_base<v9::RDFT>(context, -1, false, true);
}

OutputVector translate_fft_ifft(const NodeContext& context) {
    return translate_fft_base<v7::IDFT>(context, 1, true, true, true);
}

OutputVector translate_fft_ifft2(const NodeContext& context) {
    return translate_fft_base<v7::IDFT>(context, 2, true, true, true);
}

OutputVector translate_fft_ifftn(const NodeContext& context) {
    return translate_fft_base<v7::IDFT>(context, -1, true, true, true);
}

OutputVector translate_fft_irfft(const NodeContext& context) {
    return translate_fft_base<v9::IRDFT>(context, 1, true, false, true, true);
}

OutputVector translate_fft_irfft2(const NodeContext& context) {
    return translate_fft_base<v9::IRDFT>(context, 2, true, false, true, true);
}

OutputVector translate_fft_irfftn(const NodeContext& context) {
    // aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    return translate_fft_base<v9::IRDFT>(context, -1, true, false, true, true);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
