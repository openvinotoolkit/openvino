// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/aszp_decomposition.hpp"

#include "aszp_decomposition.hpp"
#include "backend/gna_limitations.hpp"
#include "memory"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "ngraph/rt_info.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils/transformation_helper.hpp"

using namespace ngraph;
using namespace ov::intel_gna::pass;

namespace ov {
namespace intel_gna {
namespace pass {

static std::tuple<int64_t, int64_t, int64_t> extract_height_padding(ov::CoordinateDiff pads_begin,
                                                                    ov::CoordinateDiff pads_end) {
    auto height_begin = pads_begin[0];
    auto height_end = pads_end[0];
    return std::make_tuple(height_begin, height_end, std::abs(height_begin - height_end));
}

static std::tuple<int64_t, int64_t, int64_t> extract_width_padding(ov::CoordinateDiff pads_begin,
                                                                   ov::CoordinateDiff pads_end) {
    auto width_begin = pads_begin[1];
    auto width_end = pads_end[1];
    return std::make_tuple(width_begin, width_end, std::abs(width_begin - width_end));
}

std::shared_ptr<ov::opset11::Transpose> create_transpose(const ov::Output<ov::Node>& input) {
    return std::make_shared<ov::opset11::Transpose>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
}

std::shared_ptr<ov::opset11::Reshape> create_reshape(const ov::Output<ov::Node>& input,
                                                     uint64_t ndims,
                                                     ov::Shape shape) {
    return std::make_shared<ov::opset11::Reshape>(
        input,
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{ndims}, shape)->output(0),
        false);
}

std::shared_ptr<ov::opset11::Constant> create_zero_const(ov::Shape shape) {
    return ov::opset11::Constant::create(ov::element::i64, shape, std::vector<float>(shape[0] * shape[1], 0.0f));
}

std::shared_ptr<ov::op::v0::Concat> concatenate_zeros(uint64_t pad_begin,
                                                      uint64_t pad_end,
                                                      std::shared_ptr<ov::Node> padding_const,
                                                      std::shared_ptr<ov::Node> input_node) {
    ov::OutputVector concat_vector;
    if (pad_begin > pad_end) {
        concat_vector.push_back(padding_const->output(0));
        concat_vector.push_back(input_node->output(0));
    } else {
        concat_vector.push_back(input_node->output(0));
        concat_vector.push_back(padding_const->output(0));
    }
    return std::make_shared<ov::opset11::Concat>(concat_vector, 1);
}

std::shared_ptr<ov::opset11::Transpose> get_transpose_before(std::shared_ptr<ov::Node> conv) {
    const ov::Output<ov::Node>& parent = conv->input_value(0);

    auto transpose_before = std::dynamic_pointer_cast<ov::opset11::Transpose>(parent.get_node()->shared_from_this());
    if (nullptr == transpose_before){
         return nullptr;
    }

    auto convolution_children = conv->output(0).get_target_inputs();
    auto convolution_bias = std::dynamic_pointer_cast<ov::opset11::Add>(convolution_children.begin()->get_node()->shared_from_this());

    std::shared_ptr<ov::opset11::Transpose> transpose_after;
    if (nullptr != convolution_bias) {
        auto add_children = convolution_bias->output(0).get_target_inputs();
        if (add_children.size() != 1)
            return nullptr;
        transpose_after =
            std::dynamic_pointer_cast<ov::opset11::Transpose>(add_children.begin()->get_node()->shared_from_this());
    } else {
        transpose_after = std::dynamic_pointer_cast<ov::opset11::Transpose>(
            convolution_children.begin()->get_node()->shared_from_this());
    }

    if (transpose_after == nullptr) {
        return nullptr;
    }
    return transpose_before;
}

static std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> get_input_dimensions(ov::Shape input_shape) {
    uint64_t N = input_shape[0];
    uint64_t H = input_shape[1];
    uint64_t W = input_shape[2];
    uint64_t C = input_shape[3];
    return std::make_tuple(N, H, W, C);
}

ov::Output<ov::Node> decompose_height(ov::Output<ov::Node> input,
                                      ov::CoordinateDiff pads_begin,
                                      ov::CoordinateDiff pads_end,
                                      ov::Shape conv_input_shape) {
    uint64_t height_begin, height_end, height_padding, width_padding;
    std::tie(height_begin, height_end, height_padding) = extract_height_padding(pads_begin, pads_end);
    width_padding = std::abs(pads_end[1] - pads_begin[1]);
    uint64_t N, C, H, W;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);

    if (0 == height_padding) {
        return input;
    }

    auto new_reshape = create_reshape(input, 2, ov::Shape{H, W * C});
    auto new_transpose = create_transpose(new_reshape->output(0));
    auto padding_const = create_zero_const(ov::Shape{W * C, height_padding});
    auto new_concat = concatenate_zeros(height_begin, height_end, padding_const, new_transpose);
    auto new_untranspose = create_transpose(new_concat->output(0));

    if (0 == width_padding) {
        return create_reshape(new_untranspose->output(0), 4, ov::Shape{N, H + height_padding, W, C})->output(0);
    }
    return (new_untranspose->output(0));
}

ov::Output<ov::Node> decompose_width(ov::Output<ov::Node> input,
                                     ov::CoordinateDiff pads_begin,
                                     ov::CoordinateDiff pads_end,
                                     ov::Shape conv_input_shape) {
    uint64_t width_begin, width_end, width_padding, height_padding;
    std::tie(width_begin, width_end, width_padding) = extract_width_padding(pads_begin, pads_end);
    height_padding = std::abs(pads_end[0] - pads_begin[0]);
    uint64_t N, H, W, C;
    std::tie(N, H, W, C) = get_input_dimensions(conv_input_shape);
    if (0 == width_padding) {
        return input;
    }

    auto new_reshape = create_reshape(input, 2, ov::Shape{(H + height_padding) * W, C});
    auto new_transpose = create_transpose(new_reshape->output(0));
    auto new_reshape2 = create_reshape(new_transpose->output(0), 2, ov::Shape{C * (H + height_padding), W});
    auto padding_const = create_zero_const(ov::Shape{C * (H + height_padding), width_padding});
    auto new_concat = concatenate_zeros(width_begin, width_end, padding_const, new_reshape2);
    auto new_unshape2 =
        create_reshape(new_concat->output(0), 2, ov::Shape{C, (H + height_padding) * (W + width_padding)});
    auto new_untranspose = create_transpose(new_unshape2->output(0));
    auto new_unshape = create_reshape(new_untranspose->output(0), 4, {N, H + height_padding, W + width_padding, C});

    return new_unshape->output(0);
}

void trimm_padding(ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end) {
    if (pads_begin[0] > pads_end[0]) {
        pads_begin[0] = pads_end[0];
    } else {
        pads_end[0] = pads_begin[0];
    }
    if (pads_begin[1] > pads_end[1]) {
        pads_begin[1] = pads_end[1];
    } else {
        pads_end[1] = pads_begin[1];
    }
}

std::shared_ptr<ov::Node> create_convolution(std::shared_ptr<ov::opset11::Convolution> conv,
                                             const ov::Output<ov::Node>& input,
                                             ov::CoordinateDiff pads_begin,
                                             ov::CoordinateDiff pads_end) {
    trimm_padding(pads_begin, pads_end);

    if (nullptr != conv) {
        return std::make_shared<ov::opset11::Convolution>(input,
                                                          conv->input_value(1),
                                                          conv->get_strides(),
                                                          pads_begin,
                                                          pads_end,
                                                          conv->get_dilations(),
                                                          conv->get_auto_pad());
    }

    return nullptr;
}



static bool decompose(std::shared_ptr<ov::opset11::Convolution> conv) {
    if (conv == nullptr) {
        return false;
    }

    auto pads_begin = conv->get_pads_begin();
    auto pads_end = conv->get_pads_end();
    if (pads_begin.size() < 2 || pads_end.size() < 2) {
        return false;
    }
    if (pads_begin[0] == pads_end[0] && pads_begin[1] == pads_end[1]) {
        return false;
    }
    auto transpose_before = get_transpose_before(conv);
    if (nullptr == transpose_before) {
        return false;
    }

    Output<Node> input = transpose_before->input_value(0);
    auto input_shape = input.get_shape();
    if (input_shape.size() != 4 || input_shape[0] != 1) {
        return false;
    }

    Output<Node> skip_input_H_const = decompose_height(input, pads_begin, pads_end, input_shape);
    Output<Node> skip_input_W_const = decompose_width(skip_input_H_const, pads_begin, pads_end, input_shape);

    auto final_transpose = std::make_shared<ov::opset11::Transpose>(
        skip_input_W_const,
        ov::opset11::Constant::create(ov::element::i64, Shape{4}, {0, 3, 1, 2}));

    auto new_conv = create_convolution(conv, final_transpose->output(0), pads_begin, pads_end);
    if (new_conv == nullptr) {
        return false;
    }

    new_conv->set_friendly_name(conv->get_friendly_name());
    ov::copy_runtime_info(conv, new_conv);
    ov::replace_node(conv, new_conv);
    return true;
}

    AszpDecomposition::AszpDecomposition() {
        MATCHER_SCOPE(AszpDecomposition);
        auto conv = ngraph::pattern::wrap_type<ov::opset11::Convolution>();

        ov::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            auto conv = std::dynamic_pointer_cast<ov::opset11::Convolution>(m.get_match_root());
            return decompose(conv);
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
        this->register_matcher(m, callback);
    }

}  // namespace pass
}  // namespace pass
}  // namespace intel_gna