// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API Tiling;
class TRANSFORMATIONS_API TileConvolution;
class TRANSFORMATIONS_API InitTileSize;
class TRANSFORMATIONS_API TileFunction;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::TileConvolution: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    TileConvolution() {
        // MATCHER_SCOPE(TileConvolution);
        auto conv = pattern::wrap_type<ngraph::opset7::Convolution>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto conv = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());

            const int h_count = 2;
            const int w_count = 2;

            const int oh = conv->get_output_shape(0)[2];
            const int ow = conv->get_output_shape(0)[3];

            const int ih = conv->get_input_shape(0)[2];
            const int iw = conv->get_input_shape(0)[3];

            const int kh = conv->get_input_shape(1)[2];
            const int kw = conv->get_input_shape(1)[3];

            const int stride_h = conv->get_strides()[0];
            const int stride_w = conv->get_strides()[1];

            const auto pad_begin = conv->get_pads_begin();
            const auto pad_end = conv->get_pads_end();

            std::vector<std::vector<Output<Node>>> concat(h_count, std::vector<Output<Node>>(w_count));

            const int h_step = oh / h_count;
            const int w_step = ow / w_count;

            auto get_input_coordinates = [&](int h, int w) {
                return std::make_tuple(h * stride_h, w * stride_w);
            };

            for (int i = 0, oh_begin = 0; i < h_count; ++i, oh_begin += h_step) {
                for (int j = 0, ow_begin = 0; j < w_count; ++j, ow_begin += w_step) {
                    int ih_begin, ih_end, iw_begin, iw_end;
                    std::tie(ih_begin, iw_begin) = get_input_coordinates(oh_begin, ow_begin);

                    int oh_end = oh_begin + h_step - 1;
                    int ow_end = ow_begin + w_step - 1;
                    if (j == w_count - 1) {
                        ow_end = ow - 1;
                    }
                    if (i == h_count - 1) {
                        oh_end = oh - 1;
                    }
                    std::tie(ih_end, iw_end) = get_input_coordinates(oh_end, ow_end);
                    ih_end += kh - 1;
                    iw_end += kw - 1;

                    // Handle padding
                    ih_begin -= pad_begin[0];
                    iw_begin -= pad_begin[1];

                    ih_end -= pad_begin[0];
                    iw_end -= pad_begin[1];

                    int pad_to_add_h_begin{0}, pad_to_add_w_begin{0};
                    int pad_to_add_h_end{0}, pad_to_add_w_end{0};
                    if (ih_begin < 0) {
                        pad_to_add_h_begin = abs(ih_begin);
                        ih_begin = 0;
                    }

                    if (iw_begin < 0) {
                        pad_to_add_w_begin = abs(iw_begin);
                        iw_begin = 0;
                    }

                    if (ih_end >= ih) {
                        pad_to_add_h_end = ih_end - ih + 1;
                        ih_end = ih - 1;
                    }

                    if (iw_end >= iw) {
                        pad_to_add_w_end = iw_end - iw + 1;
                        iw_end = iw - 1;
                    }

                    auto ss = create_ss(conv->input_value(0), opset7::Constant::create(element::i64, {4}, {0, 0, ih_begin, iw_begin}),
                                                              opset7::Constant::create(element::i64, {4}, {0, 0, ih_end + 1, iw_end + 1}));

                    auto new_conv = std::dynamic_pointer_cast<opset7::Convolution>(conv->copy_with_new_inputs({ss, conv->input_value(1)}));
                    new_conv->set_pads_begin({pad_to_add_h_begin, pad_to_add_w_begin});
                    new_conv->set_adding_above({pad_to_add_h_end, pad_to_add_w_end});
                    concat[i][j] = new_conv;
                }
            }

            OutputVector final_concat_inputs;
            for (int i = 0; i < concat.size(); ++i) {
                OutputVector concat_inputs;
                for (int j = 0; j < concat[i].size(); ++j) {
                    concat_inputs.push_back(concat[i][j]);
                }
                final_concat_inputs.push_back(std::make_shared<opset7::Concat>(concat_inputs, 3));
            }
            auto final_concat = std::make_shared<opset7::Concat>(final_concat_inputs, 2);

            // copy_runtime_info(conv, {conv1, conv2, conv3, conv4, concat1, concat2, concat3});

            final_concat->set_friendly_name(conv->get_friendly_name());
            replace_node(conv, final_concat);

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Check");
        register_matcher(m, callback);
    }

    std::shared_ptr<Node> create_ss(Output<Node> input, Output<Node> begin, Output<Node> end) {
        return std::make_shared<opset7::StridedSlice>(input, begin, end, opset7::Constant::create(element::i64, {4}, {1, 1, 1, 1}),
                                                      std::vector<int64_t>(4, 0), std::vector<int64_t>({1, 1, 0, 0}), std::vector<int64_t>(4, 0));
    }
};

class ngraph::pass::InitTileSize: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    InitTileSize() {
        // MATCHER_SCOPE(TileConvolution);
        auto conv = pattern::wrap_type<ngraph::opset7::Result>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto result = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());

            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Check");
        register_matcher(m, callback);
    }
};

class ngraph::pass::TileFunction: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    TileFunction() {
        // MATCHER_SCOPE(TileConvolution);
        auto conv = pattern::wrap_type<ngraph::opset7::Convolution>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto conv = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Check");
        register_matcher(m, callback);
    }
};

class ngraph::pass::Tiling: public ngraph::pass::GraphRewrite {
public:
    // NGRAPH_RTTI_DECLARATION;
    Tiling() {
        add_matcher<ngraph::pass::TileConvolution>();
    }
};