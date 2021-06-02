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
#include <ngraph/pass/manager.hpp>
#include <ngraph/rt_info.hpp>

#include <transformations/rt_info/tile_propagation_attribute.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API Tiling;
class TRANSFORMATIONS_API TileConvolution;
class TRANSFORMATIONS_API TileEltwise;
class TRANSFORMATIONS_API TileUnary;
class TRANSFORMATIONS_API TilePad;
class TRANSFORMATIONS_API TileUnsupported;
class TRANSFORMATIONS_API InitTiles;
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

            // TODO: check all consumers (intersection of all boxes)
            auto consumer = *conv->output(0).get_target_inputs().begin();
            if (!has_tiles(consumer)) return false;

            auto requested_tiles = get_tiles(consumer);

            const int ih = conv->get_input_shape(0)[2];
            const int iw = conv->get_input_shape(0)[3];

            const int kh = conv->get_input_shape(1)[2];
            const int kw = conv->get_input_shape(1)[3];

            const int stride_h = conv->get_strides()[0];
            const int stride_w = conv->get_strides()[1];

            const auto pad_begin = conv->get_pads_begin();
            const auto pad_end = conv->get_pads_end();

            Tiles tiles(requested_tiles);

            auto get_input_coordinates = [&](int h, int w) {
                return std::make_tuple(h * stride_h, w * stride_w);
            };

            for (size_t i = 0; i < requested_tiles.tiles.size(); ++i) {
                for (size_t j = 0; j < requested_tiles.tiles[i].size(); ++j) {
                    auto tile = requested_tiles.tiles[i][j];
                    int64_t ih_begin, ih_end, iw_begin, iw_end;
                    std::tie(ih_begin, iw_begin) = get_input_coordinates(tile.h_begin, tile.w_begin);
                    std::tie(ih_end, iw_end) = get_input_coordinates(tile.h_end, tile.w_end);

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

                    TileElement new_tile({ih_begin, ih_end, iw_begin, iw_end});
                    new_tile.modifier = [pad_to_add_h_begin, pad_to_add_w_begin, pad_to_add_h_end, pad_to_add_w_end](std::shared_ptr<Node> node) {
                        auto conv = std::dynamic_pointer_cast<opset7::Convolution>(node);
                        if (!conv) return;
                        conv->set_pads_begin({pad_to_add_h_begin, pad_to_add_w_begin});
                        conv->set_adding_above({pad_to_add_h_end, pad_to_add_w_end});
                    };

                    tiles.tiles[i][j] = new_tile;
                }
            }

            set_tiles(conv->input(0), tiles);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Check");
        register_matcher(m, callback);
    }
};

//class ngraph::pass::TilePad: public ngraph::pass::MatcherPass {
//public:
//    // NGRAPH_RTTI_DECLARATION;
//    TilePad() {
//        // MATCHER_SCOPE(TilePad);
//        auto pattern = pattern::wrap_type<ngraph::opset7::Pad>();
//
//        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
//            auto pad = std::dynamic_pointer_cast<opset7::Pad>(m.get_match_root());
//
//            // TODO: check all consumers (intersection of all boxes)
//            auto consumer = *pad->output(0).get_target_inputs().begin();
//            if (!has_tiles(consumer)) return false;
//
////            auto requested_tiles = get_tiles(consumer);
////
////            const int ih = pad->get_input_shape(0)[2];
////            const int iw = pad->get_input_shape(0)[3];
////
////            const int kh = 1;
////            const int kw = 1;
////
////            const int stride_h = 1;
////            const int stride_w = 1;
//
//            const auto pad_begin_node = std::dynamic_pointer_cast<opset7::Constant>(pad->input_value(1).get_node_shared_ptr());
//            const auto pad_end_node = std::dynamic_pointer_cast<opset7::Constant>(pad->input_value(2).get_node_shared_ptr());
//
//            if (!pad_begin_node || !pad_end_node) return false;
//
//            auto pad_begin = pad_begin_node->cast_vector<int64_t>();
//            auto pad_end = pad_end_node->cast_vector<int64_t>();
//
//            Tiles tiles(requested_tiles);
//
//            for (size_t i = 0; i < requested_tiles.tiles.size(); ++i) {
//                for (size_t j = 0; j < requested_tiles.tiles[i].size(); ++j) {
//                    auto tile = requested_tiles.tiles[i][j];
//                    int64_t ih_begin, ih_end, iw_begin, iw_end;
//                    std::tie(ih_begin, iw_begin) = {tile.h_begin, tile.w_begin};
//                    std::tie(ih_end, iw_end) = {tile.h_end, tile.w_end};
//
//                    // Handle padding
//                    ih_begin -= pad_begin[2];
//                    iw_begin -= pad_begin[3];
//
//                    ih_end -= pad_begin[2];
//                    iw_end -= pad_begin[3];
//
////                    int pad_to_add_h_begin{0}, pad_to_add_w_begin{0};
////                    int pad_to_add_h_end{0}, pad_to_add_w_end{0};
////                    if (ih_begin < 0) {
////                        pad_to_add_h_begin = abs(ih_begin);
////                        ih_begin = 0;
////                    }
////
////                    if (iw_begin < 0) {
////                        pad_to_add_w_begin = abs(iw_begin);
////                        iw_begin = 0;
////                    }
////
////                    if (ih_end >= ih) {
////                        pad_to_add_h_end = ih_end - ih + 1;
////                        ih_end = ih - 1;
////                    }
////
////                    if (iw_end >= iw) {
////                        pad_to_add_w_end = iw_end - iw + 1;
////                        iw_end = iw - 1;
////                    }
//
////                    TileElement new_tile({ih_begin, ih_end, iw_begin, iw_end});
////                    new_tile.modifier = [pad_to_add_h_begin, pad_to_add_w_begin, pad_to_add_h_end, pad_to_add_w_end](std::shared_ptr<Node> node) {
////                        auto pad = std::dynamic_pointer_cast<opset7::Pad>(node);
////                        if (!pad) return;
////
////                        auto old_pad_begin = pad->input_value()
////
////                        auto pad_begin_value =
////                        conv->set_pads_begin({pad_to_add_h_begin, pad_to_add_w_begin});
////                        conv->set_adding_above({pad_to_add_h_end, pad_to_add_w_end});
////                    };
//
////                    tiles.tiles[i][j] = new_tile;
//                }
//            }
//
//            set_tiles(pad->input(0), tiles);
//            return true;
//        };
//
//        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern, "Check");
//        register_matcher(m, callback);
//    }
//};

class ngraph::pass::TileEltwise: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    TileEltwise() {
        // MATCHER_SCOPE(TileConvolution);
        auto pattern = pattern::wrap_type<ngraph::op::util::BinaryElementwiseArithmetic>();

        ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
            auto node = m.get_match_root();

            // TODO: check all consumers (intersection of all boxes)
            auto consumer = *node->output(0).get_target_inputs().begin();
            if (!has_tiles(consumer)) return false;

            set_tiles(node->input(0), get_tiles(consumer));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(pattern, "Check");
        register_matcher(m, callback);
    }
};

class ngraph::pass::TileUnary: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    TileUnary() {
        // MATCHER_SCOPE(TileConvolution);
        auto unary = pattern::wrap_type<ngraph::op::util::UnaryElementwiseArithmetic, ngraph::opset7::PRelu>();

        ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
            auto node = m.get_match_root();

            // TODO: check all consumers (intersection of all boxes)
            auto consumer = *node->output(0).get_target_inputs().begin();
            if (!has_tiles(consumer)) return false;

            set_tiles(node->input(0), get_tiles(consumer));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(unary, "Check");
        register_matcher(m, callback);
    }
};

class ngraph::pass::TileUnsupported: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    TileUnsupported() {
        // MATCHER_SCOPE(TileConvolution);
        auto any_op = pattern::any_input();

        ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
            auto node = m.get_match_root();
            if (!std::dynamic_pointer_cast<opset7::Constant>(node) &&
                !std::dynamic_pointer_cast<opset7::Constant>(node)) {
                std::cout << "UNSUPPORTED TILING OP: " << node->get_type_name() << " : " << node->get_friendly_name() << std::endl;
            }

            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(any_op, "Check");
        register_matcher(m, callback);
    }
};

class ngraph::pass::InitTiles: public ngraph::pass::MatcherPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    InitTiles() {
        // MATCHER_SCOPE(TileConvolution);
        auto conv = pattern::wrap_type<ngraph::opset7::Result>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto result = m.get_match_root();

            const auto shape = result->input(0).get_shape();
            const int h = shape[2];
            const int w = shape[3];

            const int h_count = 2;
            const int w_count = 2;

            const int h_step = h / h_count;
            const int w_step = w / w_count;

            Tiles tiles(h_count, w_count);

            for (int i = 0, oh_begin = 0; i < h_count; ++i, oh_begin += h_step) {
                for (int j = 0, ow_begin = 0; j < w_count; ++j, ow_begin += w_step) {
                    int oh_end = oh_begin + h_step - 1;
                    int ow_end = ow_begin + w_step - 1;
                    if (j == w_count - 1) {
                        ow_end = w - 1;
                    }
                    if (i == h_count - 1) {
                        oh_end = h - 1;
                    }
                    tiles.tiles[i][j] = TileElement({oh_begin, oh_end, ow_begin, ow_end});
                }
            }

            set_tiles(result->input(0), tiles);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "InitTiles");
        register_matcher(m, callback);
    }
};

class ngraph::pass::TileFunction: public ngraph::pass::FunctionPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<Function> f) override {
        auto params = f->get_parameters();
        if (params.size() != 1) {
            throw ngraph_error("to much inputs");
        }
        auto param = params[0];
        assert(param->output(0).get_target_inputs().size() == 1);
        auto consumer = *param->output(0).get_target_inputs().begin();
        auto tiles = get_tiles(consumer);

        std::vector<std::vector<std::shared_ptr<Function>>> function_tiles(tiles.tiles.size(), std::vector<std::shared_ptr<Function>>(tiles.tiles[0].size()));

        for (size_t i = 0; i < tiles.tiles.size(); ++i) {
            for (size_t j = 0; j < tiles.tiles[i].size(); ++j) {
                auto f_copy = ngraph::clone_function(*f);

                for (auto && node : f_copy->get_ordered_ops()) {
                    for (auto && input : node->inputs()) {
                        if (!has_tiles(input)) continue;
                        auto input_tile = get_tiles(input).tiles[i][j];

                        if (!tile_is_equal_to_input(input_tile, input.get_shape())) {
                            auto ss = create_ss(input.get_source_output(), opset7::Constant::create(element::i64, {4},
                                                                                                    {0ll, 0ll,
                                                                                                     input_tile.h_begin +
                                                                                                     0ll,
                                                                                                     input_tile.w_begin +
                                                                                                     0ll}),
                                                opset7::Constant::create(element::i64, {4},
                                                                         {0ll, 0ll, input_tile.h_end + 1ll,
                                                                          input_tile.w_end + 1ll}));
                            input.replace_source_output(ss->output(0));
                        }

                        if (input_tile.modifier) {
                            input_tile.modifier(node);
                        }
                    }
                    node->validate_and_infer_types();
                }

                function_tiles[i][j] = f_copy;
            }
        }

        auto orig_input = f->get_parameters()[0];

        // Merge multiple nGraph Functions to single Function
        OutputVector outputs_h;
        for (size_t i = 0; i < tiles.tiles.size(); ++i) {
            OutputVector outputs_w;
            for (size_t j = 0; j < tiles.tiles[i].size(); ++j) {
                auto f_tile = function_tiles[i][j];
                auto f_tile_param = f_tile->get_parameters()[0];
                for (auto c : f_tile_param->output(0).get_target_inputs()) {
                    c.replace_source_output(orig_input);
                }
                auto f_tile_result = f_tile->get_result();
                outputs_w.push_back(f_tile_result->input_value(0));
            }
            outputs_h.push_back(std::make_shared<opset7::Concat>(outputs_w, 3));
        }

        auto final_concat = std::make_shared<opset7::Concat>(outputs_h, 2);
        auto orig_result = f->get_result();
        final_concat->set_friendly_name(orig_result->input_value(0).get_node()->get_friendly_name());
        orig_result->input(0).replace_source_output(final_concat);

        return true;
    }

    std::shared_ptr<Node> create_ss(Output<Node> input, Output<Node> begin, Output<Node> end) {
        return std::make_shared<opset7::StridedSlice>(input, begin, end, opset7::Constant::create(element::i64, {4}, {1, 1, 1, 1}),
                                                      std::vector<int64_t>(4, 0), std::vector<int64_t>({1, 1, 0, 0}), std::vector<int64_t>(4, 0));
    }

    bool tile_is_equal_to_input(const TileElement & tile, const Shape & input_shape) {
        return tile.size() == input_shape[2] * input_shape[3];
    }
};

class ngraph::pass::Tiling: public ngraph::pass::FunctionPass {
public:
    // NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<Function> f) override {
        pass::Manager m;

        auto tile_attr_prop = m.register_pass<BackwardGraphRewrite>();
        tile_attr_prop->add_matcher<ngraph::pass::InitTiles>();
        tile_attr_prop->add_matcher<ngraph::pass::TileConvolution>();
        tile_attr_prop->add_matcher<ngraph::pass::TileUnary>();
        tile_attr_prop->add_matcher<ngraph::pass::TileEltwise>();
        // tile_attr_prop->add_matcher<ngraph::pass::TilePad>();
        tile_attr_prop->add_matcher<ngraph::pass::TileUnsupported>();

        m.register_pass<TileFunction>();
        m.run_passes(f);

        return false;
    }
};