// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/variant.hpp>
#include <numeric>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/rt_info/strides_property.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

static bool can_propagate_conv_stride(const std::shared_ptr<ngraph::Node>& conv) {
    const auto& kernel_shape = conv->input_value(1).get_shape();
    return std::all_of(kernel_shape.begin() + 2, kernel_shape.end(), [](size_t s) -> bool {
        return s == 1;
    });
}

static std::tuple<ngraph::Strides, bool> check_next_ops(const std::vector<ngraph::Input<ngraph::Node>>& next_ops) {
    std::vector<ngraph::Strides> strides;
    for (const auto& op : next_ops) {
        if (!has_strides_prop(op)) {
            return std::make_tuple(ngraph::Strides{}, false);
        }
        strides.push_back(get_strides_prop(op));
    }
    bool all_ops_are_valid = std::all_of(strides.begin(), strides.end(), [&strides](const ngraph::Strides& s) -> bool {
        bool all_ones = std::all_of(s.begin(), s.end(), [](size_t i) -> bool {
            return i == 1;
        });
        return s == strides[0] && !all_ones;
    });
    return std::make_tuple(strides[0], all_ops_are_valid);
}

static void insert_pooling(const ngraph::Output<ngraph::Node>& first,
                           ngraph::Input<ngraph::Node>& second,
                           const ngraph::Strides& strides) {
    auto first_node = first.get_node_shared_ptr();
    auto rank = first.get_partial_shape().rank();
    bool do_reshape = rank.is_static() && static_cast<size_t>(rank.get_length()) < strides.size() + 2;
    if (do_reshape) {
        size_t diff = strides.size() + 2 - static_cast<size_t>(rank.get_length());
        auto ones =
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{diff}, std::vector<int64_t>(diff, 1));
        auto current_shape = std::make_shared<ngraph::opset7::ShapeOf>(first);
        std::shared_ptr<ngraph::Node> new_shape =
            std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{ones, current_shape}, 0);
        std::shared_ptr<ngraph::Node> constant_new_shape = get_constant_from_source(new_shape);
        if (constant_new_shape)
            new_shape = constant_new_shape;
        first_node = std::make_shared<ngraph::opset7::Reshape>(first_node, new_shape, false);
    }
    std::shared_ptr<ngraph::Node> new_node =
        std::make_shared<ngraph::opset7::MaxPool>(first_node,
                                                  strides,
                                                  ngraph::Shape{},
                                                  ngraph::Shape{},
                                                  ngraph::Shape(strides.size(), 1));
    if (do_reshape) {
        // squeeze dimensions back
        size_t diff = strides.size() + 2 - static_cast<size_t>(rank.get_length());
        std::vector<size_t> axes(diff);
        std::iota(axes.begin(), axes.end(), 0);
        new_node = std::make_shared<ngraph::opset7::Squeeze>(
            new_node,
            ngraph::opset7::Constant::create(ngraph::element::u64, ngraph::Shape{diff}, axes));
    }
    std::shared_ptr<ngraph::Node> constant_new_node = get_constant_from_source(new_node);
    if (constant_new_node)
        new_node = constant_new_node;
    second.replace_source_output(new_node);
}

static void handle_not_equal_stride_props(std::vector<ngraph::Input<ngraph::Node>>&& next_ops) {
    for (auto& op : next_ops) {
        if (!has_strides_prop(op))
            continue;
        auto strides = get_strides_prop(op);
        bool are_strides_ones = std::all_of(strides.begin(), strides.end(), [](size_t s) -> bool {
            return s == 1;
        });
        if (!are_strides_ones) {
            auto conv = dynamic_cast<ngraph::opset7::Convolution*>(op.get_node());
            if (conv) {
                conv->set_strides(strides);
            } else {
                insert_pooling(op.get_source_output(), op, strides);
            }
        }
    }
}

ngraph::pass::ConvStridesPropagation::ConvStridesPropagation() {
    MATCHER_SCOPE(ConvStridesPropagation);
    auto data = pattern::any_input([](const Output<Node>& node) -> bool {
        const auto& shape = node.get_partial_shape();
        const auto& rank = shape.rank();
        if (rank.is_dynamic())
            return false;
        return std::all_of(shape.begin() + 2, shape.end(), [](const Dimension& dim) -> bool {
            return dim.is_static();
        });
    });
    auto weights = pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = pattern::wrap_type<opset7::Convolution>({data, weights});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());
        if (!conv)
            return false;

        auto conv_strides = conv->get_strides();
        Strides strides_ones(conv_strides.size(), 1);
        auto next_ops = op::util::get_node_target_inputs(conv);
        bool all_ops_are_valid;
        Strides strides;
        std::tie(strides, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            handle_not_equal_stride_props(std::move(next_ops));
        } else {
            std::transform(conv_strides.begin(),
                           conv_strides.end(),
                           strides.begin(),
                           conv_strides.begin(),
                           [](size_t s1, size_t s2) -> size_t {
                               return s1 * s2;
                           });
        }

        if (can_propagate_conv_stride(conv)) {
            conv->set_strides(strides_ones);
            auto conv_input = conv->input(0);
            insert_strides_prop(conv_input, conv_strides);
        } else {
            // Retain original padding
            // Make sure that setting strides does not change padding in cases when auto_pad is not EXPLICIT.
            // When padding type is not EXPLICIT, strides make a role to paddings calculation.
            // Change in padding, results in change in image position that filter is applied,
            // so we may end up with unwanted results after that.
            conv->set_auto_pad(op::PadType::EXPLICIT);
            conv->set_strides(conv_strides);
        }
        MATCHER_SCOPE_ENABLE(ConvStridesPropagation);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::SupportedNodesStridesPropagation::SupportedNodesStridesPropagation() {
    MATCHER_SCOPE(SupportedNodesStridesPropagation);
    auto root = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, op::util::BinaryElementwiseArithmetic>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto next_ops = op::util::get_node_target_inputs(node);
        bool all_ops_are_valid;
        Strides strides;
        std::tie(strides, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            return false;
        }

        for (auto& input : node->inputs()) {
            insert_strides_prop(input, strides);
        }
        MATCHER_SCOPE_ENABLE(SupportedNodesStridesPropagation);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::UnsupportedNodesStridesPropagation::UnsupportedNodesStridesPropagation() {
    MATCHER_SCOPE(UnsupportedNodesStridesPropagation);
    auto root = pattern::any_input();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto next_ops = op::util::get_node_target_inputs(node);
        handle_not_equal_stride_props(std::move(next_ops));
        MATCHER_SCOPE_ENABLE(UnsupportedNodesStridesPropagation);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}
