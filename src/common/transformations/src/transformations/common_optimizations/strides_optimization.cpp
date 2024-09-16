// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/strides_optimization.hpp"

#include <numeric>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/strides_property.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;

static bool can_propagate_conv_stride(const std::shared_ptr<ov::Node>& conv) {
    const auto& kernel_shape = conv->input_value(1).get_shape();
    return std::all_of(kernel_shape.begin() + 2, kernel_shape.end(), [](size_t s) -> bool {
        return s == 1;
    });
}

static std::tuple<ov::Strides, bool> check_next_ops(const std::vector<ov::Input<ov::Node>>& next_ops) {
    std::vector<ov::Strides> strides;
    for (const auto& op : next_ops) {
        if (!has_strides_prop(op)) {
            return std::make_tuple(ov::Strides{}, false);
        }
        strides.push_back(get_strides_prop(op));
    }
    bool all_ops_are_valid = std::all_of(strides.begin(), strides.end(), [&strides](const ov::Strides& s) -> bool {
        bool all_ones = std::all_of(s.begin(), s.end(), [](size_t i) -> bool {
            return i == 1;
        });
        return s == strides[0] && !all_ones;
    });
    return std::make_tuple(strides[0], all_ops_are_valid);
}

static void insert_pooling(const Output<Node>& first, Input<Node>& second, const Strides& strides) {
    pass::NodeRegistry rg;
    auto first_node = first.get_node_shared_ptr();
    const auto rank = first.get_partial_shape().rank();
    const bool do_reshape = rank.is_static() && static_cast<size_t>(rank.get_length()) < strides.size() + 2;
    if (do_reshape) {
        const size_t diff = strides.size() + 2 - static_cast<size_t>(rank.get_length());
        const auto ones = rg.make<ov::op::v0::Constant>(element::i64, Shape{diff}, vector<int64_t>(diff, 1));
        const auto current_shape = rg.make<ov::op::v3::ShapeOf>(first);
        shared_ptr<Node> new_shape = rg.make<ov::op::v0::Concat>(OutputVector{ones, current_shape}, 0);
        if (const auto constant_new_shape = ov::util::get_constant_from_source(new_shape)) {
            rg.add(constant_new_shape);
            new_shape = constant_new_shape;
        }
        first_node = rg.make<ov::op::v1::Reshape>(first_node, new_shape, false);
    }
    shared_ptr<Node> new_node =
        rg.make<ov::op::v1::MaxPool>(first_node, strides, Shape{}, Shape{}, Shape(strides.size(), 1));
    if (do_reshape) {
        // squeeze dimensions back
        const size_t diff = strides.size() + 2 - static_cast<size_t>(rank.get_length());
        vector<size_t> axes(diff);
        iota(axes.begin(), axes.end(), 0);
        new_node =
            rg.make<ov::op::v0::Squeeze>(new_node, rg.make<ov::op::v0::Constant>(element::u64, Shape{diff}, axes));
    }
    if (const auto constant_new_node = ov::util::get_constant_from_source(new_node)) {
        rg.add(constant_new_node);
        new_node = constant_new_node;
    }

    copy_runtime_info(as_node_vector({second.get_source_output()}), rg.get());
    second.replace_source_output(new_node);
}

static void handle_not_equal_stride_props(std::vector<ov::Input<ov::Node>>& next_ops) {
    for (auto& op : next_ops) {
        if (!has_strides_prop(op))
            continue;
        auto strides = get_strides_prop(op);
        bool are_strides_ones = std::all_of(strides.begin(), strides.end(), [](size_t s) -> bool {
            return s == 1;
        });
        if (!are_strides_ones) {
            auto conv = ov::as_type<ov::op::v1::Convolution>(op.get_node());
            if (conv) {
                conv->set_strides(strides);
            } else {
                insert_pooling(op.get_source_output(), op, strides);
            }
        }
    }
}

static void remove_strides_property_from_nodes(std::vector<ov::Input<ov::Node>>& nodes) {
    for (auto& node : nodes) {
        remove_strides_prop(node);
    }
}

ov::pass::ConvStridesPropagation::ConvStridesPropagation() {
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
    auto conv_pattern = pattern::wrap_type<ov::op::v1::Convolution>({data, weights});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(m.get_match_root());
        if (!conv)
            return false;

        auto conv_strides = conv->get_strides();
        Strides strides_ones(conv_strides.size(), 1);
        auto next_ops = op::util::get_node_target_inputs(conv);
        bool all_ops_are_valid;
        Strides strides;
        std::tie(strides, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            handle_not_equal_stride_props(next_ops);
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

        remove_strides_property_from_nodes(next_ops);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::SupportedNodesStridesPropagation::SupportedNodesStridesPropagation() {
    MATCHER_SCOPE(SupportedNodesStridesPropagation);
    auto root = pattern::wrap_type<op::util::UnaryElementwiseArithmetic, op::util::BinaryElementwiseArithmetic>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
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

        remove_strides_property_from_nodes(next_ops);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::UnsupportedNodesStridesPropagation::UnsupportedNodesStridesPropagation() {
    MATCHER_SCOPE(UnsupportedNodesStridesPropagation);
    auto root = pattern::any_input();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto next_ops = op::util::get_node_target_inputs(node);
        handle_not_equal_stride_props(next_ops);
        remove_strides_property_from_nodes(next_ops);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::StridesOptimization::StridesOptimization() {
    using namespace ov::pass;
    ADD_MATCHER_FOR_THIS(ConvStridesPropagation);
    ADD_MATCHER_FOR_THIS(SupportedNodesStridesPropagation);
    ADD_MATCHER_FOR_THIS(UnsupportedNodesStridesPropagation);
}
