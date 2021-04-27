// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/common_optimizations/stride_optimization.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::StrideOptimization, "StrideOptimization", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvStridePropagation, "ConvStridePropagation", 0);

namespace ngraph {
template <>
class VariantWrapper<Strides> : public VariantImpl<Strides> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Strides", 0};
    const VariantTypeInfo& get_type_info() const override { return type_info; }
    VariantWrapper(const value_type& value)
        : VariantImpl<value_type>(value) {
    }
};

constexpr VariantTypeInfo VariantWrapper<Strides>::type_info;
} // namespace ngraph

static const char* STRIDES_PROP_KEY = "StridesProp";

static bool has_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    auto it = rt_map.find(STRIDES_PROP_KEY);
    return it != rt_map.end();
}

static ngraph::Strides get_strides_prop(const ngraph::Input<ngraph::Node>& node) {
    const auto& rt_map = node.get_rt_info();
    const auto& var = rt_map.at(STRIDES_PROP_KEY);
    return ngraph::as_type_ptr<ngraph::VariantWrapper<ngraph::Strides>>(var)->get();
}

static void insert_strides_prop(ngraph::Input<ngraph::Node> node, const ngraph::Strides& strides) {
    auto& rt_map = node.get_rt_info();
    rt_map[STRIDES_PROP_KEY] = std::make_shared<ngraph::VariantWrapper<ngraph::Strides>>(strides);
}

static bool can_propagate_conv_stride(const std::shared_ptr<ngraph::Node>& conv) {
    const auto& kernel_pshape = conv->input_value(1).get_partial_shape();
    if (kernel_pshape.is_dynamic())
        return false;
    const auto& kernel_shape = kernel_pshape.get_shape();
    return std::all_of(kernel_shape.begin() + 2, kernel_shape.end(), [] (size_t s) -> bool { return s == 1; });
}

static std::vector<ngraph::Input<ngraph::Node>> get_node_target_inputs(const std::shared_ptr<ngraph::Node>& node) {
    std::vector<ngraph::Input<ngraph::Node>> result;
    for (auto output : node->outputs()) {
        for (auto input : output.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

static std::tuple<std::vector<ngraph::Strides>, bool> check_next_ops(const std::vector<ngraph::Input<ngraph::Node>>& next_ops) {
    std::vector<ngraph::Strides> strides;
    for (const auto& op : next_ops) {
        if (has_strides_prop(op)) {
            strides.push_back(get_strides_prop(op));
        }
    }
    bool all_ops_are_valid = !(next_ops.size() != strides.size() || (strides.size() > 0 &&
                                                                     !std::all_of(strides.begin(), strides.end(),
                                                                         [&strides] (const ngraph::Strides& s) -> bool {
                                                                             bool all_ones = std::all_of(s.begin(), s.end(),
                                                                                                         [] (size_t i) -> bool { return i == 1; });
                                                                             return s == strides[0] && !all_ones;
                                                                         })));
    return std::make_tuple(strides, all_ops_are_valid);
}

static void insert_pooling(const std::shared_ptr<ngraph::Node>& first, ngraph::Input<ngraph::Node>& second, const ngraph::Strides& strides) {
    auto pool = std::make_shared<ngraph::opset7::MaxPool>(first, strides, ngraph::Shape{}, ngraph::Shape{}, ngraph::Shape(strides.size(), 1));
    second.replace_source_output(pool);
}

static void handle_not_equal_stride_props(const std::shared_ptr<ngraph::Node>& node, std::vector<ngraph::Input<ngraph::Node>>&& next_ops) {
    for (auto& op : next_ops) {
        if (!has_strides_prop(op))
            continue;
        auto strides = get_strides_prop(op);
        bool are_strides_ones = std::all_of(strides.begin(), strides.end(),
                                            [] (size_t s) -> bool { return s == 1; });
        if (!are_strides_ones) {
            auto conv = dynamic_cast<ngraph::opset7::Convolution*>(op.get_node());
            if (conv) {
                conv->set_strides(strides);
            } else {
                insert_pooling(node, op, strides);
            }
        }
    }
}

ngraph::pass::ConvStridePropagation::ConvStridePropagation() {
    MATCHER_SCOPE(ConvStridePropagation);
    auto data = pattern::any_input();
    auto weights = pattern::wrap_type<opset7::Constant>();
    auto conv_pattern = ngraph::pattern::wrap_type<opset7::Convolution>({data, weights});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<opset7::Convolution>(m.get_match_root());
        if (!conv)
            return false;

        auto conv_strides = conv->get_strides();
        Strides strides_ones(conv_strides.size(), 1);
        auto next_ops = get_node_target_inputs(conv);
        bool all_ops_are_valid;
        std::vector<Strides> strides_vec;
        std::tie(strides_vec, all_ops_are_valid) = check_next_ops(next_ops);

        if (!all_ops_are_valid) {
            handle_not_equal_stride_props(conv, std::move(next_ops));
        } else if (strides_vec[0].size() > 0) {
            std::transform(conv_strides.begin(), conv_strides.end(), strides_vec[0].begin(), conv_strides.begin(),
                    [] (size_t s1, size_t s2) -> size_t { return s1 * s2; });
        }

        if (can_propagate_conv_stride(conv)) {
            conv->set_strides(strides_ones);
            insert_strides_prop(conv->input(0), conv_strides);
        } else {
            conv->set_strides(conv_strides);
            insert_strides_prop(conv->input(0), strides_ones);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}


NGRAPH_RTTI_DEFINITION(ngraph::pass::StridePropagation, "StridePropagation", 0);

ngraph::pass::StridePropagation::StridePropagation() {
    MATCHER_SCOPE(StridePropagation);
    auto root = pattern::wrap_type<opset7::Parameter,
                                   opset7::Add,
                                   opset7::Relu,
                                   opset7::Maximum,
                                   opset7::Multiply>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        const auto& rank = node->get_output_partial_shape(0).rank();
        if (rank.is_dynamic() || rank.get_length() < 3)
            return false;

        auto next_ops = get_node_target_inputs(node);
        bool all_ops_are_valid;
        std::vector<Strides> strides_vec;
        std::tie(strides_vec, all_ops_are_valid) = check_next_ops(next_ops);
        Strides strides(static_cast<size_t>(rank.get_length()) - 2, 1);

        if (!all_ops_are_valid || is_type<opset7::Parameter>(node)) {
            handle_not_equal_stride_props(node, std::move(next_ops));
        } else if (strides_vec.size() > 0) {
            strides = strides_vec[0];
        }

        for (auto& input : node->inputs()) {
            insert_strides_prop(input, strides);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}
