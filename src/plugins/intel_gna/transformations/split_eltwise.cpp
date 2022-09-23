// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/split_eltwise.hpp"

#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "legacy/ngraph_ops/eltwise.hpp"
#include "ops/util/util.hpp"
#include "backend/gna_limitations.hpp"
#include "layers/gna_split_layer.hpp"

using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::ngraph_util;

namespace {
inline bool is_eltwise_has_to_be_splitted(const ngraph::Output<ngraph::Node>& node) {
    auto eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node.get_node_shared_ptr());
    if (!eltwise) return false;
    auto o_dims = eltwise->get_output_shape(0);
    auto total_elem_size = std::accumulate(std::begin(o_dims), std::end(o_dims), 1, std::multiplies<size_t>());
    return (total_elem_size > GNAPluginNS::GNALimitations::bufferMaxSize);
}

std::shared_ptr<ngraph::opset9::VariadicSplit> split_input(const std::shared_ptr<ov::Node>& node,
    const std::pair<int64_t, std::vector<uint32_t>>& split_sizes_per_axis) {
    auto split = std::make_shared<ngraph::opset9::VariadicSplit>(node,
        ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{split_sizes_per_axis.first}),
        ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes_per_axis.second.size()}), split_sizes_per_axis.second));
    split->set_friendly_name(node->get_friendly_name() + "/split");
    ngraph::copy_runtime_info(node, split);
    return split;
}

std::shared_ptr<ngraph::op::Eltwise> create_eltwise(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<ov::Node>& split0,
    const std::shared_ptr<ov::Node>& split1, size_t index) {
    auto root_eltwise = std::dynamic_pointer_cast<ngraph::op::Eltwise>(node);
    auto eltwise = std::make_shared<ngraph::op::Eltwise>(split0->output(index), split1->output(index),
        root_eltwise->eltwise_type, root_eltwise->get_output_element_type(0));
    eltwise->set_friendly_name(root_eltwise->get_friendly_name() + "/partition" + std::to_string(index));
    ngraph::copy_runtime_info(root_eltwise, eltwise);
    return eltwise;
}
} // namespace

SplitEltwise::SplitEltwise() {
    MATCHER_SCOPE(SplitEltwise);
    auto eltwise = ngraph::pattern::wrap_type<ngraph::op::Eltwise>({ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        is_eltwise_has_to_be_splitted);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto eltwise_node = pattern_map.at(eltwise).get_node_shared_ptr();
        auto consumers = eltwise_node->output(0).get_target_inputs();
        auto o_dims = eltwise_node->get_output_shape(0);

        auto split_sizes_per_axis = GNAPluginNS::AlignedSplitSizesPerAxis(o_dims);
        if (0 == split_sizes_per_axis.second.size()) {
            gnalog() << "Splitting didn't succeed for layer " << eltwise_node->get_friendly_name()
            << " on axis " << split_sizes_per_axis.first << std::endl;
            return false;
        }

        auto split_node0 = split_input(eltwise_node->get_input_node_shared_ptr(0), split_sizes_per_axis);
        auto split_node1 = split_input(eltwise_node->get_input_node_shared_ptr(1), split_sizes_per_axis);

        ov::NodeVector concat_inputs;
        for (size_t i = 0; i < split_sizes_per_axis.second.size(); i++) {
            auto eltwise_node_part = create_eltwise(eltwise_node, split_node0, split_node1, i);
            concat_inputs.push_back(eltwise_node_part);
        }
        auto concat = std::make_shared<ngraph::opset9::Concat>(concat_inputs, split_sizes_per_axis.first);
        concat->set_friendly_name(eltwise_node->get_friendly_name());
        ngraph::copy_runtime_info(eltwise_node, concat);
        for (auto&& input : consumers) {
            input.replace_source_output(concat);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, matcher_name);
    this->register_matcher(m, callback);
}