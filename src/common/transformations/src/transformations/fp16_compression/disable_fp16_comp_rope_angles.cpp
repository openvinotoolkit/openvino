// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/disable_fp16_comp_rope_angles.hpp"

#include <algorithm>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;

namespace {

// one RoPE angle chain to keep in f32: the Sin or Cos, and the nodes from the position Convert to its input
struct AngleChain {
    std::shared_ptr<ov::Node> trig;
    std::vector<std::shared_ptr<ov::Node>> nodes;
};

bool has_consumer_of_type(const ov::Output<ov::Node>& output, const ov::DiscreteTypeInfo& type) {
    const auto& targets = output.get_target_inputs();
    return std::any_of(targets.begin(), targets.end(), [&](const ov::Input<ov::Node>& input) {
        return input.get_node()->get_type_info() == type;
    });
}

// every consumer of the chain nodes before Sin/Cos belongs to a chain, or reads only the shape
bool consumers_within(const AngleChain& chain, const std::unordered_set<ov::Node*>& chain_nodes) {
    return std::all_of(chain.nodes.begin(), chain.nodes.end(), [&](const std::shared_ptr<ov::Node>& node) {
        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                const auto consumer = input.get_node();
                if (chain_nodes.count(consumer) == 0 && !ov::is_type<ov::op::util::ShapeOfBase>(consumer)) {
                    return false;
                }
            }
        }
        return true;
    });
}

bool insert_boundary(const std::shared_ptr<ov::Node>& trig) {
    std::vector<ov::Input<ov::Node>> consumers;
    for (const auto& input : trig->output(0).get_target_inputs()) {
        const auto consumer = input.get_node()->shared_from_this();
        if (ov::is_type_any_of<v0::Result, v0::Convert>(consumer) ||
            ov::is_conversion_disabled(consumer, ov::element::f32, ov::element::f16)) {
            continue;
        }
        consumers.push_back(input);
    }
    if (consumers.empty()) {
        return false;
    }
    // stops the downward propagation of the marks, see the class description
    const auto boundary = std::make_shared<v0::Convert>(trig, ov::element::f32);
    boundary->set_friendly_name(trig->get_friendly_name() + "/rope_angle_boundary");
    ov::copy_runtime_info(trig, boundary);
    for (auto& input : consumers) {
        input.replace_source_output(boundary);
    }
    return true;
}

}  // namespace

bool ov::pass::DisableFP16CompForRopeAngles::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(DisableFP16CompForRopeAngles);
    using namespace ov::pass::pattern;

    auto position_source = any_input([](const ov::Output<ov::Node>& output) {
        return output.get_element_type().is_integral_number() && !ov::is_type<v0::Constant>(output.get_node());
    });
    auto position_convert = wrap_type<v0::Convert>({position_source}, type_matches(element::f32));
    auto position_broadcast = optional<v3::Broadcast>({position_convert, any_input()});
    auto positions = optional<v1::Reshape, v0::Unsqueeze>({position_broadcast, any_input()});
    auto frequencies = any_input(type_matches(element::f32));
    // positions are the second MatMul input in HF exports, the first one with torch.outer
    auto matmul_freq_first = wrap_type<v0::MatMul>({frequencies, positions});
    auto matmul_freq_second = wrap_type<v0::MatMul>({positions, frequencies});
    auto multiply = wrap_type<v1::Multiply>({positions, frequencies});
    auto angle = matmul_freq_first | matmul_freq_second | multiply;
    auto angle_reshape = optional<v1::Reshape>({angle, any_input()});
    auto angle_transpose = optional<v1::Transpose>({angle_reshape, any_input()});
    // the angles duplicated by torch.cat((freqs, freqs), dim=-1)
    auto concat = wrap_type<v0::Concat>({angle_transpose, angle_transpose}, [](const ov::Output<ov::Node>& output) {
        const auto node = output.get_node();
        return node->input_value(0) == node->input_value(1);
    });
    auto table = angle_transpose | concat;
    auto trig = wrap_type<v0::Sin, v0::Cos>({table}, type_matches(element::f32));
    auto matcher = std::make_shared<Matcher>(trig, "DisableFP16CompForRopeAngles");

    bool is_changed = false;
    std::vector<AngleChain> chains;
    for (const auto& node : model->get_ordered_ops()) {
        // the bodies of If, Loop and TensorIterator are separate models, ConvertPrecision lowers them too
        if (const auto multi_subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            if (multi_subgraph->get_transformations_allowed()) {
                for (size_t i = 0; i < multi_subgraph->get_internal_subgraphs_size(); ++i) {
                    is_changed |= run_on_model(multi_subgraph->get_function(i));
                }
            }
        }
        if (!ov::is_type_any_of<v0::Sin, v0::Cos>(node) || !matcher->match(node->output(0)) ||
            transformation_callback(node)) {
            continue;
        }
        const auto table_output = node->input_value(0);
        if (!has_consumer_of_type(table_output, v0::Sin::get_type_info_static()) ||
            !has_consumer_of_type(table_output, v0::Cos::get_type_info_static())) {
            continue;
        }
        const auto& pattern_map = matcher->get_pattern_value_map();
        AngleChain chain{node, {}};
        // absent optionals and the alternatives of the angle operation that did not match are not bound
        for (const auto& label : {position_convert,
                                  position_broadcast,
                                  positions,
                                  matmul_freq_first,
                                  matmul_freq_second,
                                  multiply,
                                  angle_reshape,
                                  angle_transpose,
                                  concat}) {
            if (pattern_map.count(label)) {
                chain.nodes.push_back(pattern_map.at(label).get_node_shared_ptr());
            }
        }
        chains.push_back(std::move(chain));
    }

    // Drop the chains whose nodes have consumers outside of the matched chains: marking them would drag those
    // consumers into f32. Dropping a chain can invalidate another one sharing nodes with it, so iterate.
    bool dropped = true;
    while (dropped) {
        std::unordered_set<ov::Node*> chain_nodes;
        for (const auto& chain : chains) {
            chain_nodes.insert(chain.trig.get());
            for (const auto& chain_node : chain.nodes) {
                chain_nodes.insert(chain_node.get());
            }
        }
        const auto size = chains.size();
        chains.erase(std::remove_if(chains.begin(),
                                    chains.end(),
                                    [&](const AngleChain& chain) {
                                        return !consumers_within(chain, chain_nodes);
                                    }),
                     chains.end());
        dropped = chains.size() != size;
    }

    for (const auto& chain : chains) {
        for (const auto& chain_node : chain.nodes) {
            if (!ov::is_conversion_disabled(chain_node, element::f16)) {
                ov::disable_conversion(chain_node, element::f16);
                is_changed = true;
            }
        }
        if (!ov::is_conversion_disabled(chain.trig, element::f16)) {
            ov::disable_conversion(chain.trig, element::f16);
            is_changed = true;
        }
        is_changed |= insert_boundary(chain.trig);
    }
    return is_changed;
}
