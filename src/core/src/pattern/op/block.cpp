
#include "openvino/pass/pattern/op/block.hpp"

#include "openvino/core/rt_info.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::pattern::op;

Block::Block(const OutputVector& inputs, const OutputVector& outputs, const std::string& name)
    : Pattern(inputs, op::Predicate{}),
      m_inputs(inputs),
      m_outputs(outputs) {
    set_output_type(0, element::dynamic, PartialShape::dynamic());
    set_friendly_name(name);
}

bool Block::match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) {
    auto block_pattern_root = m_outputs.front();

    // Using a local matcher to match only those patterns that are encapsulated into the current Block.
    auto local_matcher = std::make_shared<Matcher>(block_pattern_root.get_node_shared_ptr(), "BlockMatcher");
    if (!local_matcher->match_value(block_pattern_root, graph_value)) {
        return false;
    }

    auto& local_pm = local_matcher->get_pattern_value_map();

    OutputVector real_inputs, real_outputs;
    for (const auto& input : m_inputs) {
        if (local_pm.count(input.get_node_shared_ptr())) {
            real_inputs.push_back(local_pm.at(input.get_node_shared_ptr()));
        }
    }

    for (const auto& output : m_outputs) {
        if (local_pm.count(output.get_node_shared_ptr())) {
            real_outputs.push_back(local_pm.at(output.get_node_shared_ptr()));
        }
    }

    // Creating a mapping Pattern Block -> Graph Block.
    // Pattern Block contains inputs, outputs from the pattern graph
    // Graph Block   contains inputs, outputs from the real graph (ov::Model)
    auto matched_block = std::make_shared<Block>(real_inputs, real_outputs, get_friendly_name());

    // Merge the local_matcher state to the external matcher.
    auto& pattern_map = matcher->get_pattern_value_map();
    pattern_map[shared_from_this()] = matched_block->output(0);
    pattern_map.merge(local_pm);

    for (const auto& matched_node : local_matcher->get_matched_nodes()) {
        matcher->add_node(matched_node);
    }
    return true;
}