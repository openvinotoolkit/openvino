#pragma once

#include <optional>

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block_util.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace {

// _MAKE_ANCHOR is an internal macro for REGISTER_ANCHORS that is not supposed to used separately.
#define _MAKE_ANCHOR(x) block->register_anchor(#x, x);

}  // namespace

namespace ov::pass::pattern::op {

/**
 * @brief A utility macro to register named anchors in a Block.
 *
 * This macro simplifies the process of registering multiple anchors in a Block.
 * It expands to a series of calls to `block->register_anchor(...)` for each provided argument.
 *
 */

#define REGISTER_ANCHORS(block, ...)        \
    do {                                    \
        FOR_EACH(_MAKE_ANCHOR, __VA_ARGS__) \
    } while (0)

/**
 * @brief Block is a reusable subgraph pattern composed of named inputs and outputs.
 *
 * It wraps a group of connected pattern nodes and allows treating them as a single unit.
 * Block uses a local Matcher internally and merges its results into the main matcher context.
 *
 * Example:
 *
 *                Input (x)
 *                   │
 *                   ▼
 *           ┌────────────────────┐
 *           │     Block:         │
 *           │   "norm_block"     │
 *           │                    │
 *           │  NormalizeL2       │
 *           │       │            │
 *           │       ▼            │
 *           │     Multiply       |
 *           └────────────────────┘
 *                   │
 *                 Output
 *
 * Block inputs/outputs are exposed for use in higher-level patterns.
 */
class OPENVINO_API Block : public Pattern {
public:
    OPENVINO_RTTI("Block");

    Block(const OutputVector& inputs, const OutputVector& outputs, const std::string& name = "");

    bool match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) override;

    const OutputVector& get_inputs() const {
        return m_inputs;
    }

    const OutputVector& get_outputs() const {
        return m_outputs;
    }

    void register_anchor(const std::string& name, const ov::Output<ov::Node>& output) {
        m_named_anchors[name] = output;
    }

    std::optional<Output<Node>> get_anchor(const std::string& name, const PatternValueMap& pm) const {
        if (m_named_anchors.empty()) {
            return std::nullopt;
        }
        auto it = m_named_anchors.find(name);
        if (it == m_named_anchors.end()) {
            return std::nullopt;
        }

        auto pattern_node = it->second.get_node_shared_ptr();
        auto matched_it = pm.find(pattern_node);
        if (matched_it == pm.end()) {
            return std::nullopt;
        }

        return matched_it->second;
    }

    std::map<std::string, Output<Node>>& get_registered_anchors() {
        return m_named_anchors;
    }

private:
    OutputVector m_inputs;
    OutputVector m_outputs;

    std::map<std::string, Output<Node>> m_named_anchors;
};

}  // namespace ov::pass::pattern::op