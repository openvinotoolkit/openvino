#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern::op {

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

private:
    OutputVector m_inputs;
    OutputVector m_outputs;
};

}  // namespace ov::pass::pattern::op