// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {
/**
 * Fused n-gram hash:
 *     out[..., h] = floor_mod(xor_i(token_i * multipliers[i]), moduli[h]) + offsets[h]
 * where the multiply/xor wrap around in int64 regardless of the tensor element type, so the result matches the
 * traced int64 graph even after the plugin's i64->i32 precision conversion.
 * Inputs: N token tensors of identical shape S and integer type T.
 * Output: shape S + [H] of type T, where H = moduli.size().
 */
class NgramHashNode : public ov::op::Op {
public:
    OPENVINO_OP("NgramHash", "cpu_plugin_opset");

    NgramHashNode() = default;
    NgramHashNode(const ov::OutputVector& tokens,
                  std::vector<int64_t> multipliers,
                  std::vector<int64_t> moduli,
                  std::vector<int64_t> offsets);
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const std::vector<int64_t>& get_multipliers() const {
        return m_multipliers;
    }
    const std::vector<int64_t>& get_moduli() const {
        return m_moduli;
    }
    const std::vector<int64_t>& get_offsets() const {
        return m_offsets;
    }

private:
    std::vector<int64_t> m_multipliers;
    std::vector<int64_t> m_moduli;
    std::vector<int64_t> m_offsets;
};
}  // namespace ov::intel_cpu
