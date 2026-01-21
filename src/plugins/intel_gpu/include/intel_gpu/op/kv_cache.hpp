// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace ov::intel_gpu::op {

/// \brief Operator that implements Key-Values cache subgraph for large language models.
/// This operation updates data of the corresponding Variable
class KVCache : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("KVCache", "gpu_opset");

    KVCache() = default;

    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const Output<Node>& beam_idx,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            int64_t gather_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    /// KVCache with seq_len trimming
    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const Output<Node>& past_seq_len,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    /// KVCache with seq_len trimming and beam_idx
    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const Output<Node>& beam_idx,
            const Output<Node>& past_seq_len,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            int64_t gather_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    /// KVCache with update&trimming for tree-based speculative decoding
    KVCache(const Output<Node>& past,
            const Output<Node>& new_token_data,
            const Output<Node>& past_seq_len,
            const Output<Node>& dst_idx,
            const Output<Node>& update_data,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            int64_t concat_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    int64_t get_concat_axis() const { return m_concat_axis; }
    int64_t get_gather_axis() const { return m_gather_axis; }

    void set_concat_axis(int64_t axis) { m_concat_axis = axis; }
    void set_gather_axis(int64_t axis) { m_gather_axis = axis; }

    bool get_indirect() const { return m_indirect; }
    bool get_trim() const { return m_trim; }
    bool get_update_kv() const { return m_update_kv; }
    
    void set_trim(bool trim) { m_trim = trim; }
    void set_update_kv(bool update_kv) { m_update_kv = update_kv; }

    int64_t get_trim_length() const { return m_trim_length; }
    void set_trim_length(int64_t trim_length) { m_trim_length = trim_length; }

protected:
    KVCache(const OutputVector& inputs,
            const std::shared_ptr<ov::op::util::Variable>& past_values,
            bool indirect,
            bool trim,
            int64_t concat_axis,
            int64_t gather_axis,
            const ov::element::Type output_type = ov::element::dynamic);

    int64_t m_concat_axis = 0;
    int64_t m_gather_axis = 0;
    bool m_indirect = false;
    bool m_trim = false;
    bool m_update_kv = false;
    int64_t m_trim_length = 0;

    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const KVCache* op, const std::vector<ov::PartialShape>& input_shapes);

}   // namespace ov::intel_gpu::op
