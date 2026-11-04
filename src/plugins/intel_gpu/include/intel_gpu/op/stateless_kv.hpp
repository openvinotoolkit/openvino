// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

#include <optional>

namespace ov::intel_gpu::op {

/// \brief Operator that implements Key-Values cache subgraph for large language models.
/// seq_len can be present or past length, to avoid extra computation or synchronization
class StatelessKV : public ov::op::Op {
public:
    OPENVINO_OP("StatelessKV", "gpu_opset");

    StatelessKV() = default;
    StatelessKV(const Output<Node>& past,
                const Output<Node>& new_token_data,
                const Output<Node>& seq_len,
                int64_t concat_axis,
                bool is_seq_len_present_len,
                int64_t window_size = 0);
    StatelessKV(const Output<Node>& past,
                const Output<Node>& new_token_data,
                const Output<Node>& seq_len,
                const Output<Node>& pos_idx,
                int64_t concat_axis,
                bool is_seq_len_present_len);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_concat_axis() const {
        return m_concat_axis;
    }
    void set_concat_axis(int64_t axis) {
        m_concat_axis = axis;
    }

    bool get_is_seq_len_present_len() const {
        return m_is_seq_len_present_len;
    }
    void set_is_seq_len_present_len(bool is_seq_len_present_len) {
        m_is_seq_len_present_len = is_seq_len_present_len;
    }

    int64_t get_window_size() const {
        return m_window_size;
    }
    void set_window_size(int64_t window_size) {
        m_window_size = window_size;
    }

    bool get_is_rolling() const {
        return m_is_rolling;
    }
    void set_is_rolling(bool is_rolling) {
        m_is_rolling = is_rolling;
    }

    std::optional<int64_t> get_update_offset() const {
        return m_update_offset;
    }
    void set_update_offset(std::optional<int64_t> update_offset) {
        m_update_offset = update_offset;
    }

    std::optional<int64_t> get_past_seq_len() const {
        return m_past_seq_len;
    }
    void set_past_seq_len(std::optional<int64_t> past_seq_len) {
        m_past_seq_len = past_seq_len;
    }

protected:
    StatelessKV(const OutputVector& inputs, int64_t concat_axis, bool is_seq_len_present_len, int64_t window_size);

    int64_t m_concat_axis = 0;
    std::optional<int64_t> m_update_offset;
    std::optional<int64_t> m_past_seq_len;
    bool m_is_seq_len_present_len = true;
    int64_t m_window_size = 0;
    bool m_is_rolling = false;
};

std::vector<ov::PartialShape> shape_infer(const StatelessKV* op, const std::vector<ov::PartialShape>& input_shapes);

}  // namespace ov::intel_gpu::op
