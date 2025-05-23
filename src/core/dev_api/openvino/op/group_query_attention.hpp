// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {

// This is an experimental operation that is implemented in the plugins.
class OPENVINO_API GroupQueryAttention : public Op {
public:
    OPENVINO_OP("GroupQueryAttention");

    GroupQueryAttention() = default;
    GroupQueryAttention(const ov::OutputVector& args,
                        int64_t num_heads,
                        int64_t kv_num_heads,
                        float scale,
                        bool do_rotary,
                        bool rotary_interleaved);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_num_heads() const {
        return m_num_heads;
    }
    int64_t get_kv_num_heads() const {
        return m_kv_num_heads;
    }
    float get_scale() const {
        return m_scale;
    }
    bool get_do_rotary() const {
        return m_do_rotary;
    }
    bool get_rotary_interleaved() const {
        return m_rotary_interleaved;
    }

private:
    int64_t m_num_heads = 0;
    int64_t m_kv_num_heads = 0;
    float m_scale = 0;
    bool m_do_rotary = false;
    bool m_rotary_interleaved = false;
};

}  // namespace ov::op::internal
