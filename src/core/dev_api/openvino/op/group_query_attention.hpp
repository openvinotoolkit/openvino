// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
class OPENVINO_API GroupQueryAttention : public ov::op::Op {
public:
    OPENVINO_OP("GroupQueryAttention");

    // helpers to make and detect inputs that are not used
    static Output<Node> null();
    static bool is_null(const Output<Node> output);

    GroupQueryAttention(const ov::OutputVector& args,
                        unsigned int num_heads,
                        unsigned int kv_num_heads,
                        float scale,
                        bool do_rotary,
                        bool rotary_interleaved);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    // TODO: implement visit_attributes

    unsigned int get_num_heads() const {
        return m_num_heads;
    }
    unsigned int get_kv_num_heads() const {
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
    unsigned int m_num_heads;
    unsigned int m_kv_num_heads;
    float m_scale = 0;
    bool m_do_rotary = false;
    bool m_rotary_interleaved = false;
};

// // NPUW friendly decomposition
// OPENVINO_API ov::OutputVector group_query_attention_decomposition(std::shared_ptr<GroupQueryAttention> gqa);

}  // namespace op
}  // namespace ov
