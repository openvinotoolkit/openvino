// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
/// \brief Implements the SDPA (Scaled Dot Product Attention) operator for specific ViT models like Qwen2-VL and
/// Qwen2.5-VL. These models exhibit distinct attention mask sparsity patterns where:
///   - Attention occurs only within individual images (for multi-image inputs)
///   - Attention is confined to individual windows (in Qwen2.5-VL)
/// \note The key difference from standard scaled_dot_product_attention is mask handling:
///       This implementation uses cu_seqlens instead of attention_mask.
class TRANSFORMATIONS_API VLSDPA : public ov::op::Op {
public:
    OPENVINO_OP("VLSDPA", "ie_internal_opset", ov::op::Op);

    VLSDPA() = default;

    VLSDPA(const OutputVector& inputs,
           const std::vector<int64_t>& order_q = {},
           const std::vector<int64_t>& order_k = {},
           const std::vector<int64_t>& order_v = {},
           const std::vector<int64_t>& order_out = {});

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::vector<int64_t> get_input0_transpose_order() const {
        return m_order_q;
    }
    std::vector<int64_t> get_input1_transpose_order() const {
        return m_order_k;
    }
    std::vector<int64_t> get_input2_transpose_order() const {
        return m_order_v;
    }
    std::vector<int64_t> get_output_transpose_order() const {
        return m_order_out;
    }

protected:
    std::vector<int64_t> m_order_q;
    std::vector<int64_t> m_order_k;
    std::vector<int64_t> m_order_v;
    std::vector<int64_t> m_order_out;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
