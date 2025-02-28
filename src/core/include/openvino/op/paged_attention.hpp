// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v16 {
/// \brief PagedAttention operation is used as a placeholder op.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedAttention : public Op {
public:
    OPENVINO_OP("PagedAttention", "opset16");
    PagedAttention() = default;
    /**
     * @brief PagedAttention operation is used as a placeholder. It copies the tensor data to the output.
     */
    PagedAttention(const Output<Node>& query,
                   const Output<Node>& key,
                   const Output<Node>& value,
                   const Output<Node>& key_cache,
                   const Output<Node>& value_cache,
                   const Output<Node>& past_lens,
                   const Output<Node>& subsequence_begins,
                   const Output<Node>& block_indices,
                   const Output<Node>& block_indices_begins,
                   const Output<Node>& scale,
                   const Output<Node>& sliding_window,
                   const Output<Node>& alibi_slopes,
                   const Output<Node>& max_context_len);

    PagedAttention(const Output<Node>& query,
                   const Output<Node>& key,
                   const Output<Node>& value,
                   const Output<Node>& key_cache,
                   const Output<Node>& value_cache,
                   const Output<Node>& past_lens,
                   const Output<Node>& subsequence_begins,
                   const Output<Node>& block_indices,
                   const Output<Node>& block_indices_begins,
                   const Output<Node>& scale,
                   const Output<Node>& sliding_window,
                   const Output<Node>& alibi_slopes,
                   const Output<Node>& max_context_len,
                   const Output<Node>& rotated_block_indices,
                   const Output<Node>& rotation_deltas,
                   const Output<Node>& rotation_trig_lut);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const ov::element::Type get_out_type(int index) const;
    void set_out_type(int index, const ov::element::Type& output_type);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic};
};
}  // namespace v16
}  // namespace op
}  // namespace ov
