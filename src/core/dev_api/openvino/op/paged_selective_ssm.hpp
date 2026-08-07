// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note PagedSelectiveSSM op class is under development and subject to change
///
/// \brief Operator performing paged SelectiveSSM computation for continuous batching.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedSelectiveSSM : public ov::op::Op {
public:
    OPENVINO_OP("PagedSelectiveSSM");

    PagedSelectiveSSM() = default;
    PagedSelectiveSSM(const Output<Node>& A,
                      const Output<Node>& dt,
                      const Output<Node>& B,
                      const Output<Node>& x,
                      const Output<Node>& C,
                      const Output<Node>& recurrent_state_table,
                      const Output<Node>& subsequence_begins,
                      const Output<Node>& la_block_indices,
                      const Output<Node>& la_block_indices_begins,
                      const Output<Node>& num_processed_tokens,
                      const Output<Node>& cache_interval);
    explicit PagedSelectiveSSM(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
