// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief MoEScatterReduce — GPU-internal op that combines the expert-sorted rows of a
/// Mixture-of-Experts block back into token order, weighted by the router weights.
///
///   output[t, :] = sum_k weights[t, k] * data[row_lut[t * top_k + k], :]
///
/// The rows are read as by Gather-8: a negative index counts from the end of data, an out-of-range index reads zeros.
///
/// Inputs:
///   0: data     [rows, hidden]        — expert outputs, e.g. sorted by expert
///   1: weights  [num_tokens, top_k]   — router weights
///   2: row_lut  [num_tokens * top_k]  — i32 or i64 row of data of each flat (token, k) position
///
/// Output:
///   0: [num_tokens, hidden], element type of data
///
class MoEScatterReduce : public ov::op::Op {
public:
    OPENVINO_OP("MoEScatterReduce", "gpu_opset");

    MoEScatterReduce() = default;
    MoEScatterReduce(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& weights, const ov::Output<ov::Node>& row_lut);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
