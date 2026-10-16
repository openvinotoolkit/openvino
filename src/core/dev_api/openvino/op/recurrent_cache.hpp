// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/op/op.hpp"

namespace ov::op::internal {
// Immutable cache [1,1,slots,features]. All indices/offsets are runtime i32/i64 tensors.
// Gather inputs: cache, active source rows, extra source rows, reset begin, reset length,
// extra offsets [destination begin, source begin]. Extra source rows can be
// a nonempty shared row map; the source offset selects its suffix, including empty suffixes.
// Outputs: active rows and reset/reordered cache. Row indices and offsets must be in bounds.
// Positive key_size/value_size transpose each active matrix from [value,key] to [key,value];
// the prepared full cache keeps its original layout.
// Inputs and outputs must not alias: reorder maps may contain cycles and duplicate sources.
class OPENVINO_API RecurrentStateGather : public ov::op::Op {
  public:
    OPENVINO_OP("RecurrentStateGather");
    RecurrentStateGather() = default;
    RecurrentStateGather(const OutputVector &args, int64_t key_size = 0, int64_t value_size = 0);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &args) const override;
    int64_t get_key_size() const { return m_key_size; }
    int64_t get_value_size() const { return m_value_size; }
    void set_matrix_shape(int64_t key_size, int64_t value_size) {
        m_key_size = key_size;
        m_value_size = value_size;
    }

  private:
    int64_t m_key_size = 0, m_value_size = 0;
};
// Immutable cache and flattened state rows [1,1,N,F]; control [1,1,1,4]
// is (destination begin, sequence count, snapshot stride, source row begin).
// Output is the full cache. Later source rows win if destination planes overlap.
// Sequence count must be positive, stride nonnegative, and source begin in [0,N].
// All selected destination rows must be in bounds. Kernel source/output allocations must not alias.
// Positive matrix dimensions convert source [head,key,value] to cache [head,value,key].
class OPENVINO_API RecurrentStateUpdate : public ov::op::Op {
  public:
    OPENVINO_OP("RecurrentStateUpdate");
    RecurrentStateUpdate() = default;
    RecurrentStateUpdate(const OutputVector &args, int64_t key_size = 0, int64_t value_size = 0);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &args) const override;
    int64_t get_key_size() const { return m_key_size; }
    int64_t get_value_size() const { return m_value_size; }

  private:
    int64_t m_key_size = 0, m_value_size = 0;
};
// Conv inputs: same six inputs, new QKV [1,batch,channels,tokens], windows [1,1,K,2].
// Each window is (start column in history+tokens, destination row); writes batch rows.
// Starts must lie in [0,tokens]; caller supplies valid-token windows for padded/short batches.
// Windows are applied in order, so the last overlapping write wins.
// Outputs: convolution input [1,batch,channels,history+tokens] and final cache.
class OPENVINO_API RecurrentConvCache : public ov::op::Op {
  public:
    OPENVINO_OP("RecurrentConvCache");
    RecurrentConvCache() = default;
    RecurrentConvCache(const OutputVector &args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector &args) const override;
    int64_t get_history() const { return m_history; }

  private:
    int64_t m_history = 0;
};
} // namespace ov::op::internal
