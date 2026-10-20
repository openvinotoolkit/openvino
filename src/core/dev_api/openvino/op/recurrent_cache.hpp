// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \brief Reset and gather recurrent matrix-state rows without modifying the source cache.
///
/// Inputs, in order:
/// 0. cache: f16/f32 [1,1,S,F], where S is the slot count and F the flattened state size.
/// 1. active_rows: i32/i64 [1,1,1,B], selecting B source rows for the recurrent computation.
/// 2. extra_rows: i32/i64 [1,1,1,N], a shared source-row map whose selected suffix is reordered.
/// 3. reset_begin: i32/i64 [1,1,1,1], first source-cache row to clear.
/// 4. reset_length: i32/i64 [1,1,1,1], number of source-cache rows to clear.
/// 5. extra_offsets: i32/i64 [1,1,1,2], (destination_begin, source_begin) in extra_rows.
///
/// Output 0 is active state [1,1,B,F]. Output 1 is the full reset/reordered cache [1,1,S,F].
/// Both outputs read the original cache after logically clearing the reset range. Output 0
/// does not read the reordered output; cyclic maps and duplicate source rows are supported.
/// Unselected output-cache rows retain their reset-cache values. An empty selected suffix
/// performs no reorder. Metadata values and all source/destination ranges must be in bounds.
/// Positive key_size/value_size transpose each active [head,value,key] matrix to
/// [head,key,value]; output 1 retains cache layout. Both attributes default to zero (no transpose).
/// GPU kernel input and output allocations must be distinct; the runtime may stage a caller's
/// shared remote input/output tensor to maintain immutable kernel reads.
class OPENVINO_API RecurrentStateGather : public ov::op::Op {
public:
    OPENVINO_OP("RecurrentStateGather");
    RecurrentStateGather() = default;
    RecurrentStateGather(const OutputVector& args, int64_t key_size = 0, int64_t value_size = 0);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override;
    int64_t get_key_size() const {
        return m_key_size;
    }
    int64_t get_value_size() const {
        return m_value_size;
    }
    void set_matrix_shape(int64_t key_size, int64_t value_size) {
        m_key_size = key_size;
        m_value_size = value_size;
    }

private:
    int64_t m_key_size = 0, m_value_size = 0;
};
/// \brief Write recurrent final states or snapshot planes into a prepared full cache.
///
/// Inputs: cache f16/f32 [1,1,S,F], states f16/f32 [1,1,N,F], and i32/i64 control [1,1,1,4].
/// Control is (destination_begin, sequence_count, snapshot_stride, source_begin).
/// Source row source_begin+r writes destination_begin+(r/sequence_count)*snapshot_stride
/// +(r%sequence_count). Later source rows win when destination planes overlap.
/// Output 0 is the full updated cache [1,1,S,F] in cache precision; other rows are preserved.
/// sequence_count must be positive, snapshot_stride nonnegative, source_begin in [0,N],
/// and all selected destination rows in bounds. source_begin=N selects an empty update.
/// Positive key_size/value_size transpose source [head,key,value] matrices to cache
/// [head,value,key]; both attributes default to zero (no transpose). F must be a multiple
/// of key_size*value_size. Source rows convert to cache precision during the write.
/// Kernel source/output allocations must be distinct, including for overlapping updates.
class OPENVINO_API RecurrentStateUpdate : public ov::op::Op {
public:
    OPENVINO_OP("RecurrentStateUpdate");
    RecurrentStateUpdate() = default;
    RecurrentStateUpdate(const OutputVector& args, int64_t key_size = 0, int64_t value_size = 0);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override;
    int64_t get_key_size() const {
        return m_key_size;
    }
    int64_t get_value_size() const {
        return m_value_size;
    }

private:
    int64_t m_key_size = 0, m_value_size = 0;
};
/// \brief Prepare convolution history and write ordered history windows into recurrent cache.
///
/// Inputs 0-5 have the RecurrentStateGather contract. Input 6 is f16/f32 QKV [1,B,C,T],
/// and input 7 is i32/i64 windows [1,1,K,2]. Positive history H is required; F=C*H.
/// Output 0 is [1,B,C,H+T] in QKV precision: reset active history concatenated with QKV.
/// Each (start,destination_row) window copies H columns of output 0 into B cache rows.
/// Starts must lie in [0,T]; destination ranges must lie in [0,S]. Later overlapping
/// windows win over earlier windows, which in turn win over the reset/reordered cache.
/// Output 1 is the final full cache [1,1,S,F] in cache precision. Window writes convert
/// through output-0 precision before cache precision, matching the materialized input.
/// Empty window lists are permitted; padded/short batches must supply valid-token windows.
/// The convolution arithmetic itself is outside this operation. Kernel inputs and outputs
/// must use distinct allocations, as for RecurrentStateGather.
class OPENVINO_API RecurrentConvCache : public ov::op::Op {
public:
    OPENVINO_OP("RecurrentConvCache");
    RecurrentConvCache() = default;
    RecurrentConvCache(const OutputVector& args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override;
    int64_t get_history() const {
        return m_history;
    }

private:
    int64_t m_history = 0;
};
}  // namespace ov::op::internal
