// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {

/// @brief Internal op mirroring ov::op::v17::GroupedMatMul with extra inputs
///        carrying the weight decompression scale (and optional zero-point).
///
///  Case 1 — 2D x 3D form (with offsets):
///    0 : mat_a       (2D)
///    1 : mat_b       (compressed weight constant, u4/u8/i4/i8/...)
///    2 : offsets     (1D)
///    3 : decompression scale
///    4 : (optional) decompression zero-point
///
///  Case 2 — 3D x 3D form (no offsets, batched uniform):
///    0 : mat_a       (3D)
///    1 : mat_b       (compressed weight constant, u4/u8/i4/i8/...)
///    2 : decompression scale
///    3 : (optional) decompression zero-point
///
class TRANSFORMATIONS_API GroupedMatMulCompressed : public ov::op::v17::GroupedMatMul {
public:
    OPENVINO_OP("GroupedMatMulCompressed", "", ov::op::v17::GroupedMatMul);

    GroupedMatMulCompressed() = default;

    // 2D x 3D constructors (with offsets)
    GroupedMatMulCompressed(const ov::Output<Node>& mat_a,
                            const ov::Output<Node>& mat_b,
                            const ov::Output<Node>& offsets,
                            const ov::Output<Node>& decompression_scale);

    GroupedMatMulCompressed(const ov::Output<Node>& mat_a,
                            const ov::Output<Node>& mat_b,
                            const ov::Output<Node>& offsets,
                            const ov::Output<Node>& decompression_scale,
                            const ov::Output<Node>& decompression_zero_point);

    // 3D x 3D factories (no offsets). Exposed as static functions to avoid a
    // 4-argument constructor overload collision with the 2D x 3D form above.
    static std::shared_ptr<GroupedMatMulCompressed> make_3d(const ov::Output<Node>& mat_a,
                                                            const ov::Output<Node>& mat_b,
                                                            const ov::Output<Node>& decompression_scale);

    static std::shared_ptr<GroupedMatMulCompressed> make_3d(const ov::Output<Node>& mat_a,
                                                            const ov::Output<Node>& mat_b,
                                                            const ov::Output<Node>& decompression_scale,
                                                            const ov::Output<Node>& decompression_zero_point);

    /// @brief True if the op carries an offsets input (2D x 3D form).
    bool has_offsets() const;

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
