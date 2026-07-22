// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {

class TRANSFORMATIONS_API GatherMatmul : public ov::op::Op {
public:
    OPENVINO_OP("GatherMatmul");

    GatherMatmul() = default;

    GatherMatmul(const ov::Output<Node>& A,
                 const ov::Output<Node>& B,
                 const ov::Output<Node>& indices,
                 const ov::Output<Node>& bias);

    GatherMatmul(const ov::Output<Node>& A, const ov::Output<Node>& B, const ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

private:
    // the weights matrix B is expected to have the transposed form [goup, N, K]
    static constexpr bool transp_a = false;
    static constexpr bool transp_b = true;
};

}  // namespace ov::op::internal
