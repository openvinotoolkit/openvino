// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface HorizonBase
 * @brief Base class for horizon operations.
 * @ingroup snippets
 */
class HorizonBase : public ov::op::Op {
public:
    OPENVINO_OP("HorizonBase", "SnippetsOpset");

    explicit HorizonBase(const Output<Node>& x);
    explicit HorizonBase(const OutputVector& x);
    HorizonBase() = default;

    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
