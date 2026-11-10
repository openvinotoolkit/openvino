// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/op/horizon.hpp"

namespace ov::snippets::op {

/**
 * @interface HorizonSum
 * @brief The operation calculates a horizon sum of a vector register
 * @ingroup snippets
 */
class HorizonSum : public HorizonBase {
public:
    OPENVINO_OP("HorizonSum", "SnippetsOpset", HorizonBase);

    using HorizonBase::HorizonBase;
    HorizonSum() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::snippets::op
