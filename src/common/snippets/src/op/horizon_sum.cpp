// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/horizon_sum.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

std::shared_ptr<Node> HorizonSum::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(HorizonSum_clone_with_new_inputs);
    return std::make_shared<HorizonSum>(new_args);
}

}  // namespace ov::snippets::op
