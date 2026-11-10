// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/horizon_max.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/itt.hpp"

namespace ov::snippets::op {

std::shared_ptr<Node> HorizonMax::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(HorizonMax_clone_with_new_inputs);
    return std::make_shared<HorizonMax>(new_args);
}

}  // namespace ov::snippets::op
