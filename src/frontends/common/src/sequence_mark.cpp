// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_mark.hpp"

namespace ov {
namespace frontend {

std::shared_ptr<Node> SequenceMark::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<SequenceMark>(inputs);
}

}  // namespace frontend
}  // namespace ov
