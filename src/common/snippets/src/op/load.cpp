// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/load.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

Load::Load(const Output<Node>& x, const size_t count) : MemoryAccess({x}, count) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Load::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Load);
    check_new_args_count(this, new_args);
    return std::make_shared<Load>(new_args.at(0), m_count);
}

}// namespace op
}// namespace snippets
}// namespace ngraph