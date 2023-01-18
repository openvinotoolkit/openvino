// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/store.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

snippets::op::Store::Store(const Output<Node>& x, const size_t count, const size_t offset) : MemoryAccess({x}, count, offset) {
    constructor_validate_and_infer_types();
}
std::shared_ptr<Node> snippets::op::Store::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Store_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Store>(new_args.at(0), m_count, m_offset);
}

} // namespace op
} // namespace snippets
} // namespace ngraph
