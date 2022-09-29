// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "snippets/op/memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Store
 * @brief Generated during Lowering stage (convert_to_snippets_dialect) where explicit instructions should be emitted for data storing
 *        where number of elements to store is determined by "count" (Default value is "1" - to store one element)
 *        and memory offset for storing is determined by "offset" (Default value is "0" - to store starting at start memory ptr)
 * @ingroup snippets
 */
class Store : public MemoryAccess {
public:
    OPENVINO_OP("Store", "SnippetsOpset");

    Store(const Output<Node>& x, const size_t count = 1lu, const size_t offset = 0lu);
    Store() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
