// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Store
 * @brief Generated during Lowering stage (convert_to_snippets_dialect) where explicit instructions should be emitted for data storing
 *        where number of elements to store is determined by "count" (Default value is "1" - to store one element)
 *        and memory offset for storing is determined by "offset" (Default value is "0" - to store starting at start memory ptr)
 * @ingroup snippets
 */
class Store : public modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("Store", "SnippetsOpset");

    Store(const Output<Node>& x, const size_t count = 1lu, const size_t offset = 0lu);
    Store() = default;

    size_t get_offset() const { return get_output_offset(0); }
    size_t get_count() const { return get_output_count(0); }

    void set_offset(size_t offset) { set_output_offset(offset, 0); }
    void set_count(size_t count) { set_output_count(count, 0); }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace snippets
} // namespace ov
