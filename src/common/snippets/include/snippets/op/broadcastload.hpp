// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <snippets/op/memory_access.hpp>

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface BroadcastLoad
 * @brief Is generated for broadcasting by least varying dimension for non-blocked cases and the second varying dimension for blocked
 * @ingroup snippets
 */
class BroadcastLoad : public MemoryAccess {
public:
    OPENVINO_OP("BroadcastLoad", "SnippetsOpset", ov::snippets::op::MemoryAccess);

    BroadcastLoad(const Output<Node>& x, ov::PartialShape output_shape, size_t offset = 0lu);
    BroadcastLoad() = default;

    size_t get_offset() const { return get_input_offset(0); }

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    ov::PartialShape output_shape;
};

} // namespace op
} // namespace snippets
} // namespace ov
