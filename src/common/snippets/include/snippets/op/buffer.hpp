// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief The operation is for intermediate data storage
 *        Notes:
 *               - All buffers in a graph have the same memory pointer. So if we have a few buffers,
 *                 each buffer should have its own offset for common memory
 *               - If Buffer is an input for operation output, this Buffer should be a single consumer for this port
 * @ingroup snippets
 */
class Buffer : public ngraph::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    BWDCMP_RTTI_DECLARATION;

    Buffer(const Output<Node>& x);
    Buffer() = default;

    size_t get_offset() const { return m_offset; }
    void set_offset(const size_t offset);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

private:
    size_t m_offset = 0lu;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
