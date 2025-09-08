// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface Fill
 * @brief Generated in Tail Loop vector representation in code generation step for cases when we should
 *        refill registers by special values.
 *        For example, for cases with ReduceMax or ReduceSum in Softmax
 *        Where:
 *          - offset - starting element index where filling is performed while beginning of input data is untouched
 *          - fill_value - hexadecimal filling value
 * @ingroup snippets
 */
class Fill : public ov::op::Op {
public:
    OPENVINO_OP("Fill", "SnippetsOpset");

    explicit Fill(const Output<Node>& x, size_t offset, uint32_t fill_value = 0x0);
    Fill() = default;

    size_t get_offset() const {
        return m_offset;
    }
    uint32_t get_fill_value() const {
        return m_fill_value;
    }

    void set_offset(const size_t offset) {
        m_offset = offset;
    }
    void set_fill_value(const uint32_t fill_value) {
        m_fill_value = fill_value;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;

protected:
    size_t m_offset = 0LU;
    uint32_t m_fill_value = 0x0;
};

}  // namespace ov::snippets::op
