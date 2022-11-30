// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface MemoryAccess
 * @brief This is a base class for memory access operations (like Load and Store).
 *        It provides universal set/get interface to manipulate the number
 *        of elements accessed during one operation call ("count").
 *        Default "count" value is "1" - it means to load/store one element
 * @ingroup snippets
 */

class MemoryAccess : public ngraph::op::Op {
public:
    OPENVINO_OP("MemoryAccess", "SnippetsOpset");

    size_t get_count() const;
    void set_count(size_t count);
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

protected:
    explicit MemoryAccess(const Output<Node>& x, size_t count = 1lu);
    MemoryAccess() = default;
    size_t m_count = 0lu;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
