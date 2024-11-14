// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "serialize_base.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SerializeControlFlow
 * @brief Serializes control flow graph of LinearIR
 * @ingroup snippets
 */
class SerializeControlFlow : public SerializeBase {
public:
    OPENVINO_RTTI("SerializeControlFlow", "Pass", SerializeBase)
    SerializeControlFlow(const std::string& xml_path, bool update_dynamic_ops = false) :
        SerializeBase(xml_path), m_update_dynamic_ops{update_dynamic_ops} {}

    bool run(LinearIR& linear_ir) override {
        return run(const_cast<const LinearIR&>(linear_ir));
    }
    // We need a const method to run from functions that can't change LIR
    bool run(const LinearIR& linear_ir);

private:
    const bool m_update_dynamic_ops = false;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
