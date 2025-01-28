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
    OPENVINO_RTTI("SerializeControlFlow", "", SerializeBase)
    SerializeControlFlow(const std::string& xml_path, bool update_dynamic_ops = false) :
        SerializeBase(xml_path), m_update_dynamic_ops{update_dynamic_ops} {}
    bool run(const LinearIR& linear_ir) override;

private:
    const bool m_update_dynamic_ops = false;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
