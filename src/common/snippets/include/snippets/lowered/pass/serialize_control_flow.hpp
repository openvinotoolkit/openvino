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
    SerializeControlFlow(const std::string& xml_path) : SerializeBase(xml_path) {}
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
