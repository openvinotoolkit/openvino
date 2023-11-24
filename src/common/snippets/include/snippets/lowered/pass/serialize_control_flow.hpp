// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
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
class SerializeControlFlow : public Pass {
public:
    OPENVINO_RTTI("SerializeControlFlow", "Pass")
    SerializeControlFlow(const std::string& xml_path, const std::string& bin_path = "");
    bool run(LinearIR& linear_ir) override;

private:
    const std::string m_xml_path;
    const std::string m_bin_path;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
