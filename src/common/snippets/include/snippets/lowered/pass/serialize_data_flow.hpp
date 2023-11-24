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
 * @interface SerializeDataFlow
 * @brief Serializes data flow graph of LinearIR
 * @attention
 *  - This pass can not be run on the LinearIR after tail loop insertion.
 *  - Control flow operations (e.g. LoopBegin/LoopEnd) are not serialized
 * @ingroup snippets
 */
class SerializeDataFlow : public Pass {
public:
    OPENVINO_RTTI("SerializeDataFlow", "Pass")
    SerializeDataFlow(const std::string& xml_path, const std::string& bin_path = "");
    bool run(LinearIR& linear_ir) override;

private:
    const std::string m_xml_path;
    const std::string m_bin_path;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
