// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/rtti.hpp"
#include "serialize_base.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface SerializeDataFlow
 * @brief Serializes data flow graph of LinearIR
 * @attention - This pass can not be run on the LinearIR after tail loop insertion.
 * @attention - Control flow operations (e.g. LoopBegin/LoopEnd) are not serialized
 * @ingroup snippets
 */
class SerializeDataFlow : public SerializeBase {
public:
    OPENVINO_RTTI("SerializeDataFlow", "", SerializeBase)
    explicit SerializeDataFlow(const std::string& xml_path) : SerializeBase(xml_path) {}
    bool run(const LinearIR& linear_ir) override;
};

}  // namespace ov::snippets::lowered::pass
