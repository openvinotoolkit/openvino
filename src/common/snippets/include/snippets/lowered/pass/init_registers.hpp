// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass_config.hpp"
#include "snippets/lowered/reg_manager.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface InitRegisters
 * @brief This pass combines all register-related transformations that are needed to initialize register info.
 * @ingroup snippets
 */
class InitRegisters : public Pass {
public:
    OPENVINO_RTTI("InitRegisters", "0", Pass)
    InitRegisters(const std::shared_ptr<const Generator>& generator, const std::shared_ptr<PassConfig>& pass_config);
    bool run(LinearIR& linear_ir) override;

private:
    lowered::RegManager m_reg_manager;
    const std::shared_ptr<PassConfig>& m_pass_config;
};

}  // namespace ov::snippets::lowered::pass
