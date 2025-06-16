// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "cache/multi_cache.h"
#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/rtti.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @class InitRepackedConstantInputs
 * @brief The pass initializes InputRepackerMap of Brgemms' constant inputs which
 *        should be repacked at model compilation stage
 */
class InitRepackedConstantInputs : public ov::snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("InitRepackedConstantInputs", "", ov::snippets::lowered::pass::ConstPass)
    InitRepackedConstantInputs(ov::intel_cpu::MultiCacheWeakPtr cache,
                               ov::intel_cpu::InputRepackerMap& repacked_const_inputs_config)
        : m_cache(std::move(cache)),
          m_repacked_const_inputs_config(repacked_const_inputs_config) {}

    bool run(const snippets::lowered::LinearIR& linear_ir) override;

private:
    ov::intel_cpu::MultiCacheWeakPtr m_cache;
    ov::intel_cpu::InputRepackerMap& m_repacked_const_inputs_config;
};

}  // namespace ov::intel_cpu::pass
