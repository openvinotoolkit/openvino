// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <utility>

#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface EliminateBrgemmCopyB
 * @brief EliminateBrgemmCopyB identifies BrgemmCopyB nodes which can be inferred outside the Subgraph.
 * If this is possible, CopyB node is removed, and the external repacking is configured on the further pipeline stages
 * in RuntimeConfigurator.
 *
 * @ingroup snippets
 */
class EliminateBrgemmCopyB : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("EliminateBrgemmCopyB");
    explicit EliminateBrgemmCopyB(ov::intel_cpu::InputRepackerMap& input_repackers)
        : m_input_repackers(input_repackers) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_cpu::InputRepackerMap& m_input_repackers;
};

}  // namespace ov::intel_cpu::pass
