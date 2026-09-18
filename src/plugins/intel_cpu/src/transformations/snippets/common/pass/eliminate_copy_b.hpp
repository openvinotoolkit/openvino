// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface EliminateCopyB
 * @brief Identifies CopyB nodes which can be inferred outside the Subgraph.
 * If this is possible, CopyB node is removed, and the external repacking is configured on the further pipeline stages.
 *
 * @ingroup snippets
 */
class EliminateCopyB : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("EliminateCopyB");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

protected:
    explicit EliminateCopyB(ov::intel_cpu::InputRepackerMap& input_repackers) : m_input_repackers(input_repackers) {}

private:
    [[nodiscard]] virtual std::shared_ptr<ov::Node> get_copy_b_pattern(
        const std::shared_ptr<ov::Node>& input) const = 0;
    [[nodiscard]] virtual bool is_supported_copy_b(const std::shared_ptr<ov::Node>& node) const = 0;

    ov::intel_cpu::InputRepackerMap& m_input_repackers;
};

}  // namespace ov::intel_cpu::pass
