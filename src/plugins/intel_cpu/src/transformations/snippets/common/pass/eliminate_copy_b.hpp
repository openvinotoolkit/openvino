// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <set>
#include <utility>

#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/model.hpp"
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
    EliminateCopyB(ov::intel_cpu::InputRepackerMap& input_repackers,
                   bool runtime_repacking_supported,
                   std::set<size_t> compile_time_repacking_idxs = {})
        : m_input_repackers(input_repackers),
          m_runtime_repacking_supported(runtime_repacking_supported),
          m_compile_time_repacking_idxs(std::move(compile_time_repacking_idxs)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool should_extract(size_t param_idx) const;

    [[maybe_unused]] ov::intel_cpu::InputRepackerMap& m_input_repackers;
    [[maybe_unused]] bool m_runtime_repacking_supported = false;
    [[maybe_unused]] std::set<size_t> m_compile_time_repacking_idxs;
};

}  // namespace ov::intel_cpu::pass
