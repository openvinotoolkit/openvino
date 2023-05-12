// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

#include "gna_data_types.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Remove Transpose/Gater layers connected to Inputs and create preprocessing model
 * to support input preprocessing on CPU
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *    Transpose/Gather
 *           |
 *        Any layer
 *
 * And transforms to
 *     Any input layer
 *           |
 *        Any layer
 */
class RemoveInputsProcessing : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveInputsProcessing", "0");
    RemoveInputsProcessing(ov::intel_gna::PrePostProcessModels* subgraph_cpu_map = nullptr)
        : m_subgraph_cpu_map(subgraph_cpu_map) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_gna::PrePostProcessModels* m_subgraph_cpu_map;
};

/**
 * @brief Remove Transpose/Gater layers connected to Outputs and create postprocessing model
 * to support input preprocessing on CPU
 *
 * Searches for next pattern
 *     Any input layer
 *           |
 *    Transpose/Gather
 *           |
 *        Result
 *
 * And transforms to
 *     Any input layer
 *           |
 *         Result
 */
class RemoveOutputsProcessing : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveOutputsProcessing", "0");
    RemoveOutputsProcessing(ov::intel_gna::PrePostProcessModels* subgraph_cpu_map = nullptr)
        : m_subgraph_cpu_map(subgraph_cpu_map) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_gna::PrePostProcessModels* m_subgraph_cpu_map;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
