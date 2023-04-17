// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_data_types.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Remove Transpose/Gather layers connected to Inputs and create pre-processing model
 * to support input pre-processing on CPU.
 * Inserts Reshape layer instead of Transpose/Gater layer to avoid changing of the shapes.
 * @param output_subgraphs Map where pre-processing model for each input will be saved
 *
 * Searches for the following pattern
 *     Any input layer
 *           |
 *    Transpose/Gather
 *           |
 *        Any layer
 *
 *   and transforms it to
 *     Any input layer
 *           |
 *        Reshape
 *           |
 *        Any layer
 */
class RemoveInputsProcessing : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveInputsProcessing", "0");
    RemoveInputsProcessing(ov::intel_gna::PrePostProcessModels* input_subgraphs = nullptr)
        : m_input_subgraphs(input_subgraphs) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_gna::PrePostProcessModels* m_input_subgraphs;
};

/**
 * @brief Remove Transpose/Gather layers connected to Outputs and create post-processing model
 * to support output pre-processing on CPU.
 * Inserts Reshape layer instead of Transpose/Gather layer to avoid changing of the shapes.
 * @param output_subgraphs Map where post-processing model for each output will be saved
 *
 * Searches for the following pattern
 *     Any input layer
 *           |
 *    Transpose/Gather
 *           |
 *        Result
 *
 *   and transforms it to
 *     Any input layer
 *           |
 *        Reshape
 *           |
 *         Result
 */
class RemoveOutputsProcessing : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RemoveOutputsProcessing", "0");
    RemoveOutputsProcessing(ov::intel_gna::PrePostProcessModels* output_subgraphs = nullptr)
        : m_output_subgraphs(output_subgraphs) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::intel_gna::PrePostProcessModels* m_output_subgraphs;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
