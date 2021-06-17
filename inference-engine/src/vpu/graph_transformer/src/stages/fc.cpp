// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>
#include "vpu/ngraph/operations/fully_connected.hpp"
#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>

namespace vpu {
// Rework logic
void FrontEnd::parseFullyConnected(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    const auto& env = CompileEnv::get();

    // IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);
    auto fc = ngraph::as_type_ptr<ngraph::vpu::op::FullyConnected>(node);
    IE_ASSERT(fc != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    auto total_out_num = fc->get_out_size() * output->desc().dim(Dim::N);
    if (total_out_num != output->desc().totalDimSize()) {
        VPU_THROW_EXCEPTION
                << "Layer Name: " << fc->get_friendly_name() << " Layer type: " << fc->get_type_name()
                << " has incorrect _out_num param. Expected: " << output->desc().totalDimSize()
                << " Actual: " << fc->get_out_size();
    }

    //
    // Check if HW is applicable
    //

    auto tryHW = env.config.compileConfig().hwOptimization;

    if (output->desc().dim(Dim::W, 1) != 1 || output->desc().dim(Dim::H, 1) != 1) {
        tryHW = false;
    }

    if (env.config.compileConfig().hwDisabled(fc->get_friendly_name())) {
        tryHW = false;
    }

    if (output->desc().totalDimSize() == 1) {
        tryHW = false;
    }

    //
    // Create const datas
    //
    const auto weightsNode = fc->input_value(1).get_node_shared_ptr();
    const auto biasesNode = (fc->get_input_size() > 2 ) ? node->input_value(2).get_node_shared_ptr() : NodePtr();
    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, fc->get_friendly_name(), weightsNode, biasesNode);

    IE_ASSERT(weights->desc().totalDimSize() >=
              input->desc().totalDimSize() / input->desc().dim(Dim::N, 1) * static_cast<int>(fc->get_out_size()));
    weights = model->duplicateData(
        weights,
        "@fc",
        DataDesc({
            input->desc().dim(Dim::W, 1) * input->desc().dim(Dim::H, 1),
            input->desc().dim(Dim::C),
            static_cast<int>(fc->get_out_size())}));

    if (biases->usage() != DataUsage::Fake) {
        IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
        biases = model->duplicateData(
            biases,
            "@fc",
            DataDesc({output->desc().dim(Dim::C)}));
    }

    //
    // Create stub stage
    //

    auto stage = model->addNewStage<StubStage>(
        fc->get_friendly_name(),
        StageType::StubFullyConnected,
        fc,
        {input, weights, biases, model->addFakeData()},
        {output});

    stage->attrs().set<bool>("tryHW", tryHW);
}

}  // namespace vpu
