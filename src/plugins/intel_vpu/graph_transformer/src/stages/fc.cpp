// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>

#include <vpu/utils/hw_disabled.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>

namespace vpu {

void FrontEnd::parseFullyConnected(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    const auto& env = CompileEnv::get();

    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::FullyConnectedLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    auto total_out_num = layer->_out_num * output->desc().dim(Dim::N);
    if (total_out_num != output->desc().totalDimSize()) {
        VPU_THROW_EXCEPTION
                << "Layer Name: " << layer->name << " Layer type: " << layer->type
                << " has incorrect _out_num param. Expected: " << output->desc().totalDimSize()
                << " Actual: " << layer->_out_num;
    }

    //
    // Check if HW is applicable
    //

    auto tryHW = env.config.get<HwAccelerationOption>();

    if (output->desc().dim(Dim::W, 1) != 1 || output->desc().dim(Dim::H, 1) != 1) {
        tryHW = false;
    }

    if (HwDisabled(env.config, layer->name)) {
        tryHW = false;
    }

    if (output->desc().totalDimSize() == 1) {
        tryHW = false;
    }

    //
    // Create const datas
    //

    Data weights, biases;
    std::tie(weights, biases) = getWeightsAndBiases(model, layer);

    IE_ASSERT(weights->desc().totalDimSize() >=
              input->desc().totalDimSize() / input->desc().dim(Dim::N, 1) * static_cast<int>(layer->_out_num));
    weights = model->duplicateData(
        weights,
        "@fc",
        DataDesc({
            input->desc().dim(Dim::W, 1) * input->desc().dim(Dim::H, 1),
            input->desc().dim(Dim::C),
            static_cast<int>(layer->_out_num)}));

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
        layer->name,
        StageType::StubFullyConnected,
        layer,
        {input, weights, biases, model->addFakeData()},
        {output});

    stage->attrs().set<bool>("tryHW", tryHW);
}

}  // namespace vpu
