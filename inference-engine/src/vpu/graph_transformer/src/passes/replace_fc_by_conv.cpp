// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;

void setConvParameters(const vpu::Stage& stage, int kX, int kY) {
    stage->attrs().set<int>("kernelSizeX", kX);
    stage->attrs().set<int>("kernelSizeY", kY);
    stage->attrs().set<int>("kernelStrideX", 1);
    stage->attrs().set<int>("kernelStrideY", 1);
    stage->attrs().set<int>("padLeft", 0);
    stage->attrs().set<int>("padRight", 0);
    stage->attrs().set<int>("padTop", 0);
    stage->attrs().set<int>("padBottom", 0);
    stage->attrs().set<int>("dilationX", 1);
    stage->attrs().set<int>("dilationY", 1);
    stage->attrs().set<int>("groupSize", 1);
    stage->attrs().set<bool>("tryHW", true);
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(replaceFCbyConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubFullyConnected) {
            continue;
        }

        auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases  = stage->input(2);
        auto output = stage->output(0);

        auto dims = input->desc().dims();

        if (input->desc().numDims() == 4) {
            bool required = dims.has(Dim::N);
            required &= dims.has(Dim::C);
            required &= dims.has(Dim::H);
            required &= dims.has(Dim::W);

            if (required &&
                input->desc().dim(Dim::H, 1) < 16 &&
                input->desc().dim(Dim::W, 1) < 16) {
                /* can convert to convolution layers */
                model->disconnectStageDatas(stage);

                auto kernelSizeX = input->desc().dim(Dim::W, 1);
                auto kernelSizeY = input->desc().dim(Dim::H, 1);
                IE_ASSERT(weights->desc().totalDimSize() >=
                        kernelSizeX * kernelSizeY * (input->desc().dim(Dim::C)) * output->desc().dim(Dim::C));

                auto newWeights = model->duplicateData(
                    weights,
                    "",
                    DataDesc({
                        kernelSizeX,
                        kernelSizeY,
                        input->desc().dim(Dim::C),
                        output->desc().dim(Dim::C)}));

                auto newBiases = model->addFakeData();
                if (biases->usage() != DataUsage::Fake) {
                    IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
                    newBiases = model->duplicateData(biases,
                        biases->name(),
                        DataDesc({output->desc().dim(Dim::C)}));
                }

                DataDesc newDesc({1, 1, output->desc().dim(Dim::C), output->desc().dim(Dim::N)});
                auto newOutput = model->duplicateData(output, "@reshapeData", newDesc);

                auto newStage = model->addNewStage<StubStage>(
                    stage->origLayerName(),
                    StageType::StubConv,
                    stage->origLayer(),
                    {input, newWeights, newBiases},
                    {newOutput});
                newStage->attrs().copyFrom(stage->attrs());
                setConvParameters(newStage, kernelSizeX, kernelSizeY);

                _stageBuilder->addReshapeStage(
                    model,
                    stage->name() + "@reshapeOut",
                    stage->origLayer(),
                    newOutput,
                    output);

                model->removeStage(stage);
            }
        } else if (dims.has(Dim::N) &&
                   dims.has(Dim::C) &&
                   (!dims.has(Dim::H)) &&
                   (!dims.has(Dim::W))) {
            IE_ASSERT(weights->desc().totalDimSize() >=
                    (input->desc().dim(Dim::C)) * output->desc().dim(Dim::C));

            model->disconnectStageDatas(stage);

            auto newWeights = model->duplicateData(weights,
                weights->name(),
                DataDesc({
                    1,
                    1,
                    input->desc().dim(Dim::C),
                    output->desc().dim(Dim::C)}));

            auto newBiases =  model->addFakeData();
            if (biases->usage() != DataUsage::Fake) {
                IE_ASSERT(biases->desc().totalDimSize() >= output->desc().dim(Dim::C));
                newBiases = model->duplicateData(biases,
                                                  biases->name(),
                                                  DataDesc({output->desc().dim(Dim::C)}));
            }

            DataDesc newDescIn({1, 1, input->desc().dim(Dim::C), input->desc().dim(Dim::N)});
            auto newInput = model->duplicateData(output, "@reshapeDataIn", newDescIn);

            DataDesc newDescOut({1, 1, output->desc().dim(Dim::C), output->desc().dim(Dim::N)});
            auto newOutput = model->duplicateData(output, "@reshapeDataOut", newDescOut);

            _stageBuilder->addReshapeStage(
                model,
                stage->name() + "@reshapeIn",
                stage->origLayer(),
                input,
                newInput);

            auto newStage = model->addNewStage<StubStage>(
                stage->origLayerName(),
                StageType::StubConv,
                stage->origLayer(),
                {newInput, newWeights, newBiases},
                {newOutput});
            newStage->attrs().copyFrom(stage->attrs());
            setConvParameters(newStage, 1, 1);

            _stageBuilder->addReshapeStage(
                model,
                stage->name() + "@reshapeOut",
                stage->origLayer(),
                newOutput,
                output);

            model->removeStage(stage);
        }
    }
}

}  // namespace

Pass::Ptr PassManager::replaceFCbyConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
