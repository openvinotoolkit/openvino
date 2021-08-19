// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/model/data_contents/deconvolution_contents.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <set>

namespace vpu {

namespace {

class DeconvStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<DeconvStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        auto finalOrder = input->desc().dimsOrder();
        if (finalOrder.dimInd(Dim::C) == 1) {
            // HCW -> CHW
            finalOrder.moveDim(Dim::C, 2);
        }

        if (finalOrder != input->desc().dimsOrder()) {
            orderInfo.setInput(inputEdge(0), finalOrder);
        }
        orderInfo.setOutput(outputEdge(0), finalOrder);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        auto finalOrder = input->desc().dimsOrder();
        if (finalOrder.dimInd(Dim::C) == 1) {
            // HCW -> CHW
            finalOrder.moveDim(Dim::C, 2);
        }

        if (type() == StageType::DepthDeconv) {
            if (finalOrder.dimInd(Dim::C) == 0) {
                // HWC
                stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
                stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
            }
        } else {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");

        Data swWeights;

        if (type() == StageType::DepthDeconv) {
            if (input->desc().dimsOrder().dimInd(Dim::C) == 0) {
                //
                // HWC case
                //

                swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
                if (swWeights == nullptr) {
                    DataDesc newWeightsDesc({
                        kernelSizeX * kernelSizeY,
                        1,
                        output->desc().dim(Dim::C)});

                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<DepthDeconvolutionHWCWeightsContent>(
                            weights->content(),
                            kernelSizeX, kernelSizeY,
                            output->desc().dim(Dim::C)));

                    weights->attrs().set<Data>("swWeights", swWeights);
                }
            } else if (input->desc().dimsOrder().dimInd(Dim::C) == 2) {
                //
                // CHW case
                //

                swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
                if (swWeights == nullptr) {
                    DataDesc newWeightsDesc({
                        kernelSizeX * kernelSizeY,
                        1,
                        output->desc().dim(Dim::C)});

                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<DepthDeconvolutionCHWWeightsContent>(
                            weights->content(),
                            kernelSizeX, kernelSizeY,
                            output->desc().dim(Dim::C)));

                    weights->attrs().set<Data>("swWeights", swWeights);
                }
            }
        } else {
            swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
            if (swWeights == nullptr) {
                DataDesc newWeightsDesc({
                    kernelSizeX * kernelSizeY,
                    input->desc().dim(Dim::C),
                    output->desc().dim(Dim::C)});

                swWeights = model()->duplicateData(
                    weights,
                    "@SW",
                    newWeightsDesc,
                    std::make_shared<DeconvolutionWeightsContent>(
                        weights->content(),
                        newWeightsDesc,
                        kernelSizeX, kernelSizeY,
                        input->desc().dim(Dim::C),
                        output->desc().dim(Dim::C)));

                weights->attrs().set<Data>("swWeights", swWeights);
            }
        }

        IE_ASSERT(swWeights != nullptr);

        model()->replaceStageInput(inputEdge(1), swWeights);
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void finalCheckImpl() const override {
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");
        auto kernelStrideX = attrs().get<int>("kernelStrideX");
        auto kernelStrideY = attrs().get<int>("kernelStrideY");
        auto padLeft = attrs().get<int>("padLeft");
        auto padTop = attrs().get<int>("padTop");

        serializer.append(static_cast<uint32_t>(kernelSizeX));
        serializer.append(static_cast<uint32_t>(kernelSizeY));
        serializer.append(static_cast<uint32_t>(kernelStrideX));
        serializer.append(static_cast<uint32_t>(kernelStrideY));
        serializer.append(static_cast<uint32_t>(padLeft));
        serializer.append(static_cast<uint32_t>(padTop));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        weights->serializeBuffer(serializer);

        if (numTempBuffers() == 1) {
            tempBuffer(0)->serializeBuffer(serializer);
        }
    }
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(swDeconvAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubDeconv)
            continue;

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto groupSize = stage->attrs().get<int>("groupSize");

        model->disconnectStage(stage);

        if (groupSize == 0 ||
            (groupSize > input->desc().dim(Dim::C)) ||
            (input->desc().dim(Dim::C) % groupSize != 0) ||
            (groupSize > output->desc().dim(Dim::C)) ||
            (output->desc().dim(Dim::C) % groupSize != 0)) {
            VPU_THROW_EXCEPTION << "DeconvolutionLayer has invalid group value";
        }

        if (groupSize == 1) {
            auto swStage = model->addNewStage<DeconvStage>(
                stage->name(),
                StageType::Deconvolution,
                stage->origLayer(),
                {input, weights},
                {output});

            swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

            swStage->attrs().set<int>("padLeft", padLeft);
            swStage->attrs().set<int>("padRight", padRight);
            swStage->attrs().set<int>("padTop", padTop);
            swStage->attrs().set<int>("padBottom", padBottom);

            if (biases->usage() != DataUsage::Fake) {
                auto biasesInput = model->duplicateData(
                    output,
                    "@pre-bias");

                const auto outputProducerEdge = output->producerEdge();
                model->replaceStageOutput(outputProducerEdge, biasesInput);

                _stageBuilder->addBiasStage(
                    model,
                    stage->name() + "@biases",
                    stage->origLayer(),
                    biasesInput, biases,
                    output);
            }

            if (scales->usage() != DataUsage::Fake) {
                auto scalesInput = model->duplicateData(
                    output,
                    "@pre-scaled");

                const auto outputProducerEdge = output->producerEdge();
                model->replaceStageOutput(outputProducerEdge, scalesInput);

                _stageBuilder->addScaleStage(
                    model,
                    stage->name() + "@scales",
                    stage->origLayer(),
                    scalesInput, scales,
                    output);
            }
        } else if (groupSize == input->desc().dim(Dim::C) &&
                   groupSize == output->desc().dim(Dim::C)) {
            auto swStage = model->addNewStage<DeconvStage>(
                stage->name(),
                StageType::DepthDeconv,
                stage->origLayer(),
                {input, weights},
                {output});

            swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

            swStage->attrs().set<int>("padLeft", padLeft);
            swStage->attrs().set<int>("padRight", padRight);
            swStage->attrs().set<int>("padTop", padTop);
            swStage->attrs().set<int>("padBottom", padBottom);

            if (biases->usage() != DataUsage::Fake) {
                auto biasesInput = model->duplicateData(
                    output,
                    "@pre-bias");

                const auto outputProducerEdge = output->producerEdge();
                model->replaceStageOutput(outputProducerEdge, biasesInput);

                _stageBuilder->addBiasStage(
                    model,
                    stage->name() + "@biases",
                    stage->origLayer(),
                    biasesInput, biases,
                    output);
            }

            if (scales->usage() != DataUsage::Fake) {
                auto scalesInput = model->duplicateData(
                    output,
                    "@pre-scaled");

                const auto outputProducerEdge = output->producerEdge();
                model->replaceStageOutput(outputProducerEdge, scalesInput);

                _stageBuilder->addScaleStage(
                    model,
                    stage->name() + "@scales",
                    stage->origLayer(),
                    scalesInput, scales,
                    output);
            }
        } else {
            VPU_THROW_EXCEPTION << "Internal error : grouped deconvolution was not processed";
        }

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swDeconvAdaptation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
