// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/model/data_contents/default_sw_weights_content.hpp>

#include <vector>
#include <memory>
#include <string>
#include <set>

namespace vpu {

namespace {

class FullyConnectedStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<FullyConnectedStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        orderInfo.setInput(inputEdge(0), input->desc().dimsOrder().createMovedDim(Dim::C, 0));
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 0));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
        auto weights = inputEdge(1)->input();

        auto swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
        if (swWeights == nullptr) {
            swWeights = model()->duplicateData(
                weights,
                "@SW",
                weights->desc(),
                std::make_shared<DefaultSwWeightsContent>(weights->content(), weights->desc()));

            weights->attrs().set<Data>("swWeights", swWeights);
        }

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

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        weights->serializeBuffer(serializer);
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
    VPU_PROFILE(swFullyConnectedAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubFullyConnected)
            continue;

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        model->disconnectStage(stage);

        model->addNewStage<FullyConnectedStage>(
            stage->name(),
            StageType::FC,
            stage->origLayer(),
            {input, weights},
            {output});

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

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swFullyConnectedAdaptation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

Stage StageBuilder::addSwFullyConnectedStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& weights,
        const Data& biases,
        const Data& scales,
        Data output) {
    auto fcWeights = model->duplicateData(
        weights,
        "@fc",
        DataDesc({
            input->desc().dim(Dim::W, 1) * input->desc().dim(Dim::H, 1),
            input->desc().dim(Dim::C),
            output->desc().dim(Dim::C)}));

    auto fcStage = model->addNewStage<FullyConnectedStage>(
        name,
        StageType::FC,
        layer,
        {input, fcWeights},
        {output});

    if (biases->usage() != DataUsage::Fake) {
        auto biasesInput = model->duplicateData(
            output,
            "@pre-bias");

        const auto outputProducerEdge = output->producerEdge();
        model->replaceStageOutput(outputProducerEdge, biasesInput);

        addBiasStage(
            model,
            name + "@biases",
            layer,
            biasesInput, biases,
            output);
    }

    if (scales->usage() != DataUsage::Fake) {
        auto scalesInput = model->duplicateData(
            output,
            "@pre-scaled");

        const auto outputProducerEdge = output->producerEdge();
        model->replaceStageOutput(outputProducerEdge, scalesInput);

        addScaleStage(
            model,
            name + "@scales",
            layer,
            scalesInput, scales,
            output);
    }

    return fcStage;
}

}  // namespace vpu
