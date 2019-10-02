// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <vector>
#include <memory>
#include <string>
#include <set>

#include <vpu/sw/utility.hpp>

namespace vpu {

namespace {

class FullyConnectedStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<FullyConnectedStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep,
            StageDataInfo<float>&) override {
        VPU_THROW_EXCEPTION << "Must never be called";
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
                std::make_shared<DefaultSwWeightsContent>(weights->content()));

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
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto biases = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(this, serializer);

        if (output->desc().dimsOrder() == DimsOrder::NC) {
            IE_ASSERT(output->desc().dim(Dim::N) == 1);

            output->serializeOldBuffer(
                this,
                serializer,
                DimsOrder::HWC,
                {
                    {Dim::W, {Dim::N}},
                    {Dim::C, {Dim::C}}
                });
        } else {
            output->serializeOldBuffer(this, serializer);
        }

        weights->serializeOldBuffer(this, serializer);

        // TODO: remove this
        biases->serializeOldBuffer(this, serializer);
    }
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(swFullyConnectedAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubFullyConnected)
            continue;

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto output = stage->output(0);

        model->disconnectStage(stage);

        if (biases->usage() != DataUsage::Fake) {
            auto tempOutput = model->duplicateData(
                output,
                "@temp");

            _stageBuilder->addBiasStage(
                model,
                stage->name() + "@biases",
                stage->origLayer(),
                tempOutput, biases,
                output);

            output = tempOutput;
        }

        model->addNewStage<FullyConnectedStage>(
            stage->name(),
            StageType::FC,
            stage->origLayer(),
            {input, weights, biases},
            {output});

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swFullyConnectedAdaptation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

Stage StageBuilder::addSwFullyConnectedStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const Data& input,
        const Data& weights,
        const Data& biases,
        Data output) {
    auto fcWeights = model->duplicateData(
        weights,
        "@fc",
        DataDesc({
            input->desc().dim(Dim::W, 1) * input->desc().dim(Dim::H, 1),
            input->desc().dim(Dim::C),
            output->desc().dim(Dim::C)}));

    if (biases->usage() != DataUsage::Fake) {
        auto tempOutput = model->duplicateData(
            output,
            "@temp");

        addBiasStage(
            model,
            name + "@biases",
            layer,
            tempOutput, biases,
            output);

        output = tempOutput;
    }

    auto fcStage = model->addNewStage<FullyConnectedStage>(
        name,
        StageType::FC,
        layer,
        {input, fcWeights, biases},
        {output});

    return fcStage;
}

}  // namespace vpu
