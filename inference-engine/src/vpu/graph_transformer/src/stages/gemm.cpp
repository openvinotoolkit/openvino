// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <memory>

namespace vpu {

namespace {

class GEMMStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<GEMMStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto inputDimsOrder0 = inputEdge(0)->input()->desc().dimsOrder();
        auto inputDimsOrder1 = inputEdge(1)->input()->desc().dimsOrder();

        auto outputDimsOrder = outputEdge(0)->output()->desc().dimsOrder();

        if (inputDimsOrder0.numDims() >= 3) {
            inputDimsOrder0.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (inputDimsOrder1.numDims() >= 3) {
            inputDimsOrder1.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (outputDimsOrder.numDims() >= 3) {
            outputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }

        orderInfo.setInput(inputEdge(0), inputDimsOrder0);
        orderInfo.setInput(inputEdge(1), inputDimsOrder1);
        orderInfo.setOutput(outputEdge(0), outputDimsOrder);

        if (numInputs() == 3) {
            auto inputDimsOrder2 = inputEdge(2)->input()->desc().dimsOrder();
            if (inputDimsOrder2.numDims() >= 3) {
                inputDimsOrder2.moveDim(Dim::C, 2);  // ->...CHW
            }
            orderInfo.setInput(inputEdge(2), inputDimsOrder2);
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }


    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 2 || numInputs() == 3);
        IE_ASSERT(numOutputs() == 1);
        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto alpha = attrs().get<float>("alpha");
        auto beta = attrs().get<float>("beta");
        auto transposeA = attrs().get<bool>("transposeA");
        auto transposeB = attrs().get<bool>("transposeB");
        auto hasThreeInputs = numInputs() == 3;

        serializer.append(static_cast<float>(alpha));
        serializer.append(static_cast<float>(beta));
        serializer.append(static_cast<int>(hasThreeInputs));
        serializer.append(static_cast<int>(transposeA));
        serializer.append(static_cast<int>(transposeB));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        inputEdge(0)->input()->serializeBuffer(serializer);
        inputEdge(1)->input()->serializeBuffer(serializer);
        if (numInputs() == 3) {
            inputEdge(2)->input()->serializeBuffer(serializer);
        }
        outputEdge(0)->output()->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseGEMM(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::GemmLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addGemmStage(model, layer->name, layer, layer->alpha, layer->beta, layer->transpose_a, layer->transpose_b, inputs, outputs[0]);
}

Stage StageBuilder::addGemmStage(
        const Model& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const float alpha,
        const float beta,
        const bool transposeA,
        const bool transposeB,
        const DataVector& inputs,
        const Data& output) {
    auto stage = model->addNewStage<GEMMStage>(
        name,
        StageType::GEMM,
        layer,
        inputs,
        {output});

    stage->attrs().set<float>("alpha", alpha);
    stage->attrs().set<float>("beta", beta);
    stage->attrs().set<bool>("transposeA", transposeA);
    stage->attrs().set<bool>("transposeB", transposeB);

    return stage;
}

}  // namespace vpu
