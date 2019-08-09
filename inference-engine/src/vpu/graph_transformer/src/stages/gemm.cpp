//
// Copyright (C) 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<GEMMStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto inputDimsOrder0 = _inputEdges[0]->input()->desc().dimsOrder();
        auto inputDimsOrder1 = _inputEdges[1]->input()->desc().dimsOrder();
        auto inputDimsOrder2 = _inputEdges[2]->input()->desc().dimsOrder();
        auto outputDimsOrder = _outputEdges[0]->output()->desc().dimsOrder();

        if (inputDimsOrder0.numDims() >= 3) {
            inputDimsOrder0.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (inputDimsOrder1.numDims() >= 3) {
            inputDimsOrder1.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (inputDimsOrder2.numDims() >= 3) {
            inputDimsOrder2.moveDim(Dim::C, 2);  // ->...CHW
        }
        if (outputDimsOrder.numDims() >= 3) {
            outputDimsOrder.moveDim(Dim::C, 2);  // ->...CHW
        }

        _orderInfo.setInput(_inputEdges[0], inputDimsOrder0);
        _orderInfo.setInput(_inputEdges[1], inputDimsOrder1);
        _orderInfo.setInput(_inputEdges[2], inputDimsOrder2);
        _orderInfo.setOutput(_outputEdges[0], outputDimsOrder);
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }


    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto alpha = attrs().get<float>("alpha");
        auto beta = attrs().get<float>("beta");
        auto transposeA = attrs().get<bool>("transposeA");
        auto transposeB = attrs().get<bool>("transposeB");

        serializer.append(static_cast<float>(alpha));
        serializer.append(static_cast<float>(beta));
        serializer.append(static_cast<int>(transposeA));
        serializer.append(static_cast<int>(transposeB));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input1 = _inputEdges[0]->input();
        auto input2 = _inputEdges[1]->input();
        auto input3 = _inputEdges[2]->input();
        auto output = _outputEdges[0]->output();

        input1->serializeNewBuffer(serializer);
        input2->serializeNewBuffer(serializer);
        input3->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseGEMM(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 3);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::GemmLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    _stageBuilder->addGemmStage(
        model,
        layer->name,
        layer,
        layer->alpha,
        layer->beta,
        layer->transpose_a,
        layer->transpose_b,
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[0]);
}

Stage StageBuilder::addGemmStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const float alpha,
        const float beta,
        const bool transposeA,
        const bool transposeB,
        const Data& inputA,
        const Data& inputB,
        const Data& inputC,
        const Data& output) {
    auto stage = model->addNewStage<GEMMStage>(
        name,
        StageType::GEMM,
        layer,
        {inputA, inputB, inputC},
        {output});

    stage->attrs().set<float>("alpha", alpha);
    stage->attrs().set<float>("beta", beta);
    stage->attrs().set<bool>("transposeA", transposeA);
    stage->attrs().set<bool>("transposeB", transposeB);

    return stage;
}

}  // namespace vpu
