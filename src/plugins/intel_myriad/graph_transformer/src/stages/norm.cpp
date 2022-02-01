// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

#include <precision_utils.h>

namespace vpu {

namespace {

class LRNStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<LRNStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();

        // LRN supports both HWC and CHW orders, but requires that input and output have the same stride

        auto reqs = StridesRequirement::compact();
        if (type() == StageType::LRN &&
            input->desc().dimsOrder().dimInd(Dim::C) != 0) {
            reqs.add(1, DimStride::Aligned);
        }

        stridesInfo.setInput(inputEdge(0), reqs);
        stridesInfo.setOutput(outputEdge(0), reqs);
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto size = attrs().get<int>("size");
        auto k = attrs().get<int>("k");
        auto alpha = attrs().get<float>("alpha");
        auto beta = attrs().get<float>("beta");

        serializer.append(static_cast<uint32_t>(size));
        serializer.append(ie::PrecisionUtils::f32tof16(static_cast<float>(k))); // why float?
        serializer.append(ie::PrecisionUtils::f32tof16(alpha));
        serializer.append(ie::PrecisionUtils::f32tof16(beta));
        serializer.append(ie::PrecisionUtils::f32tof16(0));  // for alignment
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseNorm(const Model& model, const ie::CNNLayerPtr& _layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::NormLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto stage = model->addNewStage<LRNStage>(layer->name, layer->_isAcrossMaps ? StageType::LRN : StageType::InnerLRN, layer, inputs, outputs);
    stage->attrs().set<int>("size", layer->_size);
    stage->attrs().set<int>("k", layer->_k);
    stage->attrs().set<float>("alpha", layer->_alpha);
    stage->attrs().set<float>("beta", layer->_beta);
}

}  // namespace vpu
