// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>

#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/stages/stub_stage.hpp>

namespace vpu {

namespace {

class StubPriorBoxStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StubPriorBoxStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder> &orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement> &stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport> & /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 2);
        IE_ASSERT(numOutputs() == 1);

        assertInputsOutputsTypes(this,
                                 {{DataType::FP16}, {DataType::FP16}},
                                 {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer &) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer &) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

void FrontEnd::parsePriorBox(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    model->addNewStage<StubPriorBoxStage>(layer->name, StageType::StubPriorBox, layer, inputs, outputs);
}

void FrontEnd::parsePriorBoxClustered(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    model->addNewStage<StubPriorBoxStage>(layer->name, StageType::StubPriorBoxClustered, layer, inputs, outputs);
}

}  // namespace vpu
