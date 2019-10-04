// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class StridedSliceStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StridedSliceStage>(*this);
    }

    void propagateScaleFactorsImpl(
        const SmallVector<float>& inputScales,
        ScalePropagationStep step,
        StageDataInfo<float>& scaleInfo) override {
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 4);
        IE_ASSERT(numOutputs() == 1);
        assertInputsOutputsTypes(
            this,
            {{DataType::FP16}, {DataType::S32}, {DataType::S32}, {DataType::S32}},
            {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

void FrontEnd::parseStridedSlice(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 4);
    IE_ASSERT(outputs.size() == 1);

    model->addNewStage<StridedSliceStage>(
        layer->name,
        StageType::StridedSlice,
        layer,
        inputs,
        outputs);
}

}  // namespace vpu
