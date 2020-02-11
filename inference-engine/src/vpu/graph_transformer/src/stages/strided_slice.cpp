// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <vector>

#include <vpu/stages/post_op_stage.hpp>

namespace vpu {

namespace {

class StridedSliceStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StridedSliceStage>(*this);
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
        IE_ASSERT(numInputs() == 3 || numInputs() == 4);
        IE_ASSERT(numOutputs() == 1);

        std::vector<EnumSet<DataType>> expectedInputs3Types =
            { {DataType::FP16}, {DataType::S32}, {DataType::S32} };
        std::vector<EnumSet<DataType>> expectedInputs4Types =
            { {DataType::FP16}, {DataType::S32}, {DataType::S32}, {DataType::S32} };

        assertInputsOutputsTypes(
            this,
            numInputs() == 3 ? expectedInputs3Types : expectedInputs4Types,
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

void FrontEnd::parseStridedSlice(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    IE_ASSERT(inputs.size() == 3 || inputs.size() == 4);
    IE_ASSERT(outputs.size() == 1);

    model->addNewStage<StridedSliceStage>(layer->name, StageType::StridedSlice, layer, inputs, outputs);
}

}  // namespace vpu
