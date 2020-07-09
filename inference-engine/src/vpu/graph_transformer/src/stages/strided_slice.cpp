// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/stages/post_op_stage.hpp>

#include <memory>
#include <vector>

namespace vpu {

namespace {

int maskStrToInt(std::string mask) {
    int idx = 0, result = 0;

    for (const auto& character : mask) {
        if (character == ',') continue;

        if (idx++ > 0) {
            result <<= 1;
        }
        if (character == '1') {
            result = result | 1;
        }
    }

    return result;
}

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
        VPU_THROW_UNLESS(numInputs() == 3 || numInputs() == 4,
            "Validating layer {} with type {} failed: number of input should be 3 or 4, but {} were provided",
            name(), type(), numInputs());
        VPU_THROW_UNLESS(numOutputs() == 1,
            "Validating layer {} with type {} failed: number of outputs should be 1, but {} were provided",
            name(), type(), numOutputs());

        std::string newAxisMask = origLayer()->GetParamAsString("new_axis_mask", "");
        VPU_THROW_UNLESS(maskStrToInt(newAxisMask) == 0,
            "Checking {} with type {} failed: new_axis_mask parameter is not supported",
            name(), type());
        std::string shrinkAxisMask = origLayer()->GetParamAsString("shrink_axis_mask", "");
        VPU_THROW_UNLESS(maskStrToInt(newAxisMask) == 0,
            "Checking {} with type {} failed: shrink_axis_mask parameter is not supported",
            name(), type());
        std::string ellipsisMask = origLayer()->GetParamAsString("ellipsis_mask", "");
        VPU_THROW_UNLESS(maskStrToInt(newAxisMask) == 0,
            "Checking {} with type {} failed: ellipsis_mask parameter is not supported",
            name(), type());

        const auto& strides = input(3);
        if (strides->usage() == DataUsage::Const) {
            const auto stridesPtr = strides->content()->get<int32_t>();
            VPU_THROW_UNLESS(stridesPtr != nullptr,
                             "Checking {} with type {} failed: pointer for strides is null");
            for (int i = 0; i < strides->desc().totalDimSize(); i++) {
                VPU_THROW_UNLESS(stridesPtr[i] > 0,
                                 "Checking {} with type {} failed: negative stride is not supported");
            }
        }

        std::vector<EnumSet<DataType>> expectedInputs3Types =
            { {DataType::FP16, DataType::S32}, {DataType::S32}, {DataType::S32} };
        std::vector<EnumSet<DataType>> expectedInputs4Types =
            { {DataType::FP16, DataType::S32}, {DataType::S32}, {DataType::S32}, {DataType::S32} };

        const auto& input0DataType = input(0)->desc().type();

        assertInputsOutputsTypes(
            this,
            numInputs() == 3 ? expectedInputs3Types : expectedInputs4Types,
            {{input0DataType}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        std::string beginMask = origLayer()->GetParamAsString("begin_mask", "");
        std::string endMask = origLayer()->GetParamAsString("end_mask", "");
        serializer.append(maskStrToInt(beginMask));
        serializer.append(maskStrToInt(endMask));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        input(0)->serializeBuffer(serializer);
        input(1)->serializeBuffer(serializer);
        input(2)->serializeBuffer(serializer);
        input(3)->serializeBuffer(serializer);
        output(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseStridedSlice(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 3 || inputs.size() == 4,
        "Parsing layer {} with type {} failed: number of input should be 3 or 4, but {} were provided",
        layer->name, layer->type, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 1,
        "Parsing layer {} with type {} failed: number of outputs should be 1, but {} were provided",
        layer->name, layer->type, outputs.size());

    const auto& begin = inputs[1];
    const auto& end = inputs[2];

    VPU_THROW_UNLESS(begin->usage() == DataUsage::Const,
                     "Checking {} with type {} failed: only {} type for begin is supported, but {} was provided",
                     layer->name, layer->type, DataUsage::Const, begin->usage());
    VPU_THROW_UNLESS(end->usage() == DataUsage::Const,
                     "Checking {} with type {} failed: only {} type for end is supported, but {} was provided",
                     layer->name, layer->type, DataUsage::Const, end->usage());
    VPU_THROW_UNLESS(inputs.size() == 3 || inputs[3]->usage() == DataUsage::Const,
                     "Checking {} with type {} failed: only {} type for strides is supported, but {} was provided",
                     layer->name, layer->type, DataUsage::Const, inputs[3]->usage());

    DataVector extendedInputs{inputs.begin(), inputs.end()};
    if (inputs.size() == 3) {
        extendedInputs.push_back(model->addFakeData());
    }

    model->addNewStage<StridedSliceStage>(layer->name, StageType::StridedSlice, layer, extendedInputs, outputs);
}

}  // namespace vpu
