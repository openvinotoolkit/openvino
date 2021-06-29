// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

VPU_PACKED(ExpPriorGridGeneratorParams {
    int32_t flatten;
    int32_t grid_w;
    int32_t grid_h;
    float   stride_w;
    float   stride_h;
};)

class ExpPriorGridGeneratorStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpPriorGridGeneratorStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<ExpPriorGridGeneratorParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (auto& inputEdge : inputEdges()) {
            inputEdge->input()->serializeBuffer(serializer);
        }

        outputEdges()[0]->output()->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseExpPriorGridGenerator(
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs) const {
    IE_ASSERT(inputs.size() >= 1 && inputs.size() <= 3);
    IE_ASSERT(outputs.size() == 1);
    auto expPriorGridGenerator = ngraph::as_type_ptr<ngraph::op::v6::ExperimentalDetectronPriorGridGenerator>(node);
    IE_ASSERT(expPriorGridGenerator != nullptr);
    ExpPriorGridGeneratorParams params;
    const auto attrs = expPriorGridGenerator->get_attrs();
    params.flatten  = attrs.flatten;
    params.grid_w   = attrs.w;
    params.grid_h   = attrs.h;
    params.stride_w = attrs.stride_x;
    params.stride_h = attrs.stride_y;

    auto inputPriors     = inputs[0];   // [n][4]
    auto inputFeatureMap = inputs[1];   // [b, c, h, w]
    auto inputImage      = inputs[2];   // [b, 3, im_h, im_w]
    auto outputPriorGrid = outputs[0];  // [m][4]

    const int batch      = inputImage->desc().dim(Dim::N);

    VPU_THROW_UNLESS((inputPriors->desc().dims().size() == 2) &&
                     (inputPriors->desc().dim(Dim::C) == 4),
                     "Wrong shape for input 0 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     expPriorGridGenerator->get_name(), inputPriors->desc().dims().size(), inputPriors->desc().dim(Dim::C));

    if (params.grid_h == 0 || params.grid_w == 0) {
        VPU_THROW_UNLESS((inputFeatureMap->desc().dims().size() == 4) &&
                         (inputFeatureMap->desc().dim(Dim::N) == batch),
                         "Wrong shape for input 1 of layer %s, expected 4-dimensional"
                         "with batch = %d, got: dims size = %lu, batch = %d",
                         expPriorGridGenerator->get_name(), batch, inputFeatureMap->desc().dims().size(), inputPriors->desc().dim(Dim::N));
    }

    if (params.stride_w == 0 || params.stride_h == 0) {
        VPU_THROW_UNLESS(inputImage->desc().dims().size() == 4,
                         "Wrong shape for input 2 of layer %s, expected 4-dimensional, got: dims size = %lu",
                         expPriorGridGenerator->get_name(), inputImage->desc().dims().size());
    }

    VPU_THROW_UNLESS((outputPriorGrid->desc().dims().size() == 2) &&
                     (outputPriorGrid->desc().dim(Dim::C) == 4),
                     "Wrong shape for output of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     expPriorGridGenerator->get_name(), outputPriorGrid->desc().dims().size(), outputPriorGrid->desc().dim(Dim::C));

    auto stage = model->addNewStage<ExpPriorGridGeneratorStage>(
        expPriorGridGenerator->get_name(),
        StageType::ExpPriorGridGenerator,
        expPriorGridGenerator,
        inputs,
        outputs);

    stage->attrs().set("params", params);
}

}  // namespace vpu
