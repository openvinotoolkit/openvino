// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cmath>

#include <vector>
#include <memory>

#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

namespace {

class BatchNormalizationWeightsContent final : public CalculatedDataContent {
public:
    BatchNormalizationWeightsContent(
            const DataContent::Ptr& origContent,
            float epsilon) :
            CalculatedDataContent({origContent}), _epsilon(epsilon) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(BatchNormalizationWeightsContent);

        auto srcPtr = baseContents[0]->get<fp16_t>();
        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(_desc.totalDimSize(), [this, srcPtr, dstPtr](int i) {
            float val = ie::PrecisionUtils::f16tof32(srcPtr[i]) + _epsilon;
            val = 1.0f / std::sqrt(val);
            dstPtr[i] = ie::PrecisionUtils::f32tof16(val);
        });
    }

private:
    float _epsilon;
};

class BatchNormalizationBiasesContent final : public CalculatedDataContent {
public:
    BatchNormalizationBiasesContent(
            const DataContent::Ptr& origContent,
            const DataContent::Ptr& weightsContent) :
            CalculatedDataContent({origContent, weightsContent}) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(BatchNormalizationBiasesContent);

        auto origPtr = baseContents[0]->get<fp16_t>();
        auto weightsPtr = baseContents[1]->get<fp16_t>();

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(_desc.totalDimSize(), [origPtr, weightsPtr, dstPtr](int i) {
            // TODO : need to be extracted from IE layer.
            float beta = 0.0f;

            auto wVal = ie::PrecisionUtils::f16tof32(weightsPtr[i]);
            dstPtr[i] = ie::PrecisionUtils::f32tof16(beta - wVal * ie::PrecisionUtils::f16tof32(origPtr[i]));
        });
    }
};

}  // namespace

void FrontEnd::parseBatchNorm(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto layer = std::dynamic_pointer_cast<ie::BatchNormalizationLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];
    auto output = outputs[0];

    Data origWeights, origBiases;
    std::tie(origWeights, origBiases) = getWeightsAndBiases(model, layer);

    IE_ASSERT(origWeights->desc().totalDimSize() >= input->desc().dim(Dim::C));
    auto weights = model->duplicateData(
        origWeights,
        "@batch-norm",
        DataDesc({input->desc().dim(Dim::C)}),
        std::make_shared<BatchNormalizationWeightsContent>(
            origWeights->content(),
            layer->epsilon));

    if (origBiases->usage() != DataUsage::Fake) {
        IE_ASSERT(origBiases->desc().totalDimSize() >= input->desc().dim(Dim::C));
        auto biases = model->duplicateData(
            origBiases,
            "@batch-norm",
            DataDesc({input->desc().dim(Dim::C)}),
            std::make_shared<BatchNormalizationBiasesContent>(
                origBiases->content(),
                weights->content()));

        auto tempOutput = model->duplicateData(
            output,
            "@temp");

        _stageBuilder->addBiasStage(
            model,
            layer->name,
            layer,
            tempOutput, biases,
            output);

        output = tempOutput;
    }

    _stageBuilder->addScaleStage(
        model,
        layer->name,
        layer,
        input, weights,
        output);
}

}  // namespace vpu
