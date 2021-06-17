// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/utils/numeric.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <precision_utils.h>

#include <vector>
#include <set>
#include <memory>

namespace vpu {

namespace {

void deconvolutionRelayout(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int IC, int OC,
        int g, int GR) {
    for (int goc = 0; goc < OC / GR; ++goc) {
        for (int gic = 0; gic < IC / GR; ++gic) {
            for (int ky = 0; ky < KY; ++ky) {
                for (int kx = 0; kx < KX; ++kx) {
                    int iidx =
                        gic * OC * KY * KX +
                        (g * OC / GR + goc) * KY * KX +
                        ky * KX +
                        kx;
                    IE_ASSERT(iidx < src_size);

                    int oidx =
                        gic * (OC / GR) * KY * KX +
                        goc * KY * KX +
                        ky * KX +
                        kx;
                    IE_ASSERT(oidx < dst_size);

                    dst[oidx] = src[iidx];
                }
            }
        }
    }
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitGroupedConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv &&
            stage->type() != StageType::StubDeconv) {
            continue;
        }

        IE_ASSERT(stage->numInputs() == 4);
        IE_ASSERT(stage->numOutputs() == 1);

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        if (groupSize == 1) {
            continue;
        }

        if (groupSize == input->desc().dim(Dim::C) &&
            groupSize == output->desc().dim(Dim::C)) {
            // It is a Depth[De]Convolution, it is handled separately for SW and HW
            continue;
        }

        model->disconnectStage(stage);

        auto inGroupDimC = input->desc().dim(Dim::C) / groupSize;
        auto outGroupDimC = output->desc().dim(Dim::C) / groupSize;

        DataVector subInputs(groupSize);
        DataVector subOutputs(groupSize);

        for (int groupInd = 0; groupInd < groupSize; ++groupInd) {
            auto postfix = formatString("@group=%d/%d", groupInd + 1, groupSize);

            // subInput

            auto subInputDesc = input->desc();
            subInputDesc.setDim(Dim::C, inGroupDimC);

            subInputs[groupInd] = model->duplicateData(
                input,
                postfix,
                subInputDesc);

            // subWeights

            Data subWeights;
            {
                auto content = weights->content();
                IE_ASSERT(content != nullptr);

                auto origWeights = content->get<fp16_t>();
                IE_ASSERT(origWeights != nullptr);

                auto kernWxH = kernelSizeX * kernelSizeY;
                size_t newWeightsSize = kernWxH * inGroupDimC * outGroupDimC;

                auto newWeightsBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {newWeightsSize},
                    ie::Layout::C));
                newWeightsBlob->allocate();

                auto newWeightsPtr = newWeightsBlob->buffer().as<fp16_t*>();

                std::fill_n(newWeightsPtr, newWeightsSize, ie::PrecisionUtils::f32tof16(0.0f));

                if (stage->type() == StageType::StubDeconv) {
                    deconvolutionRelayout(
                        origWeights, weights->desc().totalDimSize(),
                        newWeightsPtr, static_cast<int>(newWeightsSize),
                        kernelSizeX, kernelSizeY,
                        input->desc().dim(Dim::C),
                        output->desc().dim(Dim::C),
                        groupInd, groupSize);
                } else {
                    std::copy_n(origWeights + newWeightsSize * groupInd, newWeightsSize, newWeightsPtr);
                }

                subWeights = model->duplicateData(
                    weights,
                    postfix,
                    DataDesc({kernelSizeX, kernelSizeY, inGroupDimC, outGroupDimC}),
                    ieBlobContent(newWeightsBlob));
            }

            // subBiases

            auto subBiases = biases;
            if (biases->usage() != DataUsage::Fake) {
                auto content = biases->content();
                IE_ASSERT(content != nullptr);

                auto origBiases = content->get<fp16_t>();
                IE_ASSERT(origBiases != nullptr);

                auto newBiasesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {static_cast<size_t>(outGroupDimC)},
                    ie::Layout::C));
                newBiasesBlob->allocate();

                auto newBiasesPtr = newBiasesBlob->buffer().as<fp16_t*>();

                std::copy_n(origBiases + groupInd * outGroupDimC, outGroupDimC, newBiasesPtr);

                subBiases = model->duplicateData(
                    biases,
                    postfix,
                    DataDesc({outGroupDimC}),
                    ieBlobContent(newBiasesBlob));
            }

            // subScales

            auto subScales = scales;
            if (scales->usage() != DataUsage::Fake) {
                auto content = scales->content();
                IE_ASSERT(content != nullptr);

                auto origScales = content->get<fp16_t>();
                IE_ASSERT(origScales != nullptr);

                auto newScalesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {static_cast<size_t>(outGroupDimC)},
                    ie::Layout::C));
                newScalesBlob->allocate();

                auto newScalesPtr = newScalesBlob->buffer().as<fp16_t*>();

                std::copy_n(origScales + groupInd * outGroupDimC, outGroupDimC, newScalesPtr);

                subScales = model->duplicateData(
                    scales,
                    postfix,
                    DataDesc({outGroupDimC}),
                    ieBlobContent(newScalesBlob));
            }

            // subOutput

            auto subOutputDesc = output->desc();
            subOutputDesc.setDim(Dim::C, outGroupDimC);

            subOutputs[groupInd] = model->duplicateData(
                output,
                postfix,
                subOutputDesc);

            // subConvStage

            auto subConvStage = model->duplicateStage(
                stage,
                postfix,
                {subInputs[groupInd], subWeights, subBiases, subScales},
                {subOutputs[groupInd]});

            subConvStage->attrs().set<int>("groupSize", 1);
            subConvStage->attrs().set<int>("groupInd", groupInd);
        }

        _stageBuilder->addSplitStage(
            model,
            stage->name() + "@split",
            stage->origNode(),
            Dim::C,
            input,
            subInputs);

        _stageBuilder->addConcatStage(
            model,
            stage->name() + "@concat",
            stage->origNode(),
            Dim::C,
            subOutputs,
            output);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::splitGroupedConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
