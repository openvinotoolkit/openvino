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

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitLargeKernelConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv) {
            continue;
        }
        const auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        IE_ASSERT(stage->numInputs() == 4);
        IE_ASSERT(stage->numOutputs() == 1);

        const auto input = stage->input(0);
        const auto weights = stage->input(1);
        const auto biases = stage->input(2);
        const auto scales  = stage->input(3);
        const auto output = stage->output(0);

        const auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        const auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        const auto groupSize = stage->attrs().get<int>("groupSize");
        const auto inputC = input->desc().dim(Dim::C);
        const auto outputC = output->desc().dim(Dim::C);

        //Only 1x16 convolution is supported now, could expand support up to 1x30
        if (kernelSizeX != 16 || kernelSizeY != 1 || groupSize != 1) {
            continue;
        }

        model->disconnectStage(stage);

        int kernelGroupSize = kernelSizeX / 16 + 1;
        IE_ASSERT(kernelGroupSize == 2);
        const auto newKernelSizeX = kernelSizeX / kernelGroupSize;
        IE_ASSERT(newKernelSizeX * kernelGroupSize == kernelSizeX);
        const auto inGroupDimX = input->desc().dim(Dim::W) - newKernelSizeX;

        DataVector subInputs(kernelGroupSize);
        DataVector subOutputs(kernelGroupSize);

        for (int groupInd = 0; groupInd < kernelGroupSize; ++groupInd) {
            auto postfix = formatString("@subkernel=%d/%d", groupInd + 1, kernelGroupSize);

            //subInput
            auto subInputDesc = input->desc();
            subInputDesc.setDim(Dim::W, inGroupDimX);

                subInputs[groupInd] = model->duplicateData(
                    input,
                    postfix,
                    subInputDesc);

            //subShrinkStage
            DimValues offsets({{Dim::W, groupInd * newKernelSizeX}});

            _stageBuilder->addCropStage(
                    model,
                    stage->name() + postfix,
                    stage->origNode(),
                    input,
                    subInputs[groupInd],
                    std::move(offsets));

            // subWeights
            Data subWeights;
            {
                const auto content = weights->content();
                IE_ASSERT(content != nullptr);

                const auto origWeights = content->get<fp16_t>();
                IE_ASSERT(origWeights != nullptr);

                size_t newWeightsSize = newKernelSizeX * kernelSizeY * outputC * inputC;

                auto newWeightsBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {newWeightsSize},
                    ie::Layout::C));
                newWeightsBlob->allocate();

                const auto newWeightsPtr = newWeightsBlob->buffer().as<fp16_t*>();
                auto src = origWeights + groupInd * newKernelSizeX;
                auto dst = newWeightsPtr;

                for (int i = 0; i < kernelSizeY * inputC * outputC; ++i) {
                    std::copy_n(src + i * kernelSizeX, newKernelSizeX, dst + i * newKernelSizeX);
                }

                subWeights = model->duplicateData(
                    weights,
                    postfix,
                    DataDesc({newKernelSizeX, kernelSizeY, inputC, outputC}),
                    ieBlobContent(newWeightsBlob));
            }

            // subOutput
            auto subOutputDesc = output->desc();

            subOutputs[groupInd] = model->duplicateData(
                output,
                postfix,
                subOutputDesc);

            // subConvStage
            auto subConvStage = model->duplicateStage(
                stage,
                postfix,
                {subInputs[groupInd], subWeights, biases, scales},
                {subOutputs[groupInd]});

            subConvStage->attrs().set<int>("kernelSizeX", newKernelSizeX);
        }

        _stageBuilder->addSumStage(
            model,
            stage->name() + "@sum",
            stage->origNode(),
            subOutputs[0],
            subOutputs[1],
            output);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::splitLargeKernelConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
