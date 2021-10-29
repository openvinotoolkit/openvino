// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/middleend/sw/utility.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <precision_utils.h>

#include <set>
#include <memory>
#include <vector>
#include <utility>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(replaceWithReduceMean);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubAvgPool) {
            continue;
        }

        const auto stageInput = stage->input(0);
        const auto stageOutput = stage->output(0);

        const auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        const auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        const bool excludePad = stage->attrs().get<bool>("excludePad");

        const auto padLeft = stage->attrs().get<int>("padLeft");
        const auto padRight = stage->attrs().get<int>("padRight");
        const auto padTop = stage->attrs().get<int>("padTop");
        const auto padBottom = stage->attrs().get<int>("padBottom");

        VPU_THROW_UNLESS(
                kernelSizeX > 0 && kernelSizeY > 0,
                "[ReplaceWithReduceMean] Stage %v with type AvgPool has non-positive kernel size",
                stage->name());

        const bool isOverlapByX = kernelSizeX - padLeft >= stageInput->desc().dim(Dim::W);
        const bool isOverlapByY = kernelSizeY - padTop >= stageInput->desc().dim(Dim::H);
        const bool isOverlapByKernel = isOverlapByX && isOverlapByY;
        const bool paddingsNotExist = padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0;
        const bool isGlobalPoolingOutputFormat = stageOutput->desc().dim(Dim::W) == 1 && stageOutput->desc().dim(Dim::H) == 1;
        const bool isGlobalAvgPooling = (isGlobalPoolingOutputFormat && (isOverlapByKernel && (paddingsNotExist || excludePad)));

        if (isGlobalAvgPooling) {
            auto origLayer = stage->origLayer();

            model->removeStage(stage);

            const auto generator = [&stageInput](const ie::Blob::Ptr& blob) {
                auto buffer = blob->buffer().as<int32_t*>();
                auto numInputDims = stageInput->desc().numDims();

                // H and W are always come last in IE notation
                buffer[0] = numInputDims - 1;
                buffer[1] = numInputDims - 2;
            };

            auto axesData = model->addConstData(origLayer->name + "@axes", DataDesc(DataType::S32, DimsOrder::C, {2}), generator);

            _stageBuilder->addReduceStage(
                    model,
                    "ReduceMean",
                    StageType::ReduceMean,
                    origLayer,
                    true,
                    {stageInput, axesData},
                    stageOutput);
        }
    }
}
}  // namespace


Pass::Ptr PassManager::replaceWithReduceMean() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
