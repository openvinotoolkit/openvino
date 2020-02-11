// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/middleend/sw/utility.hpp>
#include <vpu/model/data.hpp>

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

        auto stageInput = stage->input(0);
        const auto& dimValues = stageInput->desc().dims();

        const auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        const auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");

        const auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        const auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");

        VPU_THROW_UNLESS(
                kernelSizeX > 0 && kernelSizeY > 0,
                "[ReplaceWithReduceMean] Stage %v with type AvgPool has non-positive kernel size",
                stage->name());

        if (dimValues[Dim::W] == kernelSizeX && dimValues[Dim::H] == kernelSizeY) {  // GlobalPooling
            if (kernelStrideX != 1 && kernelStrideY != 1) {
                continue;
            }

            // TODO: since ReduceMean currently is not fully optimized, we need to discard some common cases
            if (kernelSizeX * kernelSizeY < 2050) {
                continue;
            }

            auto origLayer = stage->origLayer();
            auto stageOutput = stage->output(0);

            model->removeStage(stage);

            auto axesBlob = ie::make_shared_blob<int32_t>(ie::TensorDesc(ie::Precision::I32, {2}, ie::Layout::C));
            axesBlob->allocate();
            auto buffer = axesBlob->buffer().as<int32_t *>();

            auto numInputDims = stageInput->desc().numDims();
            // H and W are always come last in IE notation
            buffer[0] = numInputDims - 1;
            buffer[1] = numInputDims - 2;

            auto axesData = model->addConstData(
                    origLayer->name + "@axes",
                    DataDesc(DataType::S32, DimsOrder::C, {2}),
                    ieBlobContent(axesBlob));

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
