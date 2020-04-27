// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class PoolStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PoolStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        orderInfo.setInput(inputEdge(0), input->desc().dimsOrder());
        orderInfo.setOutput(outputEdge(0), input->desc().dimsOrder());
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();
        auto dimsOrder = input->desc().dimsOrder();

        //
        // * AvgPool/MaxPool support both YXZ and ZYX orders:
        //   * ZYX versions support both input and output strides.
        //   * YXZ versions support only output strides.
        // * GlobalPooling supports both 3D/4D layouts.
        //

        if (type() == StageType::MaxPool || type() == StageType::AvgPool) {
            if (dimsOrder.dimInd(Dim::C) == 0) {
                stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            }
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void finalCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");
        auto kernelStrideX = attrs().get<int>("kernelStrideX");
        auto kernelStrideY = attrs().get<int>("kernelStrideY");
        auto padLeft = attrs().get<int>("padLeft");
        auto padTop = attrs().get<int>("padTop");
        auto excludePad = attrs().get<bool>("excludePad");

        serializer.append(static_cast<uint32_t>(kernelSizeX));
        serializer.append(static_cast<uint32_t>(kernelSizeY));
        serializer.append(static_cast<uint32_t>(kernelStrideX));
        serializer.append(static_cast<uint32_t>(kernelStrideY));
        serializer.append(static_cast<uint32_t>(padLeft));
        serializer.append(static_cast<uint32_t>(padTop));
        serializer.append(static_cast<uint32_t>(excludePad));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(swPoolAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubMaxPool &&
            stage->type() != StageType::StubAvgPool) {
            continue;
        }

        auto input = stage->input(0);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto excludePad = stage->attrs().get<bool>("excludePad");

        model->disconnectStage(stage);

        const bool isOverlapByX = kernelSizeX - padLeft >= input->desc().dim(Dim::W);
        const bool isOverlapByY = kernelSizeY - padTop >= input->desc().dim(Dim::H);
        const bool isOverlapByKernel = isOverlapByX && isOverlapByY;
        const bool paddingsNotExist = padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0;
        const bool isGlobalPoolingOutputFormat =
                output->desc().dim(Dim::W) == 1 && output->desc().dim(Dim::H) == 1;
        auto stageType = StageType::None;
        if (stage->type() == StageType::StubMaxPool) {
            if (isGlobalPoolingOutputFormat && isOverlapByKernel) {
                stageType = StageType::GlobalMaxPool;
            } else {
                stageType = StageType::MaxPool;
            }
        } else {
            if (isGlobalPoolingOutputFormat && (isOverlapByKernel && (paddingsNotExist || excludePad))) {
                stageType = StageType::GlobalAvgPool;
            } else {
                stageType = StageType::AvgPool;
            }
        }

        auto swStage = model->addNewStage<PoolStage>(
            stage->name(),
            stageType,
            stage->origLayer(),
            {input},
            {output});

        swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
        swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

        swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
        swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

        swStage->attrs().set<int>("padLeft", padLeft);
        swStage->attrs().set<int>("padRight", padRight);
        swStage->attrs().set<int>("padTop", padTop);
        swStage->attrs().set<int>("padBottom", padBottom);

        swStage->attrs().set<bool>("excludePad", excludePad);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swPoolAdaptation() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
