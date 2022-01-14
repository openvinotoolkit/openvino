// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/stages/stub_stage.hpp>
#include <vpu/middleend/sw/utility.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/model/data_contents/deconvolution_contents.hpp>
#include <vpu/utils/hw_disabled.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;

class UpsamplingStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<UpsamplingStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        orderInfo.setInput(inputEdge(0), DimsOrder::NCHW);
        orderInfo.setOutput(outputEdge(0), DimsOrder::NCHW);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setOutput(outputEdge(0), StridesRequirement().add(1, DimStride::Aligned));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::TwoOrOne;
    }

    void finalCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto scaleX = attrs().get<int>("upsampling_factorx_x");
        auto scaleY = attrs().get<int>("upsampling_factorx_y");
        auto scaleZ = attrs().get<int>("upsampling_factorx_z");
        auto pad_l_x = attrs().get<int>("pad_l_x");
        auto pad_r_x = attrs().get<int>("pad_r_x");
        auto pad_l_y = attrs().get<int>("pad_l_y");
        auto pad_r_y = attrs().get<int>("pad_r_y");
        auto pad_l_z = attrs().get<int>("pad_l_z");
        auto pad_r_z = attrs().get<int>("pad_r_z");

        serializer.append(static_cast<int32_t>(scaleX));
        serializer.append(static_cast<int32_t>(scaleY));
        serializer.append(static_cast<int32_t>(scaleZ));
        serializer.append(static_cast<int32_t>(pad_l_x));
        serializer.append(static_cast<int32_t>(pad_r_x));
        serializer.append(static_cast<int32_t>(pad_l_y));
        serializer.append(static_cast<int32_t>(pad_r_y));
        serializer.append(static_cast<int32_t>(pad_l_z));
        serializer.append(static_cast<int32_t>(pad_r_z));
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
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(replaceDeconvByConv);

    auto stages = model->getStages();
    for (const auto& stage : stages) {
        if (stage->type() != StageType::StubDeconv) {
            continue;
        }

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        auto padLeft  = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto deconvScale = stage->attrs().getOrDefault<float>("scaleFactor", 1.0f);

        /* Upsampling layer does not support negative paddings */
        if ((kernelSizeX - 1 - padLeft < 0) || (kernelSizeX - 1 - padRight < 0) ||
            (kernelSizeY - 1 - padTop < 0) || (kernelSizeY - 1 - padBottom < 0)) {
            continue;
        }

        if (groupSize != 1) {
            continue;
        }

        if ((padTop != padBottom) || (padLeft != padRight)) {
            continue;
        }

        if (kernelSizeX > 15 || kernelSizeY > 15) {
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases  = stage->input(2);
        auto scales  = stage->input(3);
        auto output = stage->output(0);
        const auto& env = CompileEnv::get();

        if (HwDisabled(env.config, stage->origLayer()->name)) {
            continue;
        }

        if (output->desc().numDims() < 4) {
            continue;
        }

        // problem with Deconv/CommonSingleLayerTest
        auto origOutputX = kernelStrideX * (input->desc().dim(Dim::W)  - 1) + kernelSizeX - padLeft - padRight;
        auto origOutputY = kernelStrideY * (input->desc().dim(Dim::H)  - 1) + kernelSizeY - padTop - padBottom;

        if ((origOutputX != output->desc().dim(Dim::W)) || (origOutputY != output->desc().dim(Dim::H))) {
            continue;
        }

        model->disconnectStage(stage);

        DataDesc newDesc({1, 1, output->desc().dim(Dim::C), output->desc().dim(Dim::N)});
        newDesc.setDim(Dim::N, input->desc().dim(Dim::N));
        newDesc.setDim(Dim::C, input->desc().dim(Dim::C));
        newDesc.setDim(Dim::H, (input->desc().dim(Dim::H) - 1) * kernelStrideY + 1 + (kernelSizeY - 1) * 2 - padTop - padBottom);
        newDesc.setDim(Dim::W, (input->desc().dim(Dim::W) - 1) * kernelStrideX + 1 + (kernelSizeX - 1) * 2 - padLeft - padRight);

        auto newOutput = model->duplicateData(output, "@upsampleData", newDesc);
        auto newWeights = model->duplicateData(weights, "@upsampleData", weights->desc(),
                     std::make_shared<DeconvolutionToConvolutionContent>(weights->content(), weights->desc()));

        auto upsampleStage = model->addNewStage<UpsamplingStage>(
                stage->origLayerName() + "@Upsample",
                StageType::Upsampling,
                stage->origLayer(),
                {input},
                {newOutput});

        upsampleStage->attrs().set<int>("upsampling_factorx_x", kernelStrideX);
        upsampleStage->attrs().set<int>("upsampling_factorx_y", kernelStrideY);
        upsampleStage->attrs().set<int>("upsampling_factorx_z", 1);
        upsampleStage->attrs().set<int>("pad_l_x", (kernelSizeX - 1) - padLeft);
        upsampleStage->attrs().set<int>("pad_r_x", (kernelSizeX - 1) - padRight);
        upsampleStage->attrs().set<int>("pad_l_y", (kernelSizeY - 1) - padTop);
        upsampleStage->attrs().set<int>("pad_r_y", (kernelSizeY - 1) - padBottom);
        upsampleStage->attrs().set<int>("pad_l_z", 0);
        upsampleStage->attrs().set<int>("pad_r_z", 0);

        auto newStage = model->addNewStage<StubStage>(
                stage->origLayerName() + "@UpsampleConv",
                StageType::StubConv,
                stage->origLayer(),
                {newOutput, newWeights, biases, scales},
                {output});

        newStage->attrs().set<int>("kernelSizeX", kernelSizeX);
        newStage->attrs().set<int>("kernelSizeY", kernelSizeY);
        newStage->attrs().set<int>("kernelStrideX", 1);
        newStage->attrs().set<int>("kernelStrideY", 1);
        newStage->attrs().set<int>("padLeft", 0);
        newStage->attrs().set<int>("padRight", 0);
        newStage->attrs().set<int>("padTop", 0);
        newStage->attrs().set<int>("padBottom", 0);
        newStage->attrs().set<int>("dilationX", 1);
        newStage->attrs().set<int>("dilationY", 1);
        newStage->attrs().set<int>("groupSize", 1);
        newStage->attrs().set<bool>("tryHW", true);
        newStage->attrs().set<float>("scaleFactor", deconvScale);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::replaceDeconvByConv() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
