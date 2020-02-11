// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <limits>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <set>

#include <vpu/middleend/sw/utility.hpp>

#define REFERENCE_CONVOLUTION 0

namespace vpu {

namespace {

class ConvIm2ColWeightsContent final : public CalculatedDataContent {
public:
    explicit ConvIm2ColWeightsContent(const DataContent::Ptr& origContent) :
            CalculatedDataContent({origContent}) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ConvIm2ColWeightsContent);
        kchw_to_khwc(baseContents[0]->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), desc());
    }
};

class Conv3x3WeightsContent final : public CalculatedDataContent {
public:
    explicit Conv3x3WeightsContent(const DataContent::Ptr& origContent) :
            CalculatedDataContent({origContent}) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(Conv3x3WeightsContent);
        kchw_to_hwkc(baseContents[0]->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), desc());
    }
};

class ConvCHWWeightsContent final : public CalculatedDataContent {
public:
    explicit ConvCHWWeightsContent(const DataContent::Ptr& origContent) :
            CalculatedDataContent({origContent}) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ConvCHWWeightsContent);
        kchw_to_hwkc(baseContents[0]->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), desc());
    }
};

class ConvStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConvStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        auto finalOrder = input->desc().dimsOrder();
        if (finalOrder.dimInd(Dim::C) == 1) {
            // HCW -> CHW
            finalOrder.moveDim(Dim::C, 2);
        }

        if (type() == StageType::Conv ||
            type() == StageType::Im2ColConvolution) {
            if (finalOrder != input->desc().dimsOrder()) {
                orderInfo.setInput(inputEdge(0), finalOrder);
            }
            orderInfo.setOutput(outputEdge(0), finalOrder);
        } else if (type() == StageType::DepthConv) {
            if (finalOrder != input->desc().dimsOrder()) {
                orderInfo.setInput(inputEdge(0), finalOrder);
            }
            orderInfo.setOutput(outputEdge(0), finalOrder);
        } else {
            orderInfo.setInput(inputEdge(0), finalOrder.createMovedDim(Dim::C, 0));
            orderInfo.setOutput(outputEdge(0), finalOrder.createMovedDim(Dim::C, 0));
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        if (type() != StageType::DepthConv) {
            stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
            stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto output = outputEdge(0)->output();

        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");

        Data swWeights;

        if (type() == StageType::DepthConv) {
            swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
            if (swWeights == nullptr) {
                DataDesc newWeightsDesc({
                    kernelSizeX * kernelSizeY,
                    1,
                    output->desc().dim(Dim::C)});

                swWeights = model()->duplicateData(
                    weights,
                    "@SW",
                    newWeightsDesc,
                    std::make_shared<DefaultSwWeightsContent>(weights->content()));

                weights->attrs().set<Data>("swWeights", swWeights);
            }
        } else if (input->desc().dimsOrder().dimInd(Dim::C) == 0) {
            //
            // HWC case
            //

            auto isSpatialConv = attrs().get<bool>("isSpatialConv");
            auto isConv1x1 = attrs().get<bool>("isConv1x1");
            auto isConv3x3 = attrs().get<bool>("isConv3x3");

            swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
            if (swWeights == nullptr) {
                DataDesc newWeightsDesc({
                    kernelSizeX * kernelSizeY,
                    input->desc().dim(Dim::C),
                    output->desc().dim(Dim::C)});

                if (isSpatialConv) {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<DefaultSwWeightsContent>(weights->content()));
                } else if (isConv1x1) {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        weights->content());
                } else if (isConv3x3) {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<Conv3x3WeightsContent>(weights->content()));
                } else {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<ConvIm2ColWeightsContent>(weights->content()));
                }

                weights->attrs().set<Data>("swWeights", swWeights);
            }
        } else if (input->desc().dimsOrder().dimInd(Dim::C) == 2) {
            //
            // CHW case
            //

            auto isConv1x1 = attrs().get<bool>("isConv1x1");

            if (type() == StageType::Im2ColConvolution) {
                // Transform CHW "Im2ColConvolution" into CHW "Conv"
                changeType(StageType::Conv);
            }

            swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
            if (swWeights == nullptr) {
                DataDesc newWeightsDesc({
                    kernelSizeX * kernelSizeY,
                    input->desc().dim(Dim::C),
                    output->desc().dim(Dim::C)});

                if (isConv1x1) {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        weights->content());
                } else {
                    swWeights = model()->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<ConvCHWWeightsContent>(weights->content()));
                }

                weights->attrs().set<Data>("swWeights", swWeights);
            }
        }

        IE_ASSERT(swWeights != nullptr);

        model()->replaceStageInput(inputEdge(1), swWeights);
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    void finalCheckImpl() const override {
        assertInputsOutputsTypes(this,
              {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
              {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");
        auto kernelStrideX = attrs().get<int>("kernelStrideX");
        auto kernelStrideY = attrs().get<int>("kernelStrideY");
        auto padLeft = attrs().get<int>("padLeft");
        auto padTop = attrs().get<int>("padTop");
        auto dilationX = attrs().get<int>("dilationX");
        auto dilationY = attrs().get<int>("dilationY");

        serializer.append(static_cast<uint32_t>(kernelSizeX));
        serializer.append(static_cast<uint32_t>(kernelSizeY));
        serializer.append(static_cast<uint32_t>(kernelStrideX));
        serializer.append(static_cast<uint32_t>(kernelStrideY));
        serializer.append(static_cast<uint32_t>(padLeft));
        serializer.append(static_cast<uint32_t>(padTop));
        serializer.append(static_cast<uint32_t>(dilationX));
        serializer.append(static_cast<uint32_t>(dilationY));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto weights = inputEdge(1)->input();
        auto biases = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(this, serializer);
        output->serializeOldBuffer(this, serializer);
        weights->serializeOldBuffer(this, serializer);

        if (numTempBuffers() == 1) {
            tempBuffer(0)->serializeOldBuffer(this, serializer);
        }

        // TODO: remove this
        biases->serializeOldBuffer(this, serializer);
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
    VPU_PROFILE(swConvAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv)
            continue;

        auto origStageName = stage->name();
        auto origLayer = stage->origLayer();

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto dilationX = stage->attrs().get<int>("dilationX");
        auto dilationY = stage->attrs().get<int>("dilationY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        model->removeStage(stage);

        bool isFC = (
            kernelSizeX == 1 && kernelSizeY == 1 &&
            kernelStrideX == 1 && kernelStrideY == 1 &&
            padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0 &&
            dilationX == 1 && dilationY == 1 &&
            input->desc().dim(Dim::W) == 1 && input->desc().dim(Dim::H) == 1 &&
            output->desc().dim(Dim::W) == 1 && output->desc().dim(Dim::H) == 1);

        bool isConv1x1 = (
            kernelSizeX == 1 && kernelSizeY == 1 &&
            dilationX == 1 && dilationY == 1 &&
            !isFC);

        bool isConv3x3 = (
            kernelSizeX == 3 && kernelSizeY == 3 &&
            (input->desc().dim(Dim::C) / groupSize) > 3 &&
            ((input->desc().dim(Dim::C) / groupSize) * (output->desc().dim(Dim::C) / groupSize)) > 256);

        bool iskernelSizeMatchSpatial = (
            kernelSizeX > 1 && kernelSizeX < 12 && kernelSizeX % 2 == 1);

        bool isSpatialConv = (
            iskernelSizeMatchSpatial &&
            kernelSizeY != 1 &&  // kernelSizeX != 1 was checked in iskernelSizeMatchSpatial condition
            ((input->desc().dim(Dim::C) / groupSize) * (output->desc().dim(Dim::C) / groupSize)) <= 256 &&
            groupSize == 1);

#if REFERENCE_CONVOLUTION
        isSpatialConv  = false;
        isConv3x3 = false;
        isConv1x1 = false;
#endif

        if (groupSize == 1) {
            if (isFC) {
                _stageBuilder->addSwFullyConnectedStage(
                    model,
                    origStageName,
                    origLayer,
                    input,
                    weights,
                    biases,
                    scales,
                    output);
            } else {
                if (biases->usage() != DataUsage::Fake) {
                    auto tempOutput = model->duplicateData(
                        output,
                        "@temp");

                    _stageBuilder->addBiasStage(
                        model,
                        origStageName + "@biases",
                        origLayer,
                        tempOutput, biases,
                        output);

                    output = tempOutput;
                }

                if (scales->usage() != DataUsage::Fake) {
                    auto tempOutput = model->duplicateData(
                        output,
                        "@temp");

                    _stageBuilder->addScaleStage(
                        model,
                        origStageName + "@scales",
                        origLayer,
                        tempOutput, scales,
                        output);

                    output = tempOutput;
                }

                Stage swStage;
                if (isConv1x1 || isSpatialConv || isConv3x3) {
                    swStage = model->addNewStage<ConvStage>(
                        origStageName,
                        StageType::Conv,
                        origLayer,
                        {input, weights, biases, scales},
                        {output});
                } else {
                    swStage = model->addNewStage<ConvStage>(
                        origStageName,
#if REFERENCE_CONVOLUTION
                        StageType::RefConvolution,
#else
                        StageType::Im2ColConvolution,
#endif
                        origLayer,
                        {input, weights, biases, scales},
                        {output});

                    double im2ColBufSizeF = static_cast<double>(kernelSizeX) * kernelSizeY *
                        output->desc().dim(Dim::W) * output->desc().dim(Dim::H) * input->desc().dim(Dim::C)
                        + 32;

                    if (im2ColBufSizeF >= std::numeric_limits<int>::max()) {
                        VPU_THROW_EXCEPTION << "stage: " << origStageName << ", im2col bufferSize cannot fit 32s: "
                            << std::setprecision(0) << std::fixed << im2ColBufSizeF
                            << "(" << kernelSizeX << "x" << kernelSizeY << "x"
                            << output->desc().dim(Dim::W) << "x" << output->desc().dim(Dim::H) << "x" << output->desc().dim(Dim::C) << ")";
                    }

                    model->addTempBuffer(swStage, DataDesc({static_cast<int>(im2ColBufSizeF)}));
                }

                swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
                swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

                swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
                swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

                swStage->attrs().set<int>("padLeft", padLeft);
                swStage->attrs().set<int>("padRight", padRight);
                swStage->attrs().set<int>("padTop", padTop);
                swStage->attrs().set<int>("padBottom", padBottom);

                swStage->attrs().set<int>("dilationX", dilationX);
                swStage->attrs().set<int>("dilationY", dilationY);

                swStage->attrs().set<bool>("isSpatialConv", isSpatialConv);
                swStage->attrs().set<bool>("isConv1x1", isConv1x1);
                swStage->attrs().set<bool>("isConv3x3", isConv3x3);
            }
        } else if (groupSize == input->desc().dim(Dim::C) &&
                   groupSize == output->desc().dim(Dim::C)) {
            if (biases->usage() != DataUsage::Fake) {
                auto tempOutput = model->duplicateData(
                    output,
                    "@temp");

                _stageBuilder->addBiasStage(
                    model,
                    origStageName + "@biases",
                    origLayer,
                    tempOutput, biases,
                    output);

                output = tempOutput;
            }

            if (scales->usage() != DataUsage::Fake) {
                auto tempOutput = model->duplicateData(
                    output,
                    "@temp");

                _stageBuilder->addScaleStage(
                    model,
                    origStageName + "@scales",
                    origLayer,
                    tempOutput, scales,
                    output);

                output = tempOutput;
            }

            auto swStage = model->addNewStage<ConvStage>(
                origStageName,
                StageType::DepthConv,
                origLayer,
                {input, weights, biases, scales},
                {output});

            swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

            swStage->attrs().set<int>("padLeft", padLeft);
            swStage->attrs().set<int>("padRight", padRight);
            swStage->attrs().set<int>("padTop", padTop);
            swStage->attrs().set<int>("padBottom", padBottom);

            swStage->attrs().set<int>("dilationX", dilationX);
            swStage->attrs().set<int>("dilationY", dilationY);
        } else {
            VPU_THROW_EXCEPTION << "Internal error : grouped convolution was not processed";
        }
    }
}

}  // namespace

Pass::Ptr PassManager::swConvAdaptation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
