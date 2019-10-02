// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <limits>
#include <memory>
#include <list>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

Stage StageBuilder::addScalingStage(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& origLayer,
        float scale,
        const Data& input,
        const Data& output) {
    if (input->desc().type() != DataType::FP16) {
        VPU_THROW_EXCEPTION << "Can't adjust non-FP16 data " << input->name();
    }

    if (output->desc().type() != DataType::FP16) {
        VPU_THROW_EXCEPTION << "Can't adjust non-FP16 data " << output->name();
    }

    if (input->desc().dimsOrder() != output->desc().dimsOrder()) {
        VPU_THROW_EXCEPTION << input->name() << " and " << output->name() << " have different layout";
    }

    return addPowerStage(model, input->name() + "@SCALE=" + std::to_string(scale), origLayer, scale, 1.0f, 0.0f, input, output);
}

namespace {

SmallVector<float> getInputScales(const Stage& stage) {
    SmallVector<float> out(stage->numInputs());
    for (const auto& inputEdge : stage->inputEdges()) {
        auto scaleFactor = inputEdge->input()->attrs().getOrDefault<float>("scaleFactor", 1.0f);
        out[inputEdge->portInd()] = scaleFactor;
    }
    return out;
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(propagateDataScale);

    const auto& env = CompileEnv::get();

    //
    // Get required SCALE factors per data
    //

    if (env.netConfig.hasManualDataScale()) {
        bool hasScaleReqs = false;

        for (auto info : env.netConfig.dataScale()) {
            auto name = info.first;
            auto scale = info.second;

            Data curData;
            for (const auto& data : model->datas()) {
                if (data->name() == name) {
                    curData = data;
                    break;
                }
            }
            if (curData == nullptr) {
                VPU_THROW_EXCEPTION << "There is no such data : " << name << " in network " << model->name();
            }

            if (curData->usage() != DataUsage::Input &&
                curData->usage() != DataUsage::Intermediate) {
                VPU_THROW_EXCEPTION
                        << "Scale can be used only for input and intermediate data, got "
                        << curData->name() << " as " << curData->usage();
            }

            if (curData->attrs().has("requestedScale")) {
                VPU_THROW_EXCEPTION << "Data " << name << " is mentioned twice";
            }
            if (!isFloatEqual(scale, 1.0f)) {
                hasScaleReqs = true;
                curData->attrs().set<float>("requestedScale", scale);
            }
        }

        if (!hasScaleReqs) {
            return;
        }
    } else {
        auto inputShift = model->attrs().getOrDefault<int>("inputShift", 1);
        if (inputShift == 1) {
            return;
        }

        float inputScale =  1 << inputShift;
        for (const auto& data : model->datas()) {
            if (data->usage() != DataUsage::Input)
                continue;

            data->attrs().set<float>("requestedScale", inputScale);
        }
    }

    //
    // Traverse stages
    //
    // - add SCALE for input if required
    // - propagate SCALE to next stages if possible
    // - undo SCALE if the stage doesn't support it
    //

    for (const auto& stage : model->getStages()) {
        //
        // Check if we need to add SCALE to input
        //

        bool scalesWasInitialized = false;

        for (const auto& inEdge : stage->inputEdges()) {
            auto input = inEdge->input();

            if (!input->attrs().has("requestedScale")) {
                // No SCALE requested.
                continue;
            }

            auto requestedScale = input->attrs().get<float>("requestedScale");
            auto curScaleFactor = input->attrs().getOrDefault<float>("scaleFactor", 1.0f);

            if (isFloatEqual(curScaleFactor, requestedScale)) {
                // We already added SCALE to this data.
                continue;
            }

            auto scaleMultiplier = requestedScale / curScaleFactor;

            //
            // Some stages can SCALE input internally, check them first
            //

            if (input->numConsumers() == 1) {
                auto inputScales = getInputScales(stage);
                inputScales[inEdge->portInd()] = scaleMultiplier;

                const auto& checkScales = stage->propagateScaleFactors(inputScales, ScalePropagationStep::Check);
                if (!checkScales.hasInput(inEdge)) {
                    const auto& finalScales = stage->propagateScaleFactors(inputScales, ScalePropagationStep::ScaleInput);

                    for (const auto& constInEdge : stage->inputEdges()) {
                        auto constInput = constInEdge->input();

                        if (!finalScales.hasInput(constInEdge)) {
                            continue;
                        }

                        auto curScaleFactor = constInput->attrs().getOrDefault<float>("scaleFactor", 1.0f);
                        auto newScaleFactor = finalScales.getInput(constInEdge);

                        if (isFloatEqual(curScaleFactor, newScaleFactor))
                            continue;

                        IE_ASSERT(constInput->usage() == DataUsage::Const);

                        auto scaleCoeff = newScaleFactor / curScaleFactor;

                        auto& scaledChildren = constInput->attrs().getOrSet<DataVector>("scaledChildren", DataVector());

                        Data scaledConstInput;
                        for (const auto& scaledChild : scaledChildren) {
                            auto childScaleFactor = scaledChild->attrs().getOrDefault<float>("scaleFactor", 1.0f);
                            if (isFloatEqual(childScaleFactor, newScaleFactor)) {
                                scaledConstInput = scaledChild;
                                break;
                            }
                        }
                        if (scaledConstInput == nullptr) {
                            scaledConstInput = model->duplicateData(
                                constInput,
                                formatString("@SCALE=%f", scaleCoeff),
                                constInput->desc(),
                                scaleContent(constInput->content(), scaleCoeff));

                            scaledChildren.emplace_back(scaledConstInput);
                        }

                        model->replaceStageInput(constInEdge, scaledConstInput);

                        scaledConstInput->attrs().set<float>("scaleFactor", newScaleFactor);
                    }
                    for (const auto& outEdge : stage->outputEdges()) {
                        outEdge->output()->attrs().set<float>("scaleFactor", finalScales.getOutput(outEdge));
                    }

                    scalesWasInitialized = true;
                    break;
                }
            }

            //
            // Add explicit scaling stage
            //

            auto newInput = model->duplicateData(
                input,
                formatString("@SCALE=%f", requestedScale));

            newInput->attrs().set<float>("scaleFactor", requestedScale);

            for (const auto& consumerEdge : input->consumerEdges()) {
                model->replaceStageInput(consumerEdge, newInput);
            }

            _stageBuilder->addScalingStage(model, stage->origLayer(), scaleMultiplier, input, newInput);
        }

        if (scalesWasInitialized)
            continue;

        //
        // Propagate SCALE from inputs to outputs
        //

        const auto& finalScales = stage->propagateScaleFactors(getInputScales(stage), ScalePropagationStep::Propagate);

        for (const auto& inEdge : stage->inputEdges()) {
            auto input = inEdge->input();

            if (!finalScales.hasInput(inEdge)) {
                continue;
            }

            auto curScaleFactor = input->attrs().getOrDefault<float>("scaleFactor", 1.0f);
            auto newScaleFactor = finalScales.getInput(inEdge);

            if (isFloatEqual(curScaleFactor, newScaleFactor))
                continue;

            auto scaleCoeff = newScaleFactor / curScaleFactor;

            Data scaledInput;
            if (input->usage() == DataUsage::Const) {
                auto& scaledChildren = input->attrs().getOrSet<DataVector>("scaledChildren", DataVector());

                for (const auto& scaledChild : scaledChildren) {
                    auto childScaleFactor = scaledChild->attrs().getOrDefault<float>("scaleFactor", 1.0f);
                    if (isFloatEqual(childScaleFactor, newScaleFactor)) {
                        scaledInput = scaledChild;
                        break;
                    }
                }

                if (scaledInput == nullptr) {
                    scaledInput = model->duplicateData(
                        input,
                        formatString("@SCALE=%f", scaleCoeff),
                        input->desc(),
                        scaleContent(input->content(), scaleCoeff));

                    scaledChildren.emplace_back(scaledInput);
                }
            } else {
                scaledInput = model->duplicateData(
                    input,
                    formatString("@SCALE=%f", scaleCoeff));

                _stageBuilder->addScalingStage(model, stage->origLayer(), scaleCoeff, input, scaledInput);
            }
            IE_ASSERT(scaledInput != nullptr);

            model->replaceStageInput(inEdge, scaledInput);

            scaledInput->attrs().set<float>("scaleFactor", newScaleFactor);
        }
        for (const auto& outEdge : stage->outputEdges()) {
            outEdge->output()->attrs().set<float>("scaleFactor", finalScales.getOutput(outEdge));
        }
    }

    //
    // Remove SCALE from network outputs
    //

    for (auto output : model->datas()) {
        if (output->usage() != DataUsage::Output) {
            continue;
        }

        auto outputScale = output->attrs().getOrDefault<float>("scaleFactor", 1.0f);
        if (isFloatEqual(outputScale, 1.0f)) {
            continue;
        }

        if (output->desc().type() != DataType::FP16) {
            output = output->attrs().get<Data>("fp16_copy");
            IE_ASSERT(output != nullptr);
            IE_ASSERT(output->desc().type() == DataType::FP16);
        }

        auto newData = model->duplicateData(
            output,
            formatString("@SCALE=%f", outputScale));

        newData->attrs().set<float>("scaleFactor", outputScale);
        output->attrs().set<float>("scaleFactor", 1.0f);

        auto producerEdge = output->producerEdge();
        IE_ASSERT(producerEdge != nullptr);
        model->replaceStageOutput(producerEdge, newData);

        IE_ASSERT(output->numConsumers() == 0);

        _stageBuilder->addScalingStage(
            model,
            nullptr,
            1.0f / outputScale,
            newData,
            output);
    }
}

}  // namespace

Pass::Ptr PassManager::propagateDataScale() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
