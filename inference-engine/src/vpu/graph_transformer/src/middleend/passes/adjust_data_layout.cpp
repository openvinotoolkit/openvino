// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/stage_builder.hpp>

#include <unordered_set>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include <set>
#include <algorithm>
#include <vector>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    Data addConvertedData(
            const Model& model,
            const Data& orig,
            DimsOrder order);

    Data addConvertedData(
            const Model& model,
            const Data& orig,
            const StridesRequirement& reqs);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(adjustDataLayout);

    //
    // Init StridesRequirement for fixed Datas
    //

    for (const auto& data : model->datas()) {
        if (data->usage() == DataUsage::Intermediate)
            continue;

        if (data->usage() == DataUsage::Input || data->usage() == DataUsage::Output) {
            if (!data->requiredStrides().fixedStrides().empty()) {
                continue;
            }
        }

        data->updateRequiredStrides(StridesRequirement::compact());
    }

    //
    // Adjust Data DimsOrder.
    //

    {
        for (const auto& stage : model->getStages()) {
            const auto& curStageInfo = stage->propagateDataOrder();

            //
            // Check inputs.
            //

            for (const auto& inEdge : stage->inputEdges()) {
                auto input = inEdge->input();

                if (!curStageInfo.hasInput(inEdge)) {
                    continue;
                }

                auto requiredOrder = curStageInfo.getInput(inEdge);

                if (input->desc().dimsOrder() == requiredOrder) {
                    continue;
                }

                auto& convertedData = input->attrs().getOrSet<DataVector>("convertedData", DataVector());

                Data newInput = nullptr;

                for (const auto& data : convertedData) {
                    if (data->desc().dimsOrder() == requiredOrder) {
                        newInput = data;
                        break;
                    }
                }

                if (newInput == nullptr) {
                    newInput = addConvertedData(model, input, requiredOrder);
                    _stageBuilder->addReorderStage(model,
                                                   formatString("%s@reorder-input-data=%d", stage->name(),
                                                                inEdge->portInd()),
                                                   stage->origNode(),
                                                   input,
                                                   newInput);
                    convertedData.emplace_back(newInput);
                }

                model->replaceStageInput(inEdge, newInput);
            }

            //
            // Check outputs.
            //

            for (const auto& outEdge : stage->outputEdges()) {
                auto output = outEdge->output();

                if (output->usage() == DataUsage::Fake) {
                    continue;
                }

                auto requiredOrder = output->desc().dimsOrder();

                if (curStageInfo.hasOutput(outEdge)) {
                    requiredOrder = curStageInfo.getOutput(outEdge);
                } else {
                    //
                    // Check consumers.
                    //

                    for (const auto& consumerEdge : output->consumerEdges()) {
                        const auto& consumerInfo = consumerEdge->consumer()->propagateDataOrder();
                        if (consumerInfo.hasInput(consumerEdge)) {
                            requiredOrder = consumerInfo.getInput(consumerEdge);
                            break;
                        }
                    }
                }

                if (output->desc().dimsOrder() == requiredOrder) {
                    continue;
                }

                auto newOutput = addConvertedData(model, output, requiredOrder);

                model->replaceStageOutput(outEdge, newOutput);

                if (output->usage() == DataUsage::Output) {
                    //
                    // It is a network output, need to insert convert stage.
                    //

                    _stageBuilder->addReorderStage(model,
                                                   formatString("%s@reorder-output-data=%d", stage->name(),
                                                                outEdge->portInd()),
                                                   stage->origNode(),
                                                   newOutput,
                                                   output);
                } else {
                    IE_ASSERT(output->usage() == DataUsage::Intermediate);

                    //
                    // Just change the order of output, its consumers will convert it if needed.
                    //

                    for (const auto& consumerEdge : output->consumerEdges()) {
                        model->replaceStageInput(consumerEdge, newOutput);
                    }
                }
            }
        }
    }

    //
    // Adjust Data strides.
    //

    {
        for (const auto& stage : model->getStages()) {
            const auto& curStageInfo = stage->getDataStridesRequirements();

            //
            // Check inputs.
            //

            for (const auto& inEdge : stage->inputEdges()) {
                auto input = inEdge->input();

                auto requiredStrides = StridesRequirement();

                if (curStageInfo.hasInput(inEdge)) {
                    requiredStrides = curStageInfo.getInput(inEdge);
                }

                if (input->checkStrides(requiredStrides)) {
                    input->updateRequiredStrides(requiredStrides);
                    continue;
                }

                auto& convertedData = input->attrs().getOrSet<DataVector>("convertedData", DataVector());

                Data newInput = nullptr;

                for (const auto& data : convertedData) {
                    if (data->desc().dimsOrder() == input->desc().dimsOrder() &&
                        data->checkStrides(requiredStrides)) {
                        newInput = data;
                        break;
                    }
                }

                if (newInput == nullptr) {
                    newInput = addConvertedData(model, input, requiredStrides);

                    _stageBuilder->addCopyStage(
                        model,
                        formatString("%s@input=%d@align-strides", stage->name(), inEdge->portInd()),
                        stage->origNode(),
                        input,
                        newInput,
                        "adjustDataLayout::input");

                    convertedData.emplace_back(newInput);
                }

                model->replaceStageInput(inEdge, newInput);
            }

            //
            // Check outputs.
            //

            for (const auto& outEdge : stage->outputEdges()) {
                auto output = outEdge->output();
                auto portInd = outEdge->portInd();

                if (output->usage() == DataUsage::Fake) {
                    continue;
                }

                auto requiredStrides = StridesRequirement();

                if (curStageInfo.hasOutput(outEdge)) {
                    requiredStrides = curStageInfo.getOutput(outEdge);
                }

                //
                // Check consumers.
                //

                for (const auto& consumerEdge : output->consumerEdges()) {
                    const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();
                    if (consumerInfo.hasInput(consumerEdge)) {
                        auto consumerRequiredStrides = consumerInfo.getInput(consumerEdge);

                        for (int i = 0; i < output->desc().numDims(); ++i) {
                            if (requiredStrides.get(i) == DimStride::Any) {
                                if (consumerRequiredStrides.get(i) != DimStride::Any) {
                                    requiredStrides.add(i, consumerRequiredStrides.get(i));
                                }
                            }
                        }
                    }
                }

                if (output->checkStrides(requiredStrides)) {
                    output->updateRequiredStrides(requiredStrides);
                    continue;
                }

                auto newOutput = addConvertedData(model, output, requiredStrides);

                model->replaceStageOutput(outEdge, newOutput);

                if (output->usage() == DataUsage::Output) {
                    //
                    // It is a network output, need to insert convert stage.
                    //

                    _stageBuilder->addCopyStage(
                        model,
                        formatString("%s@input=%d@align-strides", stage->name(), portInd),
                        stage->origNode(),
                        newOutput,
                        output,
                        "adjustDataLayout::output");
                } else {
                    IE_ASSERT(output->usage() == DataUsage::Intermediate);

                    //
                    // Just change the order of output, its consumers will convert it if needed.
                    //

                    for (const auto& consumerEdge : output->consumerEdges()) {
                        model->replaceStageInput(consumerEdge, newOutput);
                    }
                }
            }
        }
    }

    //
    // Final adjustment and check.
    //

    {
        for (const auto& stage : model->getStages()) {
            stage->finalizeDataLayout();

            const auto& orderInfo = stage->propagateDataOrder();
            const auto& strideInfo = stage->getDataStridesRequirements();

            for (const auto& inEdge : stage->inputEdges()) {
                if (orderInfo.hasInput(inEdge)) {
                    auto requiredOrder = orderInfo.getInput(inEdge);
                    IE_ASSERT(inEdge->input()->desc().dimsOrder() == requiredOrder);
                }

                if (strideInfo.hasInput(inEdge)) {
                    auto requiredStrides = strideInfo.getInput(inEdge);
                    IE_ASSERT(inEdge->input()->checkStrides(requiredStrides));
                }

                if (inEdge->input()->usage() == DataUsage::Const) {
                    IE_ASSERT(inEdge->input()->checkStrides(StridesRequirement::compact()));
                }
            }

            for (const auto& outEdge : stage->outputEdges()) {
                if (orderInfo.hasOutput(outEdge)) {
                    auto requiredOrder = orderInfo.getOutput(outEdge);
                    IE_ASSERT(outEdge->output()->desc().dimsOrder() == requiredOrder);
                }

                if (strideInfo.hasOutput(outEdge)) {
                    auto requiredStrides = strideInfo.getOutput(outEdge);
                    IE_ASSERT(outEdge->output()->checkStrides(requiredStrides));
                }
            }
        }
    }
}

Data PassImpl::addConvertedData(
        const Model& model,
        const Data& orig,
        DimsOrder order) {
    auto newDesc = orig->desc();
    newDesc.reorder(order);

    return model->duplicateData(
        orig,
        formatString("@order=%s", order),
        newDesc);
}

Data PassImpl::addConvertedData(
        const Model& model,
        const Data& orig,
        const StridesRequirement& reqs) {
    auto data = orig;

    if (orig->usage() == DataUsage::Const) {
        auto newData = model->addNewData(orig->name(), orig->desc());
        newData->attrs().copyFrom(orig->attrs());
        data = newData;
    } else {
        data = model->duplicateData(orig, "@adjust-strides");
    }

    data->resetRequiredStrides();
    data->updateRequiredStrides(reqs);

    return data;
}

}  // namespace

Pass::Ptr PassManager::adjustDataLayout() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
