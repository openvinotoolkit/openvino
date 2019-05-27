// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <unordered_set>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include <set>
#include <algorithm>

namespace vpu {

namespace {

class ConvertOrderStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConvertOrderStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        return DataMap<DimsOrder>();
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        return DataMap<StridesRequirement>();
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::CanBeLimited;
    }

    void finalCheckImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inDimsOrder = input->desc().dimsOrder();
        auto outDimsOrder = output->desc().dimsOrder();
        IE_ASSERT(inDimsOrder.numDims() == outDimsOrder.numDims());
        IE_ASSERT(isOrdersCompatible(inDimsOrder, outDimsOrder));

        for (const auto& p : input->desc().dims()) {
            IE_ASSERT(p.second == output->desc().dim(p.first));
        }
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto inDimsOrder = input->desc().dimsOrder();
        auto outDimsOrder = output->desc().dimsOrder();

        IE_ASSERT(inDimsOrder.numDims() == outDimsOrder.numDims());
        IE_ASSERT(isOrdersCompatible(inDimsOrder, outDimsOrder));

        for (const auto& p : input->desc().dims()) {
            IE_ASSERT(p.second == output->desc().dim(p.first));
        }

        auto operm = output->desc().dimsOrder().toPermutation();
        auto iind = input->desc().dimsOrder().toIndices();
        IE_ASSERT(operm.size() == iind.size());

        int i = 0;
        for (; i < input->desc().numDims(); i++) {
            serializer.append(static_cast<uint32_t>(iind[operm[i]]));
        }
        for (; i < MAX_DIMS_32; i++) {
            serializer.append(static_cast<uint32_t>(-1));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);
        IE_ASSERT(_tempBufferEdges.empty());

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    Data addConvertedData(
            const Model::Ptr& model,
            const Data& orig,
            DimsOrder order);

    Data addConvertedData(
            const Model::Ptr& model,
            const Data& orig,
            const StridesRequirement& reqs);

    void convertDataLayout(
            const Model::Ptr& model,
            const Stage& baseStage,
            const std::string& postfix,
            const Data& input,
            const Data& output);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(adjustDataLayout);

    //
    // Init StridesRequirement for fixed Datas
    //

    for (const auto& data : model->datas()) {
        if (data->usage() == DataUsage::Intermediate)
            continue;

        data->updateRequiredStrides(StridesRequirement::compact());
    }

    //
    // Adjust Data DimsOrder.
    //

    {
        for (const auto& stage : model->getStages()) {
            auto curStageInfo = stage->propagateDataOrder();

            //
            // Check inputs.
            //

            for (const auto& inEdge : stage->inputEdges()) {
                auto input = inEdge->input();

                auto orderIt = curStageInfo.find(input);
                if (orderIt == curStageInfo.end()) {
                    continue;
                }

                auto requiredOrder = orderIt->second;

                if (input->desc().dimsOrder() == requiredOrder) {
                    continue;
                }

                auto& convertedData = input->attrs().getOrSet<DataVector>("convertedData", DataVector());

                Data newInput;

                for (const auto& data : convertedData) {
                    if (data->desc().dimsOrder() == requiredOrder) {
                        newInput = data;
                        break;
                    }
                }

                if (newInput == nullptr) {
                    newInput = addConvertedData(model, input, requiredOrder);
                    convertDataLayout(model, stage, formatString("input=%d", inEdge->portInd()), input, newInput);
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

                auto requiredOrder = output->desc().dimsOrder();

                auto orderIt = curStageInfo.find(output);
                if (orderIt != curStageInfo.end()) {
                    requiredOrder = orderIt->second;
                } else {
                    //
                    // Check consumers.
                    //

                    for (const auto& consumer : output->consumers()) {
                        auto consumerInfo = consumer->propagateDataOrder();
                        auto consumerOrderIt = consumerInfo.find(output);
                        if (consumerOrderIt != consumerInfo.end()) {
                            requiredOrder = consumerOrderIt->second;
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

                    convertDataLayout(model, stage, formatString("output=%d", portInd), newOutput, output);
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
            auto curStageInfo = stage->getDataStridesRequirements();

            //
            // Check inputs.
            //

            for (const auto& inEdge : stage->inputEdges()) {
                auto input = inEdge->input();

                auto requiredStrides = StridesRequirement();

                auto strideIt = curStageInfo.find(input);
                if (strideIt != curStageInfo.end()) {
                    requiredStrides = strideIt->second;
                }

                if (input->checkStrides(requiredStrides)) {
                    input->updateRequiredStrides(requiredStrides);
                    continue;
                }

                auto& convertedData = input->attrs().getOrSet<DataVector>("convertedData", DataVector());

                Data newInput;

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
                        stage->origLayer(),
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
                auto portInd = outEdge->portInd();

                auto requiredStrides = StridesRequirement();

                auto strideIt = curStageInfo.find(output);

                if (strideIt != curStageInfo.end()) {
                    requiredStrides = strideIt->second;
                }

                //
                // Check consumers.
                //

                for (const auto& consumer : output->consumers()) {
                    auto consumerInfo = consumer->getDataStridesRequirements();
                    auto consumerStrideIt = consumerInfo.find(output);
                    if (consumerStrideIt != consumerInfo.end()) {
                        auto consumerRequiredStrides = consumerStrideIt->second;

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
                        stage->origLayer(),
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
    // Final adjustment and check.
    //

    {
        for (const auto& stage : model->getStages()) {
            stage->finalizeDataLayout();

            auto requiredOrder = stage->propagateDataOrder();
            auto requiredStrides = stage->getDataStridesRequirements();

            for (const auto& input : stage->inputs()) {
                auto orderIt = requiredOrder.find(input);
                if (orderIt != requiredOrder.end()) {
                    auto requiredOrder = orderIt->second;
                    IE_ASSERT(input->desc().dimsOrder() == requiredOrder);
                }

                auto strideIt = requiredStrides.find(input);
                if (strideIt != requiredStrides.end()) {
                    auto requiredStrides = strideIt->second;
                    IE_ASSERT(input->checkStrides(requiredStrides));
                }

                if (input->usage() == DataUsage::Const) {
                    IE_ASSERT(input->checkStrides(StridesRequirement::compact()));
                }
            }

            for (const auto& output : stage->outputs()) {
                auto orderIt = requiredOrder.find(output);
                if (orderIt != requiredOrder.end()) {
                    auto requiredOrder = orderIt->second;
                    IE_ASSERT(output->desc().dimsOrder() == requiredOrder);
                }

                auto strideIt = requiredStrides.find(output);
                if (strideIt != requiredStrides.end()) {
                    auto requiredStrides = strideIt->second;
                    IE_ASSERT(output->checkStrides(requiredStrides));
                }
            }
        }
    }
}

Data PassImpl::addConvertedData(
        const Model::Ptr& model,
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
        const Model::Ptr& model,
        const Data& orig,
        const StridesRequirement& reqs) {
    auto data = model->duplicateData(
        orig,
        "@adjust-strides");
    data->resetRequiredStrides();
    data->updateRequiredStrides(reqs);

    return data;
}

void PassImpl::convertDataLayout(
        const Model::Ptr& model,
        const Stage& baseStage,
        const std::string& postfix,
        const Data& input,
        const Data& output) {
    IE_ASSERT(input->desc().dims() == output->desc().dims());

    model->addNewStage<ConvertOrderStage>(
        formatString("%s@%s@reorder=%s", baseStage->name(), postfix, output->desc().dimsOrder()),
        StageType::Permute,
        baseStage->origLayer(),
        {input},
        {output});
}

}  // namespace

Pass::Ptr PassManager::adjustDataLayout() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
