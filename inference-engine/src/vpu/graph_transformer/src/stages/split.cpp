// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <utility>

namespace vpu {

namespace {

class SplitStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<SplitStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo) override {
        if (step == ScalePropagationStep::Propagate) {
            auto inputScale = inputScales[0];

            for (const auto& outEdge : outputEdges()) {
                scaleInfo.setOutput(outEdge, inputScale);
            }
        } else {
            // Split can only propagate scaling.
            scaleInfo.setInput(inputEdge(0), 1.0f);

            for (const auto& outEdge : outputEdges()) {
                scaleInfo.setOutput(outEdge, 1.0f);
            }
        }
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();

        for (const auto& outEdge : outputEdges()) {
            orderInfo.setOutput(outEdge, input->desc().dimsOrder());
        }
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto input = inputEdge(0)->input();

        auto dimsOrder = input->desc().dimsOrder();

        //
        // Get smallest Dim over which Split is done.
        //

        auto minSplitDimInd = dimsOrder.numDims();

        for (const auto& outEdge : outputEdges()) {
            auto output = outEdge->output();

            for (const auto& p : input->desc().dims()) {
                if (output->desc().dim(p.first) != p.second) {
                    minSplitDimInd = std::min(minSplitDimInd, dimsOrder.dimInd(p.first));
                }
            }
        }

        //
        // Initial StridesRequirement for inputs and output.
        //

        auto inputReqs = input->requiredStrides();

        auto outputReqs = inputReqs;

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& outEdge : outputEdges()) {
            auto curOutput = outEdge->output();

            for (const auto& consumerEdge : curOutput->consumerEdges()) {
                const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                if (consumerInfo.hasInput(consumerEdge)) {
                    const auto& consumerReqs = consumerInfo.getInput(consumerEdge);

                    for (int i = 0; i < dimsOrder.numDims(); ++i) {
                        if (inputReqs.get(i) == DimStride::Any) {
                            if (consumerReqs.get(i) != DimStride::Any) {
                                inputReqs.add(i, consumerReqs.get(i));
                                outputReqs.add(i, consumerReqs.get(i));
                            }
                        }
                    }
                }
            }
        }

        //
        // Remove extra output StridesRequirement.
        //

        for (int i = minSplitDimInd + 1; i < dimsOrder.numDims(); ++i) {
            outputReqs.remove(i);
        }

        //
        // Return merged StridesRequirements.
        //

        stridesInfo.setInput(inputEdge(0), inputReqs);
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, outputReqs);
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& /*batchInfo*/) override {
    }

    void initialCheckImpl() const override {
        IE_ASSERT(numInputs() == 1);
        IE_ASSERT(numOutputs() > 0);
        const auto& firstInputPrecision = input(0)->desc().type();
        assertAllInputsOutputsTypes(this, {firstInputPrecision}, {firstInputPrecision});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

void FrontEnd::parseSplit(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(!outputs.empty());

    auto layer = std::dynamic_pointer_cast<ie::SplitLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    auto input = inputs[0];

    auto inDesc = input->desc();
    auto perm = inDesc.dimsOrder().toPermutation();

    // Detect unused data
    DataVector onlyUsedOutputs;
    for (const auto& output : outputs) {
        if (!output->origData()->getInputTo().empty()) {
            onlyUsedOutputs.push_back(output);
        }
    }
    IE_ASSERT(!onlyUsedOutputs.empty());

    // Check whether it is Split(copy) or Slice Caffe layer
    // and we do not trust to IE layer type value.
    bool isSplit = true;

    for (const auto& output : outputs) {
        for (int i = 0; i < perm.size(); ++i) {
            if (inDesc.dim(perm[i]) != output->desc().dim(perm[i])) {
                isSplit = false;
                break;
            }
        }

        if (!isSplit)
            break;
    }

    if (isSplit) {
        // Split is just a re-usage of the input Data.

        for (int i = 0; i < outputs.size(); ++i) {
            auto output = outputs[i];

            IE_ASSERT(output->numConsumers() == 0);

            if (output->usage() == DataUsage::Output) {
                _stageBuilder->addCopyStage(
                    model,
                    formatString("%s@copy=%d/%d", layer->name, i + 1, outputs.size()),
                    layer,
                    input,
                    output);
            } else {
                IE_ASSERT(output->usage() == DataUsage::Intermediate);

                bindData(input, output->origData());
            }
        }
    } else {
        // Calculate target axis for slicing.

        DimValues sumDims;
        for (int i = 0; i < perm.size(); ++i) {
            sumDims.set(perm[i], 0);
        }
        for (const auto& output : outputs) {
            for (auto& p : sumDims) {
                p.second += output->desc().dim(p.first);
            }
        }

        auto axis = Dim::Invalid;
        for (const auto& p : sumDims) {
            if (inDesc.dim(p.first) == p.second) {
                axis = p.first;
                break;
            }
        }

        IE_ASSERT(axis != Dim::Invalid);

        for (int i = 0; i < perm.size(); ++i) {
            if (perm[i] == axis) {
                continue;
            }

            for (const auto& output : outputs) {
                IE_ASSERT(inDesc.dim(perm[i]) == output->desc().dim(perm[i]));
            }
        }

        _stageBuilder->addSplitStage(model, layer->name, layer, axis, input, onlyUsedOutputs);
    }
}

Stage StageBuilder::addSplitStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        Dim axis,
        const Data& input,
        const DataVector& outputs) {
    std::vector<DimValues> offsets;
    offsets.reserve(outputs.size());

    DimValues curOffset({{axis, 0}});
    for (const auto& output : outputs) {
        offsets.emplace_back(curOffset);
        curOffset.set(axis, curOffset[axis] + output->desc().dim(axis));
    }

    auto stage = addSplitStage(model, name, layer, std::move(offsets), input, outputs);

    stage->attrs().set("axis", axis);

    return stage;
}

Stage StageBuilder::addSplitStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        std::vector<vpu::DimValues>&& offsets,
        const Data& input,
        const DataVector& outputs) {
    IE_ASSERT(offsets.size() == outputs.size());

    auto stage = model->addNewStage<SplitStage>(
        name,
        StageType::Split,
        layer,
        {input},
        outputs);

    stage->attrs().set("offsets", std::move(offsets));

    return stage;
}

}  // namespace vpu
