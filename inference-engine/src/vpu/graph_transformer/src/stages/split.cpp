// Copyright (C) 2018-2021 Intel Corporation
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
public:
    using StageNode::StageNode;

protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<SplitStage>(*this);
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

void FrontEnd::parseSplit(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) const {
    for (auto input : node->inputs()) {
        std::cout << "input node: " <<input.get_source_output().get_node_shared_ptr()->get_friendly_name() << std::endl;
    }
    // IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(!outputs.empty());

    const auto split = ngraph::as_type_ptr<ngraph::op::v1::Split>(node);
    VPU_THROW_UNLESS(split != nullptr, "Can't parse node with name %s and type %s. Node is nullptr", node->get_friendly_name(), node->get_type_name());
    IE_ASSERT(split != nullptr);
    auto axisNode = split->input_value(1).get_node_shared_ptr();
    const auto axisNodeConst = ngraph::as_type_ptr<ngraph::op::Constant>(axisNode);
    VPU_THROW_UNLESS(axisNodeConst != nullptr, "Can't parse node with name %s and type %s. Can't get axis node as const", node->get_friendly_name(), node->get_type_name());
    const auto axisIE = axisNodeConst->get_vector<int64_t>()[0];
    const auto input = inputs[0];

    const auto ieRevAxis = input->desc().numDims() - 1 - checked_cast<int>(axisIE); //??
    const auto defPerm = DimsOrder::fromNumDims(input->desc().numDims()).toPermutation();
    const auto axis = defPerm.at(checked_cast<size_t>(ieRevAxis));

    _stageBuilder->addSplitStage(model, split->get_name(), split, axis, input, outputs);
}

Stage StageBuilder::addSplitStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        Dim axis,
        const Data& input,
        const DataVector& outputs) {
    std::vector<DimValues> offsets;
    offsets.reserve(outputs.size());
    DimValues curOffset({{axis, 0}});

    const auto haveUnusedOutput = [](const DataVector& outputs) {
        return std::any_of(outputs.begin(), outputs.end(), [](const vpu::Data& out) {
            return out == nullptr;
        });
    };

    std::vector<size_t> outAxisSizes;
    if (haveUnusedOutput(outputs)) {
        VPU_THROW_UNLESS(node != nullptr,
            "Can't build split stage whith name {} with unused outputs when layer == nullptr", name);
        // const auto outDimsSize = node->outData[0]->getDims().size();
        // const int idx = dimToIeInd(axis, outDimsSize);
        // outAxisSizes.reserve(outDimsSize);
        // for (const auto& out : layer->outData) {
        //     VPU_THROW_UNLESS(idx <= out->getDims().size(),
        //         "Split stage with name {} and type {} can't have idx = {} when out dimensions size = {}",
        //         layer->name, layer->type, idx, out->getDims().size());
        //     outAxisSizes.push_back(out->getDims()[idx]);
        // }
    } else {
        outAxisSizes.reserve(outputs.size());
        for (const auto& output : outputs) {
            outAxisSizes.push_back(output->desc().dim(axis));
        }
    }

    vpu::DataVector usedOutputs;
    for (int i = 0; i < outputs.size(); ++i) {
        if (outputs[i] != nullptr) {
            offsets.emplace_back(curOffset);
            usedOutputs.push_back(outputs[i]);
        }
        curOffset.set(axis, curOffset[axis] + outAxisSizes[i]);
    }

    auto stage = addSplitStage(model, name, node, std::move(offsets), input, usedOutputs);

    stage->attrs().set("axis", axis);

    return stage;
}

Stage StageBuilder::addSplitStage(
        const Model& model,
        const std::string& name,
        const NodePtr& node,
        std::vector<vpu::DimValues>&& offsets,
        const Data& input,
        const DataVector& outputs) {
    IE_ASSERT(offsets.size() == outputs.size());

    auto stage = model->addNewStage<SplitStage>(
        name,
        StageType::Split,
        node,
        {input},
        outputs);

    stage->attrs().set("offsets", std::move(offsets));

    return stage;
}

}  // namespace vpu
