// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <limits>
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <unordered_set>

#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace {

class ConcatStage final : public StageNode {
protected:
    StagePtr cloneImpl() const override {
        return std::make_shared<ConcatStage>(*this);
    }

    DataMap<float> propagateScaleFactorsImpl(
            const DataMap<float>& inputScales,
            ScalePropagationStep step) override {
        IE_ASSERT(!_inputEdges.empty());
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        DataMap<float> out;

        if (step == ScalePropagationStep::Propagate) {
            // Keep the largest input scale factor.
            auto maxScale = std::numeric_limits<float>::lowest();
            for (const auto& inEdge : _inputEdges) {
                maxScale = std::max(maxScale, inputScales.at(inEdge->input()));
            }

            IE_ASSERT(maxScale > 0.0f);

            for (const auto& inEdge : _inputEdges) {
                auto curScale = inputScales.at(inEdge->input());

                if (!isFloatEqual(curScale, maxScale)) {
                    out[inEdge->input()] = maxScale / curScale;
                }
            }

            out[output] = maxScale;
        } else {
            // Concat can only propagate scaling.
            for (const auto& inEdge : _inputEdges) {
                out[inEdge->input()] = 1.0f;
            }

            out[output] = 1.0f;
        }

        return out;
    }

    DataMap<DimsOrder> propagateDataOrderImpl() const override {
        IE_ASSERT(!_inputEdges.empty());
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        DimsOrderMap<int> dimsOrderVotes;
        for (const auto& inEdge : _inputEdges) {
            dimsOrderVotes[inEdge->input()->desc().dimsOrder()]++;
        }

        // Select DimsOrder with most votes.
        // For equal votes : HCW > CHW > HWC.

        DimsOrder finalOrder;
        int curVotes = -1;
        for (const auto& p : dimsOrderVotes) {
            if (p.second > curVotes) {
                finalOrder = p.first;
                curVotes = p.second;
            } else if (p.second == curVotes) {
                if (p.first.numDims() >= 3) {
                    if (p.first.dimInd(Dim::C) == 2) {
                        finalOrder = p.first;
                    } else if (p.first.dimInd(Dim::C) == 3 &&
                               finalOrder.dimInd(Dim::C) != 2) {
                        finalOrder = p.first;
                    }
                }
            }
        }

        IE_ASSERT(finalOrder.numDims() > 0);
        IE_ASSERT(curVotes > 0);

        DataMap<DimsOrder> out;

        for (const auto& inEdge : _inputEdges) {
            out[inEdge->input()] = finalOrder;
        }

        out[output] = finalOrder;

        return out;
    }

    DataMap<StridesRequirement> getDataStridesRequirementsImpl() const override {
        IE_ASSERT(!_inputEdges.empty());
        IE_ASSERT(_outputEdges.size() == 1);

        auto output = _outputEdges[0]->output();

        auto dimsOrder = output->desc().dimsOrder();

        //
        // Get smallest Dim over which Concat is done.
        //

        auto minConcatDimInd = dimsOrder.numDims();

        for (const auto& inEdge : _inputEdges) {
            auto input = inEdge->input();

            for (const auto& p : output->desc().dims()) {
                if (input->desc().dim(p.first) != p.second) {
                    minConcatDimInd = std::min(minConcatDimInd, dimsOrder.dimInd(p.first));
                }
            }
        }

        IE_ASSERT(minConcatDimInd < dimsOrder.numDims());

        //
        // Initial StridesRequirement for inputs and output.
        //

        auto outputReqs = output->requiredStrides();

        auto inputReqs = outputReqs;
        for (int i = minConcatDimInd + 1; i < dimsOrder.numDims(); ++i) {
            inputReqs.remove(i);
        }

        //
        // Merge input StridesRequirement.
        //

        for (const auto& inEdge : _inputEdges) {
            auto curInput = inEdge->input();
            auto curInputReqs = curInput->requiredStrides();

            for (int i = 0; i < minConcatDimInd + 1; ++i) {
                if (outputReqs.get(i) == DimStride::Any) {
                    if (curInputReqs.get(i) != DimStride::Any) {
                        inputReqs.add(i, curInputReqs.get(i));
                        outputReqs.add(i, curInputReqs.get(i));
                    }
                }
            }
        }

        //
        // Merge output consumers StridesRequirement.
        //

        for (const auto& consumer : output->consumers()) {
            auto consumerInfo = consumer->getDataStridesRequirements();

            auto consumerStrideIt = consumerInfo.find(output);
            if (consumerStrideIt != consumerInfo.end()) {
                auto consumerReqs = consumerStrideIt->second;

                for (int i = 0; i < minConcatDimInd + 1; ++i) {
                    if (outputReqs.get(i) == DimStride::Any) {
                        if (consumerReqs.get(i) != DimStride::Any) {
                            inputReqs.add(i, consumerReqs.get(i));
                            outputReqs.add(i, consumerReqs.get(i));
                        }
                    }
                }
            }
        }

        //
        // Return merged StridesRequirement.
        //

        DataMap<StridesRequirement> out;

        for (const auto& inEdge : _inputEdges) {
            auto input = inEdge->input();
            out[input] = inputReqs;
        }
        out[output] = outputReqs;

        return out;
    }

    void finalizeDataLayoutImpl() override {
    }

    DataMap<BatchSupport> getBatchSupportInfoImpl() const override {
        return DataMap<BatchSupport>();
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void serializeDataImpl(BlobSerializer&) const override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }
};

}  // namespace

void FrontEnd::parseConcat(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& _layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(!inputs.empty());
    IE_ASSERT(outputs.size() == 1);

    auto output = outputs[0];

    auto layer = std::dynamic_pointer_cast<ie::ConcatLayer>(_layer);
    IE_ASSERT(layer != nullptr);

    IE_ASSERT(layer->_axis < output->desc().numDims());

    auto perm = DimsOrder::fromNumDims(output->desc().numDims()).toPermutation();
    auto axis = perm[output->desc().numDims() - 1 - layer->_axis];

    _stageBuilder->addConcatStage(model, layer->name, layer, axis, inputs, output);
}

Stage StageBuilder::addConcatStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        Dim axis,
        const DataVector& inputs,
        const Data& output) {
    std::vector<DimValues> offsets;

    DimValues curOffset({{axis, 0}});
    for (const auto& input : inputs) {
        offsets.emplace_back(curOffset);
        curOffset.set(axis, curOffset[axis] + input->desc().dim(axis));
    }

    auto stage = addConcatStage(model, name, layer, offsets, inputs, output);

    stage->attrs().set("axis", axis);

    return stage;
}

Stage StageBuilder::addConcatStage(
        const Model::Ptr& model,
        const std::string& name,
        const ie::CNNLayerPtr& layer,
        const std::vector<DimValues>& offsets,
        const DataVector& inputs,
        const Data& output) {
    IE_ASSERT(offsets.size() == inputs.size());

    auto stage = model->addNewStage<ConcatStage>(
        name,
        StageType::Concat,
        layer,
        inputs,
        {output});

    stage->attrs().set<std::vector<DimValues>>("offsets", offsets);

    return stage;
}

}  // namespace vpu
