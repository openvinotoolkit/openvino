// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <set>

#include <vpu/sw/post_op_stage.hpp>

namespace vpu {

namespace {

class ArgMaxStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ArgMaxStage>(*this);
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto has_axis = attrs().get<bool>("has_axis");
        if (has_axis) {
            _orderInfo.setOutput(_outputEdges[0], input->desc().dimsOrder());
        } else {
            // axis<0 requires flatten so only NCHW layout is supported
            _orderInfo.setInput(_inputEdges[0], DimsOrder::fromNumDims(input->desc().numDims()));
            _orderInfo.setOutput(_outputEdges[0], DimsOrder::fromNumDims(output->desc().numDims()));
        }
    }

    void getDataStridesRequirementsImpl() const override {
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const override {
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        auto out_max_val = attrs().get<int32_t>("out_max_val");
        auto top_k = attrs().get<int32_t>("top_k");
        auto has_axis = attrs().get<bool>("has_axis");
        int axis_index = -1;
        if (has_axis) {
            auto axis = attrs().get<Dim>("axis");
            axis_index = input->desc().dimsOrder().dimInd(axis);
        }

        // NOTE: when axis is passed into VPU, it's not an index, but a name
        //       with meaning like 0:N, 1:C, 2:H, 3;W
        serializer.append(static_cast<int32_t>(out_max_val));
        serializer.append(static_cast<int32_t>(top_k));
        serializer.append(static_cast<int32_t>(axis_index));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        input->serializeNewBuffer(serializer);
        output->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseArgMax(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 1);
    IE_ASSERT(outputs.size() == 1);

    auto stage = model->addNewStage<ArgMaxStage>(
        layer->name,
        StageType::ArgMax,
        layer,
        inputs,
        outputs);

    stage->attrs().set<int32_t>("out_max_val", layer->GetParamAsInt("out_max_val"));
    stage->attrs().set<int32_t>("top_k", layer->GetParamAsInt("top_k"));

    int axis = layer->GetParamAsInt("axis", -1);
    if (axis >= 0) {
        auto perm = DimsOrder::fromNumDims(inputs[0]->desc().numDims()).toPermutation();
        auto axisDim = perm[outputs[0]->desc().numDims() - 1 - axis];
        stage->attrs().set<bool>("has_axis", true);
        stage->attrs().set<Dim>("axis", axisDim);
    } else {
        stage->attrs().set<bool>("has_axis", false);
    }
}

}  // namespace vpu
