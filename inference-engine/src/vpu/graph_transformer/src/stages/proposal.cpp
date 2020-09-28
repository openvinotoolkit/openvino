// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>
#include <vpu/compile_env.hpp>

#include <caseless.hpp>

namespace vpu {

namespace {

class ProposalStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ProposalStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();

        orderInfo.setInput(inputEdge(0), input0->desc().dimsOrder().createMovedDim(Dim::C, 2));
        orderInfo.setInput(inputEdge(1), input1->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setInput(inputEdge(0), StridesRequirement::compact());
        stridesInfo.setInput(inputEdge(1), StridesRequirement::compact());
        stridesInfo.setInput(inputEdge(2), StridesRequirement::compact());
        stridesInfo.setOutput(outputEdge(0), StridesRequirement::compact());
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto feat_stride = attrs().get<int>("feat_stride");
        auto base_size = attrs().get<int>("base_size");
        auto min_size = attrs().get<int>("min_size");
        auto pre_nms_topn = attrs().get<int>("pre_nms_topn");
        auto post_nms_topn = attrs().get<int>("post_nms_topn");
        auto nms_thresh = attrs().get<float>("nms_thresh");
        auto pre_nms_thresh = attrs().get<float>("pre_nms_thresh");
        auto box_size_scale = attrs().get<float>("box_size_scale");
        auto box_coordinate_scale = attrs().get<float>("box_coordinate_scale");
        auto coordinates_offset = attrs().get<float>("coordinates_offset");
        auto initial_clip = attrs().get<bool>("initial_clip");
        auto clip_before_nms = attrs().get<bool>("clip_before_nms");
        auto clip_after_nms = attrs().get<bool>("clip_after_nms");
        auto normalize = attrs().get<bool>("normalize");

        auto shift_anchors = attrs().get<bool>("shift_anchors");
        auto round_ratios = attrs().get<bool>("round_ratios");
        auto swap_xy = attrs().get<bool>("swap_xy");
        const auto& scales = attrs().get<std::vector<float>>("scales");
        const auto& ratios = attrs().get<std::vector<float>>("ratios");

        serializer.append(static_cast<uint32_t>(feat_stride));
        serializer.append(static_cast<uint32_t>(base_size));
        serializer.append(static_cast<uint32_t>(min_size));
        serializer.append(static_cast<int32_t>(pre_nms_topn));
        serializer.append(static_cast<int32_t>(post_nms_topn));
        serializer.append(static_cast<float>(nms_thresh));
        serializer.append(static_cast<float>(pre_nms_thresh));
        serializer.append(static_cast<float>(box_size_scale));
        serializer.append(static_cast<float>(box_coordinate_scale));
        serializer.append(static_cast<float>(coordinates_offset));
        serializer.append(static_cast<uint32_t>(initial_clip));
        serializer.append(static_cast<uint32_t>(clip_before_nms));
        serializer.append(static_cast<uint32_t>(clip_after_nms));
        serializer.append(static_cast<uint32_t>(normalize));
        serializer.append(static_cast<uint32_t>(shift_anchors));
        serializer.append(static_cast<uint32_t>(round_ratios));
        serializer.append(static_cast<uint32_t>(swap_xy));

        auto serializeVector = [&serializer](const std::vector<float>& array) {
            serializer.append(static_cast<uint32_t>(array.size()));
            for (auto elem : array) {
                serializer.append(static_cast<float>(elem));
            }
        };

        serializeVector(scales);
        serializeVector(ratios);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input0 = inputEdge(0)->input();
        auto input1 = inputEdge(1)->input();
        auto input2 = inputEdge(2)->input();
        auto output = outputEdge(0)->output();

        input0->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        tempBuffer(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseProposal(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(inputs.size() == 3);

    // TODO: implement 2nd output, see:
    // #-37327: Several models Failed to compile layer "proposals"
    IE_ASSERT(outputs.size() == 1 || outputs.size() == 2);

    const DataVector outputs1 = { outputs[0] }; // ignore 2nd output
    auto stage = model->addNewStage<ProposalStage>(layer->name, StageType::Proposal, layer, inputs, outputs1);

    stage->attrs().set<int>("feat_stride", layer->GetParamAsInt("feat_stride", 16));
    stage->attrs().set<int>("base_size", layer->GetParamAsInt("base_size", 16));
    stage->attrs().set<int>("min_size", layer->GetParamAsInt("min_size", 16));
    stage->attrs().set<int>("pre_nms_topn", layer->GetParamAsInt("pre_nms_topn", 6000));
    stage->attrs().set<int>("post_nms_topn", layer->GetParamAsInt("post_nms_topn", 300));
    stage->attrs().set<float>("nms_thresh", layer->GetParamAsFloat("nms_thresh", 0.7f));
    stage->attrs().set<float>("pre_nms_thresh", layer->GetParamAsFloat("pre_nms_thresh", 0.1f));
    stage->attrs().set<float>("box_size_scale", layer->GetParamAsFloat("box_size_scale", 1.0f));
    stage->attrs().set<float>("box_coordinate_scale", layer->GetParamAsFloat("box_coordinate_scale", 1.0f));
    stage->attrs().set<bool>("clip_before_nms", layer->GetParamAsBool("clip_before_nms", true));
    stage->attrs().set<bool>("clip_after_nms", layer->GetParamAsBool("clip_after_nms", false));
    stage->attrs().set<bool>("normalize", layer->GetParamAsBool("normalize", false));

    if (cmp(layer->GetParamAsString("framework", ""), "TensorFlow")) {
        // Settings for TensorFlow
        stage->attrs().set<float>("coordinates_offset", 0.0f);
        stage->attrs().set<bool>("initial_clip", true);
        stage->attrs().set<bool>("shift_anchors", true);
        stage->attrs().set<bool>("round_ratios", false);
        stage->attrs().set<bool>("swap_xy", true);
    } else {
        // Settings for Caffe

        stage->attrs().set<float>("coordinates_offset", 1.0f);
        stage->attrs().set<bool>("initial_clip", false);
        stage->attrs().set<bool>("shift_anchors", false);
        stage->attrs().set<bool>("round_ratios", true);
        stage->attrs().set<bool>("swap_xy", false);
    }

    auto scales = layer->GetParamAsFloats("scale", {});
    auto ratios = layer->GetParamAsFloats("ratio", {});

    stage->attrs().set("scales", scales);
    stage->attrs().set("ratios", ratios);

    int number_of_anchors = ratios.size() * scales.size();

    // Allocate slightly larger buffer than needed for handling remnant in distribution among SHAVEs
    int buffer_size = (inputs[0]->desc().dim(Dim::H) + 16) * inputs[0]->desc().dim(Dim::W) * number_of_anchors * 5 * sizeof(float);

    struct SortItem {
        int  index;
        float score;
    };
    const int num_proposals = number_of_anchors * inputs[0]->desc().dim(Dim::H) * inputs[0]->desc().dim(Dim::W);
    const int pre_nms_topn = std::min(num_proposals, stage->attrs().get<int>("pre_nms_topn"));
    const int required_cmx_size_per_shave = std::max(2 * (1 + pre_nms_topn) * sizeof(SortItem),
                                                     (1 + pre_nms_topn) * sizeof(SortItem) + number_of_anchors * sizeof(float));
    const auto& env = CompileEnv::get();
    const int required_cmx_buffer_size = env.resources.numSHAVEs * required_cmx_size_per_shave;

    model->addTempBuffer(stage, buffer_size + required_cmx_buffer_size);
}

}  // namespace vpu
