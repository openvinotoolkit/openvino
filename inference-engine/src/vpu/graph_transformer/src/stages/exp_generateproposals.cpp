// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>

namespace vpu {

namespace {

VPU_PACKED(ExpGenerateProposalsParams {
    float   min_size;
    float   nms_threshold;
    int32_t pre_nms_topn;
    int32_t post_nms_topn;
};)

class ExpGenerateProposalsStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ExpGenerateProposalsStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        for (const auto& inEdge : inputEdges()) {
            stridesInfo.setInput(inEdge, StridesRequirement::compact());
        }
        for (const auto& outEdge : outputEdges()) {
            stridesInfo.setOutput(outEdge, StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
             {{DataType::FP16}, {DataType::FP16}, {DataType::FP16}, {DataType::FP16}},
             {{DataType::FP16}, {DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        const auto& params = attrs().get<ExpGenerateProposalsParams>("params");

        serializer.append(params);
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        for (auto& inputEdge : inputEdges()) {
            inputEdge->input()->serializeBuffer(serializer);
        }

        for (auto& outputEdge : outputEdges()) {
            outputEdge->output()->serializeBuffer(serializer);
        }

        tempBuffer(0)->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseExpGenerateProposals(
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs) const {
    auto expGenerateProposals = ngraph::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node);
    VPU_THROW_UNLESS(inputs.size() == 4, "Layer %s must have 4 input tensors.", expGenerateProposals->get_name());
    VPU_THROW_UNLESS(outputs.size() == 2, "Layer %s must have 2 output tensors.", expGenerateProposals->get_name());

    ExpGenerateProposalsParams params;
    const auto attrs = expGenerateProposals->get_attrs();
    params.min_size      = attrs.min_size;
    params.nms_threshold = attrs.nms_threshold;
    params.pre_nms_topn  = attrs.pre_nms_count;
    params.post_nms_topn = attrs.post_nms_count;

    auto imInfo       = inputs[0];
    auto inputAnchors = inputs[1];
    auto inputDeltas  = inputs[2];
    auto inputScores  = inputs[3];
    auto outputRois   = outputs[0];
    auto outputScores = outputs[1];

    VPU_THROW_UNLESS((inputAnchors->desc().dims().size() == 2) &&
                     (inputAnchors->desc().dim(Dim::C) == 4),
                     "Wrong shape for input 1 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     expGenerateProposals->get_name(), inputAnchors->desc().dims().size(), inputAnchors->desc().dim(Dim::C));
    VPU_THROW_UNLESS((imInfo->desc().dims().size() == 1) &&
                     (imInfo->desc().dim(Dim::C) == 3),
                     "Wrong shape for input 0 of layer %s, expected (3), got: dims size = %lu, dim C = %d",
                     expGenerateProposals->get_name(), imInfo->desc().dims().size(), imInfo->desc().dim(Dim::C));

    VPU_THROW_UNLESS(inputDeltas->desc().dims().size() == 3,
                     "Wrong shape for input 2 of layer %s, expected dim size = 3, got: %lu",
                     expGenerateProposals->get_name(), inputDeltas->desc().dims().size());
    VPU_THROW_UNLESS(inputScores->desc().dims().size() == 3,
                     "Wrong shape for input 3 of layer %s, expected dim size = 3, got: %lu",
                     expGenerateProposals->get_name(), inputScores->desc().dims().size());

    VPU_THROW_UNLESS((inputDeltas->desc().dim(Dim::H) == inputScores->desc().dim(Dim::H)) &&
                     (inputDeltas->desc().dim(Dim::W) == inputScores->desc().dim(Dim::W)),
                     "Inputs 2 and 3 of layer %s must have same H and W, got: input2 (H = %d, W = %d), input3 (H = %d, W = %d)",
                     expGenerateProposals->get_name(), inputDeltas->desc().dim(Dim::H), inputDeltas->desc().dim(Dim::W),
                     inputScores->desc().dim(Dim::H), inputScores->desc().dim(Dim::W));

    VPU_THROW_UNLESS((outputRois->desc().dims().size() == 2) &&
                     (outputRois->desc().dim(Dim::C) == 4),
                     "Wrong shape for output 0 of layer %s, expected (N, 4), got: dims size = %lu, dim C = %d",
                     expGenerateProposals->get_name(), outputRois->desc().dims().size(), outputRois->desc().dim(Dim::C));
    VPU_THROW_UNLESS(outputScores->desc().dims().size() == 1,
                     "Wrong shape for output 1 of layer %s, expected dim size = 1, got: %lu",
                     expGenerateProposals->get_name(), outputScores->desc().dims().size());

    VPU_THROW_UNLESS(outputRois->desc().dim(Dim::N) == outputScores->desc().dim(Dim::C),
                     "Layer %s: output0 dim N and output1 dim C must be equal, got: output0 (N = %d), output1 (C = %d)",
                     expGenerateProposals->get_name(), outputRois->desc().dim(Dim::N), outputScores->desc().dim(Dim::C));

    auto stage = model->addNewStage<ExpGenerateProposalsStage>(
        expGenerateProposals->get_name(),
        StageType::ExpGenerateProposals,
        expGenerateProposals,
        inputs,
        outputs);

    stage->attrs().set("params", params);

    //This structure is needed to compute sizeProposalBuf.
    //Since its outside the scope of the file, we write structure here
    typedef struct {
        int32_t idx;
        fp16_t x0;
        fp16_t y0;
        fp16_t x1;
        fp16_t y1;
        fp16_t score;
    } t_ExpGenerateProposalsProposal;

    const int ALIGN_VALUE = 64;
    const int sizeProposalsBuf = sizeof(t_ExpGenerateProposalsProposal) *
                             inputScores->desc().dim(Dim::H) *
                             inputScores->desc().dim(Dim::W) *
                             inputScores->desc().dim(Dim::C) + ALIGN_VALUE;
    const int sizeAuxBuf = sizeof(int8_t) * params.pre_nms_topn + ALIGN_VALUE;
    const int sizeRoiIndicesBuf = sizeof(int32_t) * params.post_nms_topn + ALIGN_VALUE;

    int buffer_size = 2 * sizeProposalsBuf + sizeAuxBuf + sizeRoiIndicesBuf;

    model->addTempBuffer(stage, buffer_size);
}

}  // namespace vpu
