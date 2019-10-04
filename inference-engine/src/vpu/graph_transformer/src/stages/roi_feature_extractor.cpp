// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

#define MAX_PYRAMID_LEVELS 16

typedef SmallVector<int32_t, MAX_PYRAMID_LEVELS> PyramidLevelsVector;

class ROIFeatureExtractorStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<ROIFeatureExtractorStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto output = outputEdge(0)->output();

        auto levels_num = attrs().get<int>("levels_num");
        for (int i = 1; i < levels_num + 1; i++) {
            orderInfo.setInput(inputEdge(i), inputEdge(i)->input()->desc().dimsOrder().createMovedDim(Dim::C, 2));
        }
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 2));
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
        auto levels_num = attrs().get<int>("levels_num");
        IE_ASSERT(numInputs() == levels_num + 1);
        IE_ASSERT(numOutputs() == 1 || numOutputs() == 2);

        assertAllInputsOutputsTypes(this, DataType::FP16, DataType::FP16);
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto pooled_w = attrs().get<int>("pooled_w");
        auto pooled_h = attrs().get<int>("pooled_h");
        auto sampling_ratio = attrs().get<int>("sampling_ratio");
        auto levels_num = attrs().get<int>("levels_num");
        auto use_output_rois = attrs().get<int>("use_output_rois");
        auto pyramid_scales = attrs().get<PyramidLevelsVector>("pyramid_scales");

        serializer.append(static_cast<uint32_t>(pooled_w));
        serializer.append(static_cast<uint32_t>(pooled_h));
        serializer.append(static_cast<uint32_t>(sampling_ratio));
        serializer.append(static_cast<uint32_t>(levels_num));
        serializer.append(static_cast<uint32_t>(use_output_rois));

        for (int i = 0; i < pyramid_scales.size(); i++) {
            serializer.append(static_cast<int32_t>(pyramid_scales[i]));
        }
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto levels_num = attrs().get<int>("levels_num");

        IE_ASSERT(numInputs() == levels_num + 1);
        IE_ASSERT(numOutputs() == 1 || numOutputs() == 2);

        for (int i = 0; i < levels_num + 1; i++) {
            inputEdge(i)->input()->serializeNewBuffer(serializer);
        }

        for (auto i = 0; i < numOutputs(); i++) {
            auto output = outputEdge(i)->output();
            output->serializeNewBuffer(serializer);
        }

        tempBuffer(0)->serializeNewBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseROIFeatureExtractor(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() > 1);
    IE_ASSERT(outputs.size() == 1 || outputs.size() == 2);
    auto levels_num = inputs.size() - 1;

    auto stage = model->addNewStage<ROIFeatureExtractorStage>(
        layer->name,
        StageType::ROIFeatureExtractor,
        layer,
        inputs,
        outputs);

    auto output_dim_ = layer->GetParamAsInt("output_size");
    auto pyramid_scales_ = layer->GetParamAsInts("pyramid_scales");
    auto sampling_ratio_ = layer->GetParamAsInt("sampling_ratio");
    auto pooled_height_ = output_dim_;
    auto pooled_width_ = output_dim_;

    auto rois = inputs[0];
    auto num_rois = rois->desc().dim(Dim::N);
    auto channels_num = inputs[1]->desc().dim(Dim::C);

    stage->attrs().set<int>("levels_num", levels_num);
    stage->attrs().set<int>("pooled_w", pooled_width_);
    stage->attrs().set<int>("pooled_h", pooled_height_);
    stage->attrs().set<int>("sampling_ratio", sampling_ratio_);
    stage->attrs().set<int>("use_output_rois", outputs.size() == 2);

    IE_ASSERT(pyramid_scales_.size() <= MAX_PYRAMID_LEVELS);

    PyramidLevelsVector pyramidScales(MAX_PYRAMID_LEVELS, 1);
    for (int i = 0; i < pyramid_scales_.size(); i++) {
        pyramidScales[i] = pyramid_scales_[i];
    }
    stage->attrs().set<PyramidLevelsVector>("pyramid_scales", pyramidScales);

    const int feaxels_per_roi = pooled_height_ * pooled_width_ * channels_num;

    const int roi_height_max = 320; const int roi_width_max = 320;
    int roi_bin_grid_h = (sampling_ratio_ > 0) ? sampling_ratio_ : static_cast<int>(ceil(roi_height_max / pooled_height_));
    int roi_bin_grid_w = (sampling_ratio_ > 0) ? sampling_ratio_ : static_cast<int>(ceil(roi_width_max / pooled_width_));

    struct PreCalc {
      int pos1;
      int pos2;
      int pos3;
      int pos4;
      float w1;
      float w2;
      float w3;
      float w4;
    };

    int ALIGN_VALUE = 64;
    int size_levels_id_buf = sizeof(int) * num_rois + ALIGN_VALUE;
    int size_reordered_rois_buf = sizeof(int16_t) * 4 * num_rois + ALIGN_VALUE;
    int size_original_rois_mapping_buf = sizeof(int) * num_rois + ALIGN_VALUE;
    int size_output_rois_features_temp_buf = sizeof(int16_t) * feaxels_per_roi * num_rois + ALIGN_VALUE;
    int size_rois_per_level_buf = (levels_num + 1) * sizeof(int) + ALIGN_VALUE;
    int size_dummy_mapping_buf = sizeof(int) * num_rois + ALIGN_VALUE;
    int size_pre_calc_buf = sizeof(PreCalc) * roi_bin_grid_h * roi_bin_grid_w * pooled_width_ * pooled_height_ + ALIGN_VALUE;

    int buffer_size = size_levels_id_buf +
                      size_reordered_rois_buf +
                      size_original_rois_mapping_buf +
                      size_output_rois_features_temp_buf +
                      size_rois_per_level_buf +
                      size_dummy_mapping_buf +
                      size_pre_calc_buf;

    model->addTempBuffer(
        stage,
        DataDesc({buffer_size}));
}

}  // namespace vpu
