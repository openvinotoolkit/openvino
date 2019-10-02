// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <algorithm>
#include <vector>
#include <memory>

#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

namespace {

class PriorBoxClusteredContent final : public CalculatedDataContent {
public:
    PriorBoxClusteredContent(
            const DataDesc& inDesc0,
            const DataDesc& inDesc1,
            const DataDesc& outDesc,
            const ie::CNNLayerPtr& layer) :
            _inDesc0(inDesc0), _inDesc1(inDesc1), _outDesc(outDesc),
            _layer(layer) {
        IE_ASSERT(layer != nullptr);
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>&, void* tempBuf) const override {
        VPU_PROFILE(PriorBoxClusteredContent);

        auto tempPtr = static_cast<fp16_t*>(tempBuf);

        auto widths_ = _layer->GetParamAsFloats("width");
        auto heights_ = _layer->GetParamAsFloats("height");
        auto clip_ = _layer->GetParamAsInt("clip");
        auto variance_ = _layer->GetParamAsFloats("variance");
        auto img_h_ = _layer->GetParamAsInt("img_h", 0);
        auto img_w_ = _layer->GetParamAsInt("img_w", 0);
        auto step_ = _layer->GetParamAsFloat("step", 0);
        auto step_h_ = _layer->GetParamAsFloat("step_h", 0);
        auto step_w_ = _layer->GetParamAsFloat("step_w", 0);
        auto offset_ = _layer->GetParamAsFloat("offset", 0);

        auto num_priors_ = widths_.size();

        if (variance_.empty()) {
            variance_.push_back(0.1);
        }

        auto layer_width  = _inDesc0.dim(Dim::W);
        auto layer_height = _inDesc0.dim(Dim::H);

        auto img_width  = img_w_ == 0 ? _inDesc1.dim(Dim::W) : img_w_;
        auto img_height = img_h_ == 0 ? _inDesc1.dim(Dim::H) : img_h_;

        auto step_w = step_w_ == 0 ? step_ : step_w_;
        auto step_h = step_h_ == 0 ? step_ : step_h_;
        if (step_w == 0 || step_h == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        }

        auto expetected_output_dimx = layer_height * layer_width * num_priors_ * 4;
        if (_outDesc.dim(Dim::W) != expetected_output_dimx || _outDesc.dim(Dim::H) != 2) {
            THROW_IE_EXCEPTION << "PriorBoxClustered output has invalid dimension, exptected " << expetected_output_dimx << "x2"
                               << ", got " << _outDesc.dim(Dim::W) << "x" << _outDesc.dim(Dim::H) << ", layer name is: " << _layer->name;
        }

        auto offset = _outDesc.dim(Dim::W);
        auto var_size = variance_.size();

        auto top_data_0 = tempPtr;
        auto top_data_1 = top_data_0 + offset;

        ie::parallel_for2d(layer_height, layer_width, [=](int h, int w) {
            auto center_x = (w + offset_) * step_w;
            auto center_y = (h + offset_) * step_h;

            for (int s = 0; s < num_priors_; ++s) {
                auto box_width  = widths_[s];
                auto box_height = heights_[s];

                auto xmin = (center_x - box_width  / 2.0f) / img_width;
                auto ymin = (center_y - box_height / 2.0f) / img_height;
                auto xmax = (center_x + box_width  / 2.0f) / img_width;
                auto ymax = (center_y + box_height / 2.0f) / img_height;

                if (clip_) {
                    xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                    ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                    xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                    ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                }

                top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 0] = ie::PrecisionUtils::f32tof16(xmin);
                top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 1] = ie::PrecisionUtils::f32tof16(ymin);
                top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 2] = ie::PrecisionUtils::f32tof16(xmax);
                top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 3] = ie::PrecisionUtils::f32tof16(ymax);

                for (int j = 0; j < var_size; j++) {
                    auto index = h * layer_width * num_priors_ * var_size + w * num_priors_ * var_size + s * var_size + j;
                    top_data_1[index] = ie::PrecisionUtils::f32tof16(variance_[j]);
                }
            }
        });
    }

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    ie::CNNLayerPtr _layer;
};

}  // namespace

void FrontEnd::parsePriorBoxClustered(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs) {
    IE_ASSERT(inputs.size() == 2);
    IE_ASSERT(outputs.size() == 1);

    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];

    auto resultData = model->addConstData(
        output->name(),
        output->desc(),
        std::make_shared<PriorBoxClusteredContent>(input0->desc(), input1->desc(), output->desc(), layer));

    if (output->usage() == DataUsage::Output || output->numConsumers() > 0) {
        _stageBuilder->addCopyStage(model, layer->name, layer, resultData, output, "parsePriorBoxClustered");
    } else {
        IE_ASSERT(output->usage() == DataUsage::Intermediate);

        bindData(resultData, output->origData());
    }
}

}  // namespace vpu
