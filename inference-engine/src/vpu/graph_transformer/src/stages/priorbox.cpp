// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cmath>

#include <algorithm>
#include <vector>
#include <memory>

#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/utils/numeric.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

namespace {

class PriorBoxContent final : public CalculatedDataContent {
public:
    PriorBoxContent(
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
        VPU_PROFILE(PriorBoxContent);

        auto tempPtr = static_cast<fp16_t*>(tempBuf);

        auto _min_sizes = _layer->GetParamAsFloats("min_size", {});
        auto _max_sizes = _layer->GetParamAsFloats("max_size", {});
        auto aspect_ratios = _layer->GetParamAsFloats("aspect_ratio");
        auto _flip = static_cast<bool>(_layer->GetParamAsInt("flip"));
        auto _clip = static_cast<bool>(_layer->GetParamAsInt("clip"));
        auto _variance = _layer->GetParamAsFloats("variance");
        auto _img_h = _layer->GetParamAsInt("img_h", 0);
        auto _img_w = _layer->GetParamAsInt("img_w", 0);
        auto _step = _layer->GetParamAsFloat("step", 0);
        auto _offset = _layer->GetParamAsFloat("offset", 0);
        auto _scale_all_sizes = static_cast<bool>(_layer->GetParamAsInt("scale_all_sizes", 1));

        SmallVector<float> _aspect_ratios;
        _aspect_ratios.reserve(aspect_ratios.size() + 1);

        _aspect_ratios.push_back(1.0f);
        for (auto aspect_ratio : aspect_ratios) {
            bool exist = false;

            for (float _aspect_ratio : _aspect_ratios) {
                if (fabs(aspect_ratio - _aspect_ratio) < 1e-6) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                _aspect_ratios.push_back(aspect_ratio);
                if (_flip) {
                    if (isFloatEqual(aspect_ratio, 0.f)) {
                        THROW_IE_EXCEPTION << "[VPU] PriorBox has 0.0 aspect ratio param in flip mode, "
                                           << " possible division by zero";
                    }
                    _aspect_ratios.push_back(1.0f / aspect_ratio);
                }
            }
        }

        int _num_priors;
        if (_scale_all_sizes) {
            _num_priors = static_cast<int>(_aspect_ratios.size() * _min_sizes.size());
        } else {
            _num_priors = static_cast<int>(_aspect_ratios.size() + _min_sizes.size() - 1);
        }

        for (auto it = _max_sizes.begin(); it != _max_sizes.end(); it++) {
            _num_priors += 1;
        }

        auto W  = _inDesc0.dim(Dim::W);
        auto H  = _inDesc0.dim(Dim::H);
        auto IW = _img_w == 0 ? _inDesc1.dim(Dim::W) : _img_w;
        auto IH = _img_h == 0 ? _inDesc1.dim(Dim::H) : _img_h;

        auto OW = (_outDesc.numDims() >= 4) ? _outDesc.dim(Dim::N) : 1;
        auto OH = _outDesc.dim(Dim::W);

        float step_x = 0.0f;
        float step_y = 0.0f;

        if (_step == 0) {
            step_x = static_cast<float>(IW) / W;
            step_y = static_cast<float>(IH) / H;
        } else {
            step_x = _step;
            step_y = _step;
        }

        auto dst_data = tempPtr;

        int dim = H * W * _num_priors * 4;
        float center_x = 0.0f;
        float center_y = 0.0f;

        float box_width;
        float box_height;

        if (_outDesc.dim(Dim::W) != dim || _outDesc.dim(Dim::H) != 2) {
            THROW_IE_EXCEPTION << "[VPU] PriorBox output have invalid dimension, exptected " << dim << "x2"
                               << ", got " << _outDesc.dim(Dim::W) << "x" << _outDesc.dim(Dim::H) << ", layer name is: " << _layer->name;
        }

        int idx = 0;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W;  ++w) {
                for (size_t msIdx = 0; msIdx < _min_sizes.size(); msIdx++) {
                    if (_step == 0) {
                        center_x = (w + 0.5f) * step_x;
                        center_y = (h + 0.5f) * step_y;
                    } else {
                        center_x = (_offset + w) * _step;
                        center_y = (_offset + h) * _step;
                    }

                    box_width = _min_sizes[msIdx];
                    box_height = _min_sizes[msIdx];

                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x - box_width / 2.0f) / IW);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y - box_height / 2.0f) / IH);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x + box_width / 2.0f) / IW);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y + box_height / 2.0f) / IH);

                    if (_max_sizes.size() > msIdx) {
                        box_width = box_height = sqrt(_min_sizes[msIdx] * _max_sizes[msIdx]);

                        dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x - box_width / 2.0f) / IW);
                        dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y - box_height / 2.0f) / IH);
                        dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x + box_width / 2.0f) / IW);
                        dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y + box_height / 2.0f) / IH);
                    }

                    if (_scale_all_sizes || (!_scale_all_sizes && (msIdx == _min_sizes.size() - 1))) {
                        size_t sIdx = _scale_all_sizes ? msIdx : 0;
                        for (float ar : _aspect_ratios) {
                            if (fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            box_width = _min_sizes[sIdx] * sqrt(ar);
                            box_height = _min_sizes[sIdx] / sqrt(ar);

                            dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x - box_width / 2.0f) / IW);
                            dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y - box_height / 2.0f) / IH);
                            dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x + box_width / 2.0f) / IW);
                            dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y + box_height / 2.0f) / IH);
                        }
                    }
                }
            }
        }

        if (_clip) {
            for (int d = 0; d < dim; ++d) {
                dst_data[d] = (std::min)((std::max)(dst_data[d], ie::PrecisionUtils::f32tof16(0.0f)), ie::PrecisionUtils::f32tof16(1.0f));
            }
        }

        int channel_size = OH * OW;

        dst_data += channel_size;

        if (_variance.size() == 1) {
            ie::parallel_for(channel_size, [&](int i) {
                dst_data[i] = ie::PrecisionUtils::f32tof16(_variance[0]);
            });
        } else {
            ie::parallel_for4d(H, W, _num_priors, 4, [&](int h, int w, int i, int j) {
                dst_data[j + 4 * (i + _num_priors * (w + W * h))] = ie::PrecisionUtils::f32tof16(_variance[j]);
            });
        }
    }

private:
    DataDesc _inDesc0;
    DataDesc _inDesc1;
    DataDesc _outDesc;
    ie::CNNLayerPtr _layer;
};

}  // namespace

void FrontEnd::parsePriorBox(
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
        std::make_shared<PriorBoxContent>(input0->desc(), input1->desc(), output->desc(), layer));

    if (output->usage() == DataUsage::Output || output->numConsumers() > 0) {
        _stageBuilder->addCopyStage(model, layer->name, layer, resultData, output);
    } else {
        IE_ASSERT(output->usage() == DataUsage::Intermediate);

        bindData(resultData, output->origData());
    }
}

}  // namespace vpu
