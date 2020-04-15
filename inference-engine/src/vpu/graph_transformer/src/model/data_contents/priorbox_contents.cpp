// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/priorbox_contents.hpp>

#include <vpu/utils/profiling.hpp>

#include <precision_utils.h>
#include <ie_layers.h>
#include <ie_parallel.hpp>

namespace vpu {

//
// PriorBoxContent
//

PriorBoxContent::PriorBoxContent(
        const DataDesc& inDesc0,
        const DataDesc& inDesc1,
        const DataDesc& outDesc,
        const ie::CNNLayerPtr &layer) :
        _inDesc0(inDesc0), _inDesc1(inDesc1), _outDesc(outDesc),
        _layer(layer) {
    IE_ASSERT(layer != nullptr);
}

size_t PriorBoxContent::byteSize() const {
    return checked_cast<size_t>(_outDesc.totalDimSize()) *
           checked_cast<size_t>(_outDesc.elemSize());
}

void PriorBoxContent::fillTempBuf(void* tempBuf) const {
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

    auto _fixed_sizes = _layer->GetParamAsFloats("fixed_size", {});
    auto _fixed_ratios = _layer->GetParamAsFloats("fixed_ratio", {});
    auto _densitys = _layer->GetParamAsFloats("density", {});

    SmallVector<float> _aspect_ratios;
    _aspect_ratios.reserve(aspect_ratios.size() + 1);

    _aspect_ratios.push_back(1.0f);
    for (const auto& aspect_ratio : aspect_ratios) {
        bool exist = false;

        for (const auto& _aspect_ratio : _aspect_ratios) {
            if (fabsf(aspect_ratio - _aspect_ratio) < 1e-6) {
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

    if (!_fixed_sizes.empty()) {
        _num_priors = static_cast<int>(_aspect_ratios.size() * _fixed_sizes.size());
    }

    if (!_densitys.empty()) {
        for (const auto& _density : _densitys) {
            if (!_fixed_ratios.empty()) {
                _num_priors += _fixed_ratios.size() * (static_cast<int>(pow(_density, 2)) - 1);
            } else {
                _num_priors += _aspect_ratios.size() * (static_cast<int>(pow(_density, 2)) - 1);
            }
        }
    }

    _num_priors += _max_sizes.size();

    auto W  = _inDesc0.dim(Dim::W);
    auto H  = _inDesc0.dim(Dim::H);
    auto IW = _img_w == 0 ? _inDesc1.dim(Dim::W) : _img_w;
    auto IH = _img_h == 0 ? _inDesc1.dim(Dim::H) : _img_h;
    auto IWI = 1.0f / static_cast<float>(IW);
    auto IHI = 1.0f / static_cast<float>(IH);

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

    float box_width = 0.0f;
    float box_height = 0.0f;

    if (_outDesc.dim(Dim::W) != dim || _outDesc.dim(Dim::H) != 2) {
        THROW_IE_EXCEPTION << "[VPU] PriorBox output have invalid dimension, exptected " << dim << "x2"
                           << ", got " << _outDesc.dim(Dim::W) << "x" << _outDesc.dim(Dim::H)
                           << ", layer name is: " << _layer->name;
    }

    auto max_fp16 = [](const float value, const float min) {
        return ie::PrecisionUtils::f32tof16(value > min ? value : min);
    };

    auto min_fp16 = [](const float value, const float max) {
        return ie::PrecisionUtils::f32tof16(value < max ? value : max);
    };

    size_t idx = 0;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W;  ++w) {
            if (_step == 0) {
                center_x = (static_cast<float>(w) + 0.5f) * step_x;
                center_y = (static_cast<float>(h) + 0.5f) * step_y;
            } else {
                center_x = (_offset + static_cast<float>(w)) * _step;
                center_y = (_offset + static_cast<float>(h)) * _step;
            }

            for (size_t s = 0; s < _fixed_sizes.size(); ++s) {
                auto fixed_size_ = static_cast<size_t>(_fixed_sizes[s]);
                box_width = box_height = fixed_size_ * 0.5f;

                int density_ = 0;
                int shift = 0;
                if (s < _densitys.size()) {
                    density_ = static_cast<size_t>(_densitys[s]);
                    shift = static_cast<int>(_fixed_sizes[s] / density_);
                }

                if (!_fixed_ratios.empty()) {
                    for (const auto& fr : _fixed_ratios) {
                        const auto box_width_ratio = _fixed_sizes[s] * 0.5f * std::sqrt(fr);
                        const auto box_height_ratio = _fixed_sizes[s] * 0.5f / std::sqrt(fr);

                        for (size_t r = 0; r < density_; ++r) {
                            for (size_t c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                dst_data[idx++] = max_fp16((center_x_temp - box_width_ratio) * IWI, 0.f);
                                dst_data[idx++] = max_fp16((center_y_temp - box_height_ratio) * IHI, 0.f);
                                dst_data[idx++] = min_fp16((center_x_temp + box_width_ratio) * IWI, 1.f);
                                dst_data[idx++] = min_fp16((center_y_temp + box_height_ratio) * IHI, 1.f);
                            }
                        }
                    }
                } else {
                    if (!_densitys.empty()) {
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                dst_data[idx++] = max_fp16((center_x_temp - box_width) * IWI, 0.f);
                                dst_data[idx++] = max_fp16((center_y_temp - box_height) * IHI, 0.f);
                                dst_data[idx++] = min_fp16((center_x_temp + box_width) * IWI, 1.f);
                                dst_data[idx++] = min_fp16((center_y_temp + box_height) * IHI, 1.f);
                            }
                        }
                    }
                    //  Rest of priors
                    for (const auto& ar : _aspect_ratios) {
                        if (fabs(ar - 1.) < 1e-6) {
                            continue;
                        }

                        const auto box_width_ratio = _fixed_sizes[s] * 0.5f * std::sqrt(ar);
                        const auto box_height_ratio = _fixed_sizes[s] * 0.5f / std::sqrt(ar);
                        for (int r = 0; r < density_; ++r) {
                            for (int c = 0; c < density_; ++c) {
                                const auto center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                const auto center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                dst_data[idx++] = max_fp16((center_x_temp - box_width_ratio) * IWI, 0.f);
                                dst_data[idx++] = max_fp16((center_y_temp - box_height_ratio) * IHI, 0.f);
                                dst_data[idx++] = min_fp16((center_x_temp + box_width_ratio) * IWI, 1.f);
                                dst_data[idx++] = min_fp16((center_y_temp + box_height_ratio) * IHI, 1.f);
                            }
                        }
                    }
                }
            }

            for (size_t msIdx = 0; msIdx < _min_sizes.size(); msIdx++) {
                box_width = _min_sizes[msIdx];
                box_height = _min_sizes[msIdx];

                dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x - box_width / 2.0f) / IW);
                dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y - box_height / 2.0f) / IH);
                dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x + box_width / 2.0f) / IW);
                dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y + box_height / 2.0f) / IH);

                if (_max_sizes.size() > msIdx) {
                    box_width = box_height = std::sqrt(_min_sizes[msIdx] * _max_sizes[msIdx]);

                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x - box_width / 2.0f) / IW);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y - box_height / 2.0f) / IH);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_x + box_width / 2.0f) / IW);
                    dst_data[idx++] = ie::PrecisionUtils::f32tof16((center_y + box_height / 2.0f) / IH);
                }

                if (_scale_all_sizes || (!_scale_all_sizes && (msIdx == _min_sizes.size() - 1))) {
                    size_t sIdx = _scale_all_sizes ? msIdx : 0;
                    for (const auto& ar : _aspect_ratios) {
                        if (std::fabs(ar - 1.0f) < 1e-6) {
                            continue;
                        }

                        box_width = _min_sizes[sIdx] * std::sqrt(ar);
                        box_height = _min_sizes[sIdx] / std::sqrt(ar);

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

//
// PriorBoxClusteredContent
//

PriorBoxClusteredContent::PriorBoxClusteredContent(
        const DataDesc& inDesc0,
        const DataDesc& inDesc1,
        const DataDesc& outDesc,
        const ie::CNNLayerPtr& layer) :
        _inDesc0(inDesc0), _inDesc1(inDesc1), _outDesc(outDesc),
        _layer(layer) {
    IE_ASSERT(layer != nullptr);
}

size_t PriorBoxClusteredContent::byteSize() const {
    return checked_cast<size_t>(_outDesc.totalDimSize()) *
           checked_cast<size_t>(_outDesc.elemSize());
}

void PriorBoxClusteredContent::fillTempBuf(void* tempBuf) const {
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

} // namespace vpu
