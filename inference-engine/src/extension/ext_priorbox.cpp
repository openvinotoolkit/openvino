// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PriorBoxImpl: public ExtLayerBase {
    static inline float clip_great(float x, float threshold) {
        return x < threshold ? x : threshold;
    }

    static inline float clip_less(float x, float threshold) {
        return x > threshold ? x : threshold;
    }

public:
    explicit PriorBoxImpl(const CNNLayer *layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 ||
                    layer->insData[1].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "PriorBox supports only 4D blobs!";

            _offset = layer->GetParamAsFloat("offset");
            _step = layer->GetParamAsFloat("step", 0);
            _min_sizes = layer->GetParamAsFloats("min_size", {});
            _max_sizes = layer->GetParamAsFloats("max_size", {});
            _flip = layer->GetParamAsBool("flip", false);
            _clip = layer->GetParamAsBool("clip", false);
            _scale_all_sizes = layer->GetParamAsBool("scale_all_sizes", true);

           _fixed_sizes = layer->GetParamAsFloats("fixed_size", {});
           _fixed_ratios = layer->GetParamAsFloats("fixed_ratio", {});
           _densitys = layer->GetParamAsFloats("density", {});

            bool exist;

            _aspect_ratios.push_back(1.0f);

            const std::vector<float> aspect_ratios = layer->GetParamAsFloats("aspect_ratio", {});

            for (float aspect_ratio : aspect_ratios) {
                exist = false;

                if (std::fabs(aspect_ratio) < std::numeric_limits<float>::epsilon()) {
                    THROW_IE_EXCEPTION << "aspect_ratio param can't be equal to zero";
                }

                for (float _aspect_ratio : _aspect_ratios) {
                    if (fabs(aspect_ratio - _aspect_ratio) < 1e-6) {
                        exist = true;
                        break;
                    }
                }

                if (exist) {
                    continue;
                }

                _aspect_ratios.push_back(aspect_ratio);

                if (_flip) {
                    _aspect_ratios.push_back(1.0f / aspect_ratio);
                }
            }

            if (_scale_all_sizes) {
                _num_priors = static_cast<int>(_aspect_ratios.size() * _min_sizes.size());
            } else {
                _num_priors = static_cast<int>(_aspect_ratios.size() + _min_sizes.size() - 1);
            }

            if (_fixed_sizes.size() > 0) {
                _num_priors = static_cast<int>(_aspect_ratios.size() * _fixed_sizes.size());
            }

            if (_densitys.size() > 0) {
                for (size_t i = 0; i < _densitys.size(); ++i) {
                    if (_fixed_ratios.size() > 0) {
                       _num_priors += (_fixed_ratios.size()) * (static_cast<size_t>(pow(_densitys[i], 2)) - 1);
                    } else {
                        _num_priors += (_aspect_ratios.size()) * (static_cast<size_t>(pow(_densitys[i], 2)) - 1);
                    }
                }
            }

            for (auto it = _max_sizes.begin(); it != _max_sizes.end(); it++) {
                _num_priors += 1;
            }

            const std::vector<float> variance = layer->GetParamAsFloats("variance", {});

            if (variance.size() == 1 || variance.size() == 4) {
                for (float i : variance) {
                    if (i < 0) {
                        THROW_IE_EXCEPTION << "Variance must be > 0.";
                    }

                    _variance.push_back(i);
                }
            } else if (variance.empty()) {
                _variance.push_back(0.1f);
            } else {
                THROW_IE_EXCEPTION << "Wrong number of variance values. Not less than 1 and more than 4 variance values.";
            }

            addConfig(layer, {{ConfLayout::ANY, true}, {ConfLayout::ANY, true}}, {{ConfLayout::PLN, true}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        return OK;
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if (inputs.size() != 2 || outputs.empty()) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        auto& dataMemPtr = inputs[0];
        auto& imageMemPtr = inputs[1];
        auto& dstMemPtr = outputs[0];
        SizeVector _data_dims = dataMemPtr->getTensorDesc().getDims();
        SizeVector _image_dims = imageMemPtr->getTensorDesc().getDims();
        const int W = _data_dims[3];
        const int H = _data_dims[2];
        const int IW = _image_dims[3];
        const int IH = _image_dims[2];

        const int OH = dstMemPtr->getTensorDesc().getDims()[2];
        const int OW = (dstMemPtr->getTensorDesc().getDims().size() == 3) ? 1 : dstMemPtr->getTensorDesc().getDims()[3];

        float step_x = 0.0f;
        float step_y = 0.0f;

        if (_step == 0) {
            step_x = static_cast<float>(IW) / W;
            step_y = static_cast<float>(IH) / H;
        } else {
            step_x = _step;
            step_y = _step;
        }

        float IWI = 1.0f / static_cast<float>(IW);
        float IHI = 1.0f / static_cast<float>(IH);

        float* dst_data = dstMemPtr->buffer();

        int idx = 0;
        float center_x = 0.0f;
        float center_y = 0.0f;

        float box_width;
        float box_height;

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (_step == 0) {
                    center_x = (w + 0.5f) * step_x;
                    center_y = (h + 0.5f) * step_y;
                } else {
                    center_x = (_offset + w) * _step;
                    center_y = (_offset + h) * _step;
                }

                for (size_t s = 0; s < _fixed_sizes.size(); ++s) {
                    size_t fixed_size_ = static_cast<size_t>(_fixed_sizes[s]);
                    box_width = box_height = fixed_size_ * 0.5f;

                    if (_fixed_ratios.size() > 0) {
                        for (float ar : _fixed_ratios) {
                            size_t density_ = static_cast<size_t>(_densitys[s]);
                            int shift = static_cast<int>(_fixed_sizes[s] / density_);
                            ar = sqrt(ar);
                            float box_width_ratio = _fixed_sizes[s] * 0.5f * ar;
                            float box_height_ratio = _fixed_sizes[s] * 0.5f / ar;
                            for (size_t r = 0; r < density_; ++r) {
                                for (size_t c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                    // xmin
                                    dst_data[idx++] = clip_less((center_x_temp - box_width_ratio) * IWI, 0);
                                    // ymin
                                    dst_data[idx++] = clip_less((center_y_temp - box_height_ratio) * IHI, 0);
                                    // xmax
                                    dst_data[idx++] = clip_great((center_x_temp + box_width_ratio) * IWI, 1);
                                    // ymax
                                    dst_data[idx++] = clip_great((center_y_temp + box_height_ratio) * IHI, 1);
                                }
                            }
                        }
                    } else {
                        if (_densitys.size() > 0) {
                            int density_ = static_cast<int>(_densitys[s]);
                            int shift = static_cast<int>(_fixed_sizes[s] / density_);
                            for (int r = 0; r < density_; ++r) {
                                for (int c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;

                                    // xmin
                                    dst_data[idx++] = clip_less((center_x_temp - box_width) * IWI, 0);
                                    // ymin
                                    dst_data[idx++] = clip_less((center_y_temp - box_height) * IHI, 0);
                                    // xmax
                                    dst_data[idx++] = clip_great((center_x_temp + box_width) * IWI, 1);
                                    // ymax
                                    dst_data[idx++] = clip_great((center_y_temp + box_height) * IHI, 1);
                                }
                            }
                        }
                        //  Rest of priors
                        for (float ar : _aspect_ratios) {
                            if (fabs(ar - 1.) < 1e-6) {
                                continue;
                            }

                            int density_ = static_cast<int>(_densitys[s]);
                            int shift = static_cast<int>(_fixed_sizes[s] / density_);
                            ar = sqrt(ar);
                            float box_width_ratio = _fixed_sizes[s] * 0.5f * ar;
                            float box_height_ratio = _fixed_sizes[s] * 0.5f / ar;
                            for (int r = 0; r < density_; ++r) {
                                for (int c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                    // xmin
                                    dst_data[idx++] = clip_less((center_x_temp - box_width_ratio) * IWI, 0);
                                    // ymin
                                    dst_data[idx++] = clip_less((center_y_temp - box_height_ratio) * IHI, 0);
                                    // xmax
                                    dst_data[idx++] = clip_great((center_x_temp + box_width_ratio) * IWI, 1);
                                    // ymax
                                    dst_data[idx++] = clip_great((center_y_temp + box_height_ratio) * IHI, 1);
                                }
                            }
                        }
                    }
                }

                for (size_t msIdx = 0; msIdx < _min_sizes.size(); msIdx++) {
                    box_width = _min_sizes[msIdx] * 0.5f;
                    box_height = _min_sizes[msIdx] * 0.5f;

                    dst_data[idx++] = (center_x - box_width) * IWI;
                    dst_data[idx++] = (center_y - box_height) * IHI;
                    dst_data[idx++] = (center_x + box_width) * IWI;
                    dst_data[idx++] = (center_y + box_height) * IHI;

                    if (_max_sizes.size() > msIdx) {
                        box_width = box_height = sqrt(_min_sizes[msIdx] * _max_sizes[msIdx]) * 0.5f;

                        dst_data[idx++] = (center_x - box_width) * IWI;
                        dst_data[idx++] = (center_y - box_height) * IHI;
                        dst_data[idx++] = (center_x + box_width) * IWI;
                        dst_data[idx++] = (center_y + box_height) * IHI;
                    }

                    if (_scale_all_sizes || (!_scale_all_sizes && (msIdx == _min_sizes.size() - 1))) {
                        size_t sIdx = _scale_all_sizes ? msIdx : 0;
                        for (float ar : _aspect_ratios) {
                            if (fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            ar = sqrt(ar);
                            box_width = _min_sizes[sIdx] * 0.5f * ar;
                            box_height = _min_sizes[sIdx] * 0.5f / ar;

                            dst_data[idx++] = (center_x - box_width) * IWI;
                            dst_data[idx++] = (center_y - box_height) * IHI;
                            dst_data[idx++] = (center_x + box_width) * IWI;
                            dst_data[idx++] = (center_y + box_height) * IHI;
                        }
                    }
                }
            }
        }

        if (_clip) {
            parallel_for((H * W * _num_priors * 4), [&](size_t i) {
                dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0f), 1.0f);
            });
        }

        size_t channel_size = OH * OW;
        dst_data += channel_size;
        if (_variance.size() == 1) {
            parallel_for(channel_size, [&](size_t i) {
                dst_data[i] = _variance[0];
            });
        } else {
            parallel_for((H * W * _num_priors), [&](size_t i) {
                for (size_t j = 0; j < 4; ++j) {
                    dst_data[i * 4 + j] = _variance[j];
                }
            });
        }
        return OK;
    }

private:
    float _offset = 0;
    float _step = 0;
    std::vector<float> _min_sizes;
    std::vector<float> _max_sizes;
    bool _flip = false;
    bool _clip = false;
    bool _scale_all_sizes = true;

    std::vector<float> _fixed_sizes;
    std::vector<float> _fixed_ratios;
    std::vector<float> _densitys;

    std::vector<float> _aspect_ratios;
    std::vector<float> _variance;

    int _num_priors = 0;
};

REG_FACTORY_FOR(ImplFactory<PriorBoxImpl>, PriorBox);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
