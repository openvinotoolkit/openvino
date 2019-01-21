// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <vector>
#include <string>
#include <cmath>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PriorBoxImpl: public ExtLayerBase {
public:
    explicit PriorBoxImpl(const CNNLayer *layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->dims.size() != 4 ||
                    layer->insData[1].lock()->dims.size() != 4)
                THROW_IE_EXCEPTION << "PriorBox supports only 4D blobs!";

            _offset = layer->GetParamAsFloat("offset");
            _step = layer->GetParamAsFloat("step", 0);
            _min_sizes = layer->GetParamAsFloats("min_size", {});
            _max_sizes = layer->GetParamAsFloats("max_size", {});
            _flip = static_cast<bool>(layer->GetParamAsInt("flip"));
            _clip = static_cast<bool>(layer->GetParamAsInt("clip"));
            _scale_all_sizes = static_cast<bool>(layer->GetParamAsInt("scale_all_sizes", 1));

            bool exist;

            _aspect_ratios.push_back(1.0f);

            const std::vector<float> aspect_ratios = layer->GetParamAsFloats("aspect_ratio", {});

            for (float aspect_ratio : aspect_ratios) {
                exist = false;

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

        float* dst_data = dstMemPtr->buffer();

        int dim = H * W * _num_priors * 4;
        int idx = 0;
        float center_x = 0.0f;
        float center_y = 0.0f;

        float box_width;
        float box_height;

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
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

                    dst_data[idx++] = (center_x - box_width / 2.0f) / IW;
                    dst_data[idx++] = (center_y - box_height / 2.0f) / IH;
                    dst_data[idx++] = (center_x + box_width / 2.0f) / IW;
                    dst_data[idx++] = (center_y + box_height / 2.0f) / IH;

                    if (_max_sizes.size() > msIdx) {
                        box_width = box_height = sqrt(_min_sizes[msIdx] * _max_sizes[msIdx]);

                        dst_data[idx++] = (center_x - box_width / 2.0f) / IW;
                        dst_data[idx++] = (center_y - box_height / 2.0f) / IH;
                        dst_data[idx++] = (center_x + box_width / 2.0f) / IW;
                        dst_data[idx++] = (center_y + box_height / 2.0f) / IH;
                    }

                    if (_scale_all_sizes || (!_scale_all_sizes && (msIdx == _min_sizes.size() - 1))) {
                        size_t sIdx = _scale_all_sizes ? msIdx : 0;
                        for (float ar : _aspect_ratios) {
                            if (fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            box_width = _min_sizes[sIdx] * sqrt(ar);
                            box_height = _min_sizes[sIdx] / sqrt(ar);

                            dst_data[idx++] = (center_x - box_width / 2.0f) / IW;
                            dst_data[idx++] = (center_y - box_height / 2.0f) / IH;
                            dst_data[idx++] = (center_x + box_width / 2.0f) / IW;
                            dst_data[idx++] = (center_y + box_height / 2.0f) / IH;
                        }
                    }
                }
            }
        }

        if (_clip) {
            for (int d = 0; d < dim; ++d) {
                dst_data[d] = (std::min)((std::max)(dst_data[d], 0.0f), 1.0f);
            }
        }

        int channel_size = OH * OW;

        dst_data += channel_size;

        int count = 0;
        if (_variance.size() == 1) {
            for (int i = 0; i < channel_size; i++) {
                dst_data[i] = _variance[0];
            }
        } else {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int i = 0; i < _num_priors; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            dst_data[count] = _variance[j];
                            ++count;
                        }
                    }
                }
            }
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

    std::vector<float> _aspect_ratios;
    std::vector<float> _variance;

    int _num_priors = 0;
};

REG_FACTORY_FOR(ImplFactory<PriorBoxImpl>, PriorBox);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
