/*
 * INTEL CONFIDENTIAL
 * Copyright (C) 2018-2019 Intel Corporation.
 *
 * The source code contained or described herein and all documents
 * related to the source code ("Material") are owned by Intel Corporation
 * or its suppliers or licensors. Title to the Material remains with
 * Intel Corporation or its suppliers and licensors. The Material may
 * contain trade secrets and proprietary and confidential information
 * of Intel Corporation and its suppliers and licensors, and is protected
 * by worldwide copyright and trade secret laws and treaty provisions.
 * No part of the Material may be used, copied, reproduced, modified,
 * published, uploaded, posted, transmitted, distributed, or disclosed
 * in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by implication,
 * inducement, estoppel or otherwise. Any license under such intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Include any supplier copyright notices as supplier requires Intel to use.
 *
 * Include supplier trademarks or logos as supplier requires Intel to use,
 * preceded by an asterisk. An asterisked footnote can be added as follows:
 * *Third Party trademarks are the property of their respective owners.
 *
 * Unless otherwise agreed by Intel in writing, you may not remove or alter
 * this notice or any other notice embedded in Materials by Intel or Intel's
 * suppliers or licensors in any way.
 */

#pragma once
#include <cmath>
#include <numeric>

#include "nnet_base_matcher.hpp"
#include "dnn.h"
#include "pwl.h"
#include "iostream"

class PWLQuantizationMetricsMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    const float rmse_threshold;
    const uint32_t activation_type;
    const uint16_t segment_threshold;
 public:
    PWLQuantizationMetricsMatcher(uint32_t type, float presicion_threshold, uint16_t segments) :
                                                            activation_type(type),
                                                            rmse_threshold(presicion_threshold),
                                                            segment_threshold(segments) {}

    bool MatchAndExplain(const intel_nnet_type_t *nnet, ::testing::MatchResultListener *listener) const override {
        float rmse = 0.f;
        const float test_arg_scale_factor = 16384;

        if (nnet == nullptr)
            return false;

        for(int i = 0; i < nnet->nLayers; ++i) {
            if (nnet->pLayers[i].nLayerKind != INTEL_AFFINE &&
                nnet->pLayers[i].nLayerKind != INTEL_AFFINE_DIAGONAL &&
                nnet->pLayers[i].nLayerKind != INTEL_CONVOLUTIONAL) continue;

            auto affine = reinterpret_cast<intel_affine_layer_t*>(nnet->pLayers[i].pLayerStruct);

            if (affine == nullptr ||
                affine->pwl.nSegments == 0 ||
                affine->pwl.pSegments == nullptr) continue;

            if (affine->pwl.nSegments > segment_threshold) {
                return false;
            }

            int32_t domain = 0;
            std::function<float(float)> activation_func = nullptr;
            switch (activation_type) {
                case kActSigmoid:
                    domain = 10000;
                    activation_func = [](float x)-> float {
                                    float exp_value;
                                    exp_value =
                                            exp(static_cast<double>(-(x)));
                                    return  1 / (1 + exp_value);};
                    break;
                case kActTanh:
                    domain = 5000;
                    activation_func = [](float x)-> float {return tanh(x);};
                    break;
                case kActIdentity:
                    domain = 1000;
                    activation_func = [](float x)-> float {return x;};
                    break;
                case kActRelu:
                    domain = 1000;
                    activation_func = [](float x)-> float {return relu(x);};
                    break;
                case kActLeakyRelu:
                    domain = 1000;
                    activation_func = [](float x)-> float {return leaky_relu(x);};
                    break;
                case kActKaldiLstmClipping:
                    domain = 16000;
                    activation_func = [](float x)-> float {
                                        return clipping(x,
                                                KALDI_LSTM_CLIP_LOWER,
                                                KALDI_LSTM_CLIP_UPPER);};
                    break;
                default:
                    domain = 50000;
                    activation_func = [](float x)-> float {return 0;};
            }

            std::vector<double> y_diviation(2*domain);
            std::vector<intel_pwl_segment_t*> segments_vector(affine->pwl.nSegments);
            std::iota(segments_vector.begin(), segments_vector.begin()+affine->pwl.nSegments,
                                                                                affine->pwl.pSegments);

            auto current_segment = segments_vector.begin();
            auto diviation_itr = y_diviation.begin();

            for(int i=-domain; i<domain; ++i) {
                float value = 0.0;
                const float arg = i/1000.0;
                while(current_segment != segments_vector.end() &&
                        arg > static_cast<int32_t>((*current_segment)->xBase & XBASEMASK) / test_arg_scale_factor) {
                    ++current_segment;
                }
                auto prev_segment = std::prev(current_segment,1);
                value = activation_func(arg);

                float base_arg = static_cast<int32_t>((*prev_segment)->xBase & XBASEMASK) / test_arg_scale_factor;
                float base_value = static_cast<int32_t>((*prev_segment)->yBase) / ACTIVATION_SCALE_FACTOR;

                uint32_t slope_scale_index = (*prev_segment)->xBase & ~XBASEMASK;

                uint64_t slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                float slope =
                        test_arg_scale_factor*(static_cast<float>((*prev_segment)->slope ) / (slope_scale*ACTIVATION_SCALE_FACTOR));

                float quant_value = (arg - base_arg)*slope + base_value;

                *diviation_itr = std::pow(std::abs(value-quant_value),2);
                ++diviation_itr;
            }

            // sort ascending to do not lost precision
            std::sort(y_diviation.begin(),y_diviation.end());
            double sum = std::accumulate(y_diviation.begin(), y_diviation.end(), 0.0);
            rmse = std::sqrt(sum/static_cast<float>(y_diviation.size()));
        }

        return rmse_threshold > rmse;
    };
    void DescribeTo(::std::ostream *os) const override {
        *os << "Has the activation layer type " <<  activation_type <<" rmse less that threshold "<< rmse_threshold
                                                << " or segments count less that threshold " <<  segment_threshold
                                                << " ?";
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> PrecisionOfQuantizedPwlMetrics(uint32_t type,
                                                                                    float threshold,
                                                                                    uint16_t segments) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new PWLQuantizationMetricsMatcher(type, threshold, segments));
    return ::testing::MakeMatcher(c.release());
}
