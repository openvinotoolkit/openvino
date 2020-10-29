// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cmath>
#include <numeric>
#include <iostream>

#include <runtime/pwl.h>

#include "nnet_base_matcher.hpp"

class PWLQuantizationMetricsMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    const float rmse_threshold;
    const uint32_t activation_type;
    const uint16_t segment_threshold;
 public:
    PWLQuantizationMetricsMatcher(uint32_t type, float precision_threshold, uint16_t segments) :
                                                            activation_type(type),
                                                            rmse_threshold(precision_threshold),
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
