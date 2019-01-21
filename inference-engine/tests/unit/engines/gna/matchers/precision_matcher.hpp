// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "nnet_base_matcher.hpp"

class NNetPrecisionMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    GnaPluginTestEnvironment::NnetPrecision nnetPrecision;
    intel_layer_kind_t layerKind = (intel_layer_kind_t)-1;
 public:
    explicit  NNetPrecisionMatcher(GnaPluginTestEnvironment::NnetPrecision nnetPrecision,
                                   intel_layer_kind_t layerKind = (intel_layer_kind_t)-1) : nnetPrecision(nnetPrecision), layerKind(layerKind) {}
    bool MatchAndExplain(const intel_nnet_type_t* foo, ::testing::MatchResultListener* listener) const override {

        auto ioPrecision = (foo->pLayers->nBytesPerInput == nnetPrecision.input_precision.size()) &&
            (foo->pLayers->nBytesPerOutput== nnetPrecision.output_precision.size());
        if (!ioPrecision) {
            return false;
        }
        if (layerKind != (intel_layer_kind_t)-1) {
            if (foo->pLayers->nLayerKind != layerKind) {
                return false;
            }
            switch (layerKind) {
                case INTEL_AFFINE : {
                    auto affine = (intel_affine_layer_t *) (foo->pLayers->pLayerStruct);

                    return affine->affine.nBytesPerBias == nnetPrecision.biases_precision.size() &&
                        affine->affine.nBytesPerWeight == nnetPrecision.weights_precision.size();
                }
                default :
                    return false;
            }

        }
        return  true;
    }

    void DescribeTo(::std::ostream* os) const override {
        *os << "intel_nnet_layer_t nBytesPerInput equals " << nnetPrecision.input_precision.size() << std::endl;
        *os << "intel_nnet_layer_t nBytesPerOutput equals " << nnetPrecision.output_precision.size() << std::endl;
        *os << "intel_nnet_layer_t nBytesPerWeights equals " << nnetPrecision.weights_precision.size() << std::endl;
        *os << "intel_nnet_layer_t nBytesPerBises equals " << nnetPrecision.biases_precision.size() << std::endl;
        *os << "foo->pLayers->nLayerKind INTEL_AFFINE" ;
    }
};

inline ::testing::Matcher<const intel_nnet_type_t*> BitnessOfNNetEq(GnaPluginTestEnvironment::NnetPrecision nnetPrecision,
                                                         intel_layer_kind_t component) {
    std::unique_ptr<NNetComponentMatcher> c (new NNetComponentMatcher());
    c->add(new NNetPrecisionMatcher(nnetPrecision, component));
    return ::testing::MakeMatcher(c.release());
}
