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
