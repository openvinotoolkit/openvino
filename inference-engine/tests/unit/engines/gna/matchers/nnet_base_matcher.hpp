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


class NNetComponentMatcher : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    std::vector<std::shared_ptr<::testing::MatcherInterface<const intel_nnet_type_t*>>> matchers;
    mutable int failIdx = -1;
    mutable std::stringstream reason;
    int bitness;
 public:
    NNetComponentMatcher(int bitness  = 16) : bitness(bitness) {}
    NNetComponentMatcher& add(::testing::MatcherInterface<const intel_nnet_type_t*> * p) {
        matchers.push_back(std::shared_ptr<::testing::MatcherInterface<const intel_nnet_type_t*>>(p));
        return *this;
    }
    bool empty() const {
        return matchers.empty();
    }
    bool MatchAndExplain(const intel_nnet_type_t* foo, ::testing::MatchResultListener* listener) const override {
        if (foo == nullptr)
            return false;
        reason.str("");
        // checking pointers are set
        for (int i=0; i < foo->nLayers; i++) {
            if (nullptr == foo->pLayers[i].pInputs ||
                nullptr == foo->pLayers[i].pOutputs) {
                reason << "input/output pointers in pLayers[" << i << "] shouldn't be null NULL";
                return false;
            }
            if (foo->pLayers[i].nBytesPerInput * 8 != bitness) {
                reason << "numberOfBytes per input in pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                    << foo->pLayers[i].nBytesPerInput;
                return false;
            }

            if (foo->pLayers[i].nBytesPerOutput * 8 != bitness) {
                // if this output is a output to a bias this is fine
                // also if this output is defacto network output - other words this whouldnt use in inputs,
                for (int j=0; j < foo->nLayers; j++) {
                    // bad
                    if (foo->pLayers[j].pInputs == foo->pLayers[i].pOutputs) {
                        reason << "numberOfBytes per output int pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                               << foo->pLayers[i].nBytesPerOutput << "cannot use this output as inputs for layer :" << j;
                        return false;
                    }
                    if (foo->pLayers[j].nLayerKind == INTEL_AFFINE ||
                        foo->pLayers[j].nLayerKind == INTEL_AFFINE_DIAGONAL) {
                        auto pAffine = reinterpret_cast<intel_affine_func_t*>(foo->pLayers[j].pLayerStruct);

                        if (pAffine->pWeights == foo->pLayers[i].pOutputs) {
                            reason << "numberOfBytes per output int pLayers[" << i << "] should be " << (bitness/8) << ", but was "
                                   << foo->pLayers[i].nBytesPerOutput << "cannot use this output as weights for affine layer :" << j;
                            return false;
                        }
                    }
                }
            }
        }

        int i = 0;
        for (auto && matcher : matchers) {
            bool res = matcher->MatchAndExplain(foo, listener);
            if (!res) {
                failIdx = i;
                return false;
            }
            i++;
        }
        return true;
    }

    void DescribeTo(::std::ostream* os) const override {

        if (failIdx != -1) {
            matchers[failIdx]->DescribeTo(os);
            return;
        }

        *os << reason.str();
    }

};

