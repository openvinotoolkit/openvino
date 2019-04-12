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


class OutputFiller : public ::testing::MatcherInterface<const intel_nnet_type_t*> {
    mutable std::stringstream reason;
    int32_t fill32BValue;
    int16_t fill16BValue;

 public:
    OutputFiller(int32_t fill32BValue, int16_t fill16BValue) : fill32BValue(fill32BValue), fill16BValue(fill16BValue) {}


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
            auto nElements = foo->pLayers[i].nOutputColumns * foo->pLayers[i].nOutputRows;
            if (foo->pLayers[i].nBytesPerOutput == 2) {
                std::fill_n((int16_t *) foo->pLayers[i].pOutputs, nElements, fill16BValue);
            } else if (foo->pLayers[i].nBytesPerOutput == 4) {
                std::fill_n((int32_t *) foo->pLayers[i].pOutputs, nElements, fill32BValue);
            } else {
                reason << "output bitness of layer [" << i << "] shouldn't be 16 or 32, but was " << foo->pLayers[i].nBytesPerOutput;
                return false;
            }
        }
        return true;
    }

    void DescribeTo(::std::ostream *os) const override {
        *os << "Not a Matcher but a fake, but error happened anyway: " << reason.str();
    }

};

