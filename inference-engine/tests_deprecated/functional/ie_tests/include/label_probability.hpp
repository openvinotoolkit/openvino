// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace Regression {
namespace Reference {

/**
 * @class LabelProbability
 * @brief A LabelProbability represents predicted data in easy to use format
 */
class LabelProbability {
private:
    /**
     * @brief Index of current label
     */
    int labelIdx = 0;
    /**
     * @brief Name of class from file with labels
     */
    std::string className;
    /**
     * @brief The probability of prediction
     */
    float probability = 0.0f;

public:
    /**
     * @brief A constructor of InferenceResults class
     * @param labelIdx - index of current label
     * @param probability - the probability of prediction
     * @param className - name of class from file with labels
     * @return InferenceResults object
     */
    LabelProbability(int labelIdx, float probability, std::string className) : labelIdx(labelIdx),
                                                                               className(className),
                                                                               probability(probability) {}

    /**
     * @brief Gets label index
     * @return index of current label
     */
    const int &getLabelIndex() const {
        return labelIdx;
    }

    /**
     * @brief Gets label name
     * @return label
     */
    const std::string &getLabel() const {
        return className;
    }

    /**
     * @brief Gets probability
     * @return probability
     */
    const float &getProbability() const {
        return probability;
    }
};

}  // namespace Reference
}  // namespace Regression

