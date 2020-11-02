// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include "base_matcher.hpp"
#include "regression_reference.hpp"
#include <precision_utils.h>
#include <ie_iexecutable_network.hpp>
#include "label_probability.hpp"

namespace Regression { namespace Matchers {

// this is one more version of classification matcher for new api of async/sync requests
class ClassificationMatcher : public BaseMatcher {
private:
    size_t checkResultNumber;
    std::vector<std::shared_ptr<InferenceEngine::IExecutableNetwork>> _executableNetworks;
    std::vector <std::vector<Reference::LabelProbability>> _results;
    ResponseDesc _resp;
    InferenceEngine::InputsDataMap _inputsInfo;
    InferenceEngine::OutputsDataMap _outputsInfo;
 public:
    explicit ClassificationMatcher(RegressionConfig &config);
    void to(std::string modelType);
    void to(const std::vector <Regression::Reference::ClassificationScoringResultsForTests> &expected);


 private:
    void readLabels(std::string labelFilePath);
    int getIndexByLabel(const std::string &label);
    std::string getLabel(unsigned int index);
    void checkResult(size_t checkNumber,
                     const std::vector <Regression::Reference::ClassificationScoringResultsForTests> &expected);
    virtual void match(size_t top);
    void match_n(size_t top, int index);
    void saveResults(const std::vector<unsigned> &topIndexes, const std::vector<float> &probs, size_t top);

    size_t top = 5;
};

} } // namespace matchers
