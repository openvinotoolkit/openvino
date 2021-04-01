// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <string>
#include "label_probability.hpp"

namespace Regression {
namespace Reference {

struct ClassificationScoringResultsForTests : public LabelProbability{
    ClassificationScoringResultsForTests(float prob, const std::string & label)
            : LabelProbability(0, prob, label ){
    }
};

extern std::map<std::string, std::vector<ClassificationScoringResultsForTests>> values;

}  // namespace Reference
}  // namespace Regression
