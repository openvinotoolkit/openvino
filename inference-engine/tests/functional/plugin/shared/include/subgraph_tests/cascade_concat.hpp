// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>, //input shapes 1
        std::vector<std::vector<size_t>>, //input shapes 2
        std::vector<std::vector<size_t>>, //input shapes 3
        InferenceEngine::Precision,       //Network precision
        bool,                             //Multioutput -> True, Single out ->false
        std::string,                      //Device name
        std::map<std::string, std::string>//config
        > CascadeConcatTuple;

class CascadeConcat
        : public testing::WithParamInterface<CascadeConcatTuple>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CascadeConcatTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
