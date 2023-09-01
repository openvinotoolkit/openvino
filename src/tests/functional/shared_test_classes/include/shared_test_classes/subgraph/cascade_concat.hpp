// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

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
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CascadeConcatTuple> &obj);
protected:
    void SetUp() override;
};

typedef std::tuple<
        std::vector<size_t>,               //input shapes
        InferenceEngine::Precision,        //Network precision
        std::string,                       //Device name
        std::map<std::string, std::string> //config
        > CascadeConcatWithMultiConnReshapeTuple;

class CascadeConcatWithMultiConnReshape
        : public testing::WithParamInterface<CascadeConcatWithMultiConnReshapeTuple>,
          virtual public LayerTestsUtils::LayerTestsCommon {
              public:
static std::string getTestCaseName(const testing::TestParamInfo<CascadeConcatWithMultiConnReshapeTuple> &obj);
protected:
    void SetUp() override;
};

} // namespace SubgraphTestsDefinitions
