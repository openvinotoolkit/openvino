// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,  //input shapes
        InferenceEngine::Precision,        //Network precision
        std::string,                       //Device name
        std::map<std::string, std::string> //Configuration
> MultioutputEltwiseReshapeEltwiseTuple;

class MultioutputEltwiseReshapeEltwise
        : public testing::WithParamInterface<MultioutputEltwiseReshapeEltwiseTuple>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultioutputEltwiseReshapeEltwiseTuple> &obj);
protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
