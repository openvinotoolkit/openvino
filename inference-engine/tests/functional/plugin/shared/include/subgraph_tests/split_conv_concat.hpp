// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class SplitConvConcat : public LayerTestsUtils::LayerTestsCommonClass<LayerTestsUtils::basicParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions