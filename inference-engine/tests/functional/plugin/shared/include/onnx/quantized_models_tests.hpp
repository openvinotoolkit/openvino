// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ONNXTestsDefinitions {

class QuantizedModelsTests : public testing::WithParamInterface<std::string>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj);

protected:
    void SetUp() override;
    void runModel(const char* model, const std::unordered_map<std::string, ngraph::element::Type_t>& expected_layer_types);
};

} // namespace ONNXTestsDefinitions
