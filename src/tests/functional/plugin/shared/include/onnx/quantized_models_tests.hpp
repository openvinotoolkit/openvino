// Copyright (C) 2018-2022 Intel Corporation
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
    using LayerInputTypes = std::unordered_map<std::string, std::vector<ngraph::element::Type_t>>;
    void runModel(const char* model, const LayerInputTypes& expected_layer_input_types, float thr);
};

} // namespace ONNXTestsDefinitions
