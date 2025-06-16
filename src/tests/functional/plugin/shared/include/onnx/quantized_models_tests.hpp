// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ONNXTestsDefinitions {

class QuantizedModelsTests : public testing::WithParamInterface<std::string>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj);

protected:
    void SetUp() override;
    using LayerInputTypes = std::unordered_map<std::string, std::vector<ov::element::Type>>;
    void run_model(const char* model, const LayerInputTypes& expected_layer_input_types, float thr);
};

} // namespace ONNXTestsDefinitions
