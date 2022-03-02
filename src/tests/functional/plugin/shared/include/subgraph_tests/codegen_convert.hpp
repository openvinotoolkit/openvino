// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ov::element::Type,           // Input Type
        InferenceEngine::SizeVector, // Input Shape
        std::string                  // Target Device
> convertParams;

class CodegenConvert : public testing::WithParamInterface<LayerTestsDefinitions::convertParams>,
virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::convertParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
