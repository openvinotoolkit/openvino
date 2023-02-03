// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using ElementType = ov::element::Type_t;
using TargetDevice = std::string;
using LocalElementType = ov::element_type_traits<ngraph::element::i32>::value_type;

using EyeLayerTestParams = std::tuple<std::vector<ov::Shape>,  // eye shape
                                      std::vector<int>,        // output batch shape
                                      std::vector<int>,        // eye params (rows, cols, diag_shift)
                                      ElementType,             // Net precision
                                      TargetDevice>;           // Device name

class EyeLayerTest : public testing::WithParamInterface<EyeLayerTestParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLayerTestParams> obj);
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
