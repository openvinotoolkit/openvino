// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "openvino/op/nv12_to_rgb.hpp"

namespace LayerTestsDefinitions {

using ConvertColorNV12ParamsTuple = std::tuple<
        ngraph::Shape,                                 // Input Shape
        ngraph::element::Type,                         // Element type
        bool,                                          // Conversion type
        bool,                                          // 1 or 2 planes
        std::string>;                                  // Device name

class ConvertColorNV12LayerTest : public testing::WithParamInterface<ConvertColorNV12ParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertColorNV12ParamsTuple> &obj);

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions