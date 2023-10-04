// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        ngraph::Shape,              // input shape
        std::vector<int64_t>,       // pad_begin
        std::vector<int64_t>,       // pad_end
        float,                      // pad_value
        ngraph::op::PadMode,        // pad_mode
        std::string                 // Device name
        > PadParams;

class ConvertPadToConvTests
        : public testing::WithParamInterface<PadParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PadParams> &obj);

protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
