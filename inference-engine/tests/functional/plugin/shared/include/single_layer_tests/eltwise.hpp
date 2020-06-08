// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#include <gtest/gtest.h>

#include <map>
#include <functional_test_utils/layer_test_utils.hpp>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {
namespace EltwiseParams {
enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};

enum class OpType {
    SCALAR,
    VECTOR
};
} // namespace EltwiseParams

typedef std::tuple<
    std::vector<std::vector<size_t>>,             // input shapes
    ngraph::helpers::EltwiseTypes,                // eltwise op type
    EltwiseParams::InputLayerType,                // secondary input type
    EltwiseParams::OpType,                        // op type
    InferenceEngine::Precision,                   // Net precision
    std::string,                                  // Device name
    std::map<std::string, std::string>            // Additional network configuration
> EltwiseTestParams;

class EltwiseLayerTest : public testing::WithParamInterface<EltwiseTestParams>,
    public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseTestParams> obj);
};
} // namespace LayerTestsDefinitions
