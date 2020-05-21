// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#include <gtest/gtest.h>

#include <map>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_core.hpp"

namespace EltwiseTestNamespace {

    using ParameterInputIdx = int;
    enum class InputLayerType {
        CONSTANT,
        PARAMETER
    };
    enum class EltwiseOpType {
        ADD,
        SUBSTRACT,
        MULTIPLY
    };
    const char* InputLayerType_to_string(InputLayerType lt);
    const char* EltwiseOpType_to_string(EltwiseOpType eOp);
}// namespace EltwiseTestNamespace

typedef std::tuple<
    EltwiseTestNamespace::EltwiseOpType,       // eltwise op type
    EltwiseTestNamespace::ParameterInputIdx,   // primary input idx
    EltwiseTestNamespace::InputLayerType,      // secondary input type
    InferenceEngine::Precision,                // Net precision
    InferenceEngine::SizeVector,               // Input shapes
    std::string,                               // Device name
    std::map<std::string, std::string>         // Additional network configuration
> eltwiseLayerTestParamsSet;

class EltwiseLayerTest : public testing::WithParamInterface<eltwiseLayerTestParamsSet>,
    public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(testing::TestParamInfo<eltwiseLayerTestParamsSet> obj);
};
