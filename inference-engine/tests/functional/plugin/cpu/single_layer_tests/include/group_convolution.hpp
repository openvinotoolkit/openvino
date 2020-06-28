// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/group_convolution.hpp>
#include "../../test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        groupConvLayerTestParamsSet,
        CPUSpecificParams> groupConvLayerCPUTestParamsSet;

class GroupConvolutionLayerCPUTest : public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                     public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerCPUTestParamsSet> obj);

protected:
    void SetUp();

    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};

} // namespace CPULayerTestsDefinitions
