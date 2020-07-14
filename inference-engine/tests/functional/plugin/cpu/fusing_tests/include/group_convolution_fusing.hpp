// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/group_convolution.hpp>
#include "../../test_utils/fusing_test_utils.hpp"

using namespace FusingTestUtils;
using namespace CPUTestUtils;

namespace CPUFusingTestsDefinitions {

typedef std::tuple<
        groupConvLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams> groupConvLayerFusingTestParamsSet;

class GroupConvolutionLayerFusingTest : public testing::WithParamInterface<groupConvLayerFusingTestParamsSet>,
                                        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerFusingTestParamsSet> obj);

protected:
    void SetUp();

    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;

    std::shared_ptr<ngraph::Function> postFunction;
    std::vector<std::shared_ptr<ngraph::Node>> postNodes;
    std::vector<std::string> fusedOps;
};

} // namespace CPUFusingTestsDefinitions