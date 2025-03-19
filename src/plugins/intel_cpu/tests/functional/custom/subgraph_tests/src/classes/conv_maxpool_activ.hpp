// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class ConvPoolActivTest : public testing::WithParamInterface<fusingSpecificParams>,
                          public CpuTestWithFusing,
                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fusingSpecificParams> obj);

protected:
    void SetUp() override;
    bool primTypeCheck(std::string primType) const override;
};

}  // namespace test
}  // namespace ov