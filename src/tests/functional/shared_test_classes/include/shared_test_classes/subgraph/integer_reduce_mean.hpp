// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,    // input precision
                   std::vector<size_t>,  // input shape
                   std::vector<size_t>,  // axes
                   bool,                 // quantized
                   const char*           // plugin
                   > IntegerReduceMeanParams;

// IntegerReduceMeanTest covers the two rounding scenarios in ReduceMean with integer inputs.
// Scenario 1: ReduceMean has both input and output precisions to be integers from the original model, so rounding to zero should
//             be done before converting intermediate floating point value to integer. Covered by test suite smoke_ReduceMeanIntegerInput.
// Scenario 2: Integer inputs of ReduceMean are resulted from quantization, then such rounding should not be done, in order to maintain
//             accuracy. Coverd by test suite smoke_ReduceMeanQuantized.
class IntegerReduceMeanTest : public testing::WithParamInterface<IntegerReduceMeanParams>,
                       public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IntegerReduceMeanParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
