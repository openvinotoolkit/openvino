// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<bool,  // is then subgraph constant?
                   bool   // is else subgraph constant?
                   >
    IfConstNonConstTestParams;

class IfConstNonConst : public testing::WithParamInterface<IfConstNonConstTestParams>,
                        virtual public ov::test::SubgraphBaseTest,
                        public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IfConstNonConstTestParams>& obj);
    // void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes);
    // void prepare();
    // void reset();
    // std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
