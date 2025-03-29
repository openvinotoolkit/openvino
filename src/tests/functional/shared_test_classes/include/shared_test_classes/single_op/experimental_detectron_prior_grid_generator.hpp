// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
class ExperimentalDetectronPriorGridGeneratorTestParam {
public:
};

typedef std::tuple<
    std::vector<InputShape>,
    ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes,
    ElementType,                // Model type
    std::string                 // Device name
> ExperimentalDetectronPriorGridGeneratorTestParams;

class ExperimentalDetectronPriorGridGeneratorLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronPriorGridGeneratorTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorTestParams>& obj);
};
} // namespace test
} // namespace ov
