// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

class ExperimentalDetectronPriorGridGeneratorTestParam {
public:
    ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes attributes;
    std::vector<InputShape> inputShapes;
};

typedef std::tuple<
    ExperimentalDetectronPriorGridGeneratorTestParam,
    std::pair<std::string, std::vector<ov::Tensor>>,
    ElementType,                // Network precision
    std::string                 // Device name>;
> ExperimentalDetectronPriorGridGeneratorTestParams;

class ExperimentalDetectronPriorGridGeneratorLayerTest :
        public testing::WithParamInterface<ExperimentalDetectronPriorGridGeneratorTestParams>,
        virtual public SubgraphBaseTest {
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorTestParams>& obj);
};
} // namespace subgraph
} // namespace test
} // namespace ov
