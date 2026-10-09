// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>

#include "openvino/op/divide.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

class DividePythonDivCPUTest : public testing::WithParamInterface<bool>, public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const bool pythondiv = GetParam();
        init_input_shapes({InputShape{{4}, {{4}}}, InputShape{{4}, {{4}}}});

        ov::ParameterVector parameters;
        for (const auto& shape : inputDynamicShapes) {
            parameters.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape));
        }
        auto divide = std::make_shared<ov::op::v1::Divide>(parameters[0], parameters[1], pythondiv);
        function = std::make_shared<ov::Model>(ov::OutputVector{divide}, parameters, "DividePythonDiv");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        static constexpr std::array<std::array<int32_t, 4>, 2> values = {{{-5, 5, -5, 5}, {2, -2, -2, 2}}};
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            ov::Tensor tensor(ov::element::i32, targetInputStaticShapes[i]);
            std::copy(values[i].begin(), values[i].end(), tensor.data<int32_t>());
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(DividePythonDivCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerDivide,
                         DividePythonDivCPUTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "PythonDiv" : "TruncateDiv";
                         });

}  // namespace ov::test
