// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_transpose_reorder.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"
#include "openvino/openvino.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string FuseTransposeAndReorderTest::getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj) {
    std::ostringstream result;
    ov::Shape input_shape;
    ov::element::Type in_prec;
    std::tie(input_shape, in_prec) = obj.param;

    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "Precision=" << in_prec.to_string();

    return result.str();
}

void FuseTransposeAndReorderTest::check_transpose_count(size_t expectedTransposeCount) {
    auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(nullptr, runtime_model);
    size_t actual_transpose_count = 0;
    for (const auto &node : runtime_model->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };
        if (getExecValue(ov::exec_model_info::LAYER_TYPE) == "Transpose") {
            actual_transpose_count++;
        }
    }

    ASSERT_EQ(expectedTransposeCount, actual_transpose_count);
}

void FuseTransposeAndReorderTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(input_shape, in_prec) = this->GetParam();
    create_model();
}

} // namespace test
} // namespace ov