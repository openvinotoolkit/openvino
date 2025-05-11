// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> getDefaultNGraphFunctionForTheDeviceNPU(
        std::vector<size_t> inputShape = {1, 2, 32, 32}, ov::element::Type_t ngPrc = ov::element::Type_t::f32) {
    return ov::test::utils::make_conv_pool_relu(inputShape, ngPrc);
}

class OVInferRequestTestsNPU : public OVInferRequestTests {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceNPU();
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, target_device, params);
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov
