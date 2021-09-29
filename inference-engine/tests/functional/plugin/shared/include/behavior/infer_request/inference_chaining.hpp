// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/function.hpp"

namespace ov {
namespace test {

using OVInferenceChaining = ov::test::BehaviorTestsBasic;

std::shared<ov::Function> getStaticFunction(ov::element::Type type) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape { 1, 3, 10, 10 });
    auto conv = 

    return std::make_shared<ov::Function>(param, result);
}

TEST_P(OVInferenceChaining, StaticOutputToStaticInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::runtime::ExecutableNetwork execNet1;
    ASSERT_NO_THROW(execNet1 = ie->compile_model(function, targetDevice, configuration));

}

}  // namespace test
}  // namespace ov
