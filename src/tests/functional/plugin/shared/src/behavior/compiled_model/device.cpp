// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/device.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(OVCompiledModel, OVCompiledModelCorrectDevice, ::testing::Values("CPU", "GPU", "MULTI:CPU(4),GPU(4)"));


TEST_P(OVCompiledModelCorrectDevice, CompileModelWithCorrectDevice) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(auto compiled_model = ie.compile_model(actualNetwork, deviceName));
}

INSTANTIATE_TEST_SUITE_P(ov_compiled_model, OVCompiledModelIncorrectDevice, ::testing::Values("MULTI:CPU(4),GPU(4) ", "MULTI:CPU(4),GPU(4)a"));

TEST_P(OVCompiledModelIncorrectDevice, CanNotCompileModelWithIncorrectDevice) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(auto compiled_model = ie.compile_model(actualNetwork, deviceName), ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
