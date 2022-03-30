// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVCompiledModelPropertiesBase : public APIBaseTest {
public:
    std::shared_ptr<Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> model;
    AnyMap properties;
};

class OVCompiledModelEmptyPropertiesTests : public testing::WithParamInterface<std::string>,
                               public OVCompiledModelPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);

    void SetUp() override;
};

using PropertiesParams = std::tuple<std::string, AnyMap>;

class OVCompiledModelPropertiesTests : public testing::WithParamInterface<PropertiesParams>,
                          public OVCompiledModelPropertiesBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj);

    void SetUp() override;

    void TearDown() override;

protected:
    void set_api_entity() override { api_entity = ov::test::utils::ov_entity::ov_compiled_model; }
};

using OVCompiledModelPropertiesIncorrectTests = OVCompiledModelPropertiesTests;
using OVCompiledModelPropertiesDefaultTests = OVCompiledModelPropertiesTests;

}  // namespace behavior
}  // namespace test
}  // namespace ov