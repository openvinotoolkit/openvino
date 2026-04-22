// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_tensor_utils.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> createMaxPoolModel() {
    auto input =
        std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{1, 16, 720, ov::Dimension(10, 1280)});
    input->set_friendly_name("input1");

    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                         Strides{1, 1},
                                                         Shape{0, 0},
                                                         Shape{0, 0},
                                                         Shape{1, 1},
                                                         op::RoundingType::FLOOR,
                                                         op::PadType::EXPLICIT);
    maxpool->set_friendly_name("MaxPool_2");

    auto result = std::make_shared<ov::op::v0::Result>(maxpool);
    result->set_friendly_name("output");

    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "MaxPool");
}

using InferWithHostCompileParams = std::tuple<std::string,  // Device name
                                              ov::AnyMap    // Config
                                              >;

// These tests are required by the NPU plugin to verify the support of dynamic shape during
// compilation and inference on different NPU drivers
class InferWithHostCompileTests : public testing::WithParamInterface<InferWithHostCompileParams>,
                                  public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferWithHostCompileParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        return result.str();
    }

    void SetUp() {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        std::tie(target_device, configuration) = this->GetParam();

        std::vector<std::string> deviceNames =
            core->get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
        for (auto name : deviceNames) {
            if (target_device.find(name) != std::string::npos) {
                isTargetDevice = true;
                break;
            }
        }

        APIBaseTest::SetUp();
    }

    void TearDown() {
        core->set_property("NPU", ov::log::level(ov::log::Level::ERR));
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    bool isTargetDevice = false;
};

TEST_P(InferWithHostCompileTests, Compile) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    ov::InferRequest reqDynamic;

    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        if (std::string(e.what()).find("Cannot load library") == std::string::npos) {
            FAIL() << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        }
    }
}

TEST_P(InferWithHostCompileTests, CompileAndImport) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    ov::CompiledModel importedModel;
    OV_ASSERT_NO_THROW(core->import_model(modelStream, target_device, configuration));

    ov::InferRequest reqDynamic;

    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        if (std::string(e.what()).find("Cannot load library") == std::string::npos) {
            FAIL() << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        }
    }
}

TEST_P(InferWithHostCompileTests, CompileAndImportAndInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    ov::CompiledModel importedModel;
    OV_ASSERT_NO_THROW(core->import_model(modelStream, target_device, configuration));

    ov::InferRequest reqDynamic;

    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        if (std::string(e.what()).find("Cannot load library") == std::string::npos) {
            FAIL() << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        } else {
            GTEST_SKIP() << "Cannot load library, skip test.";
        }
    }

    OV_ASSERT_NO_THROW(reqDynamic.infer());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
