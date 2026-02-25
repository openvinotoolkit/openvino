// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

        APIBaseTest::SetUp();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
};

TEST_P(InferWithHostCompileTests, CompileAndImport) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    // With HostCompile, the modelStream shall contain "llvm.func"
    std::string line;
    auto pos = modelStream.tellg();
    modelStream.seekg(0, std::ios::beg);
    bool isLLVMStream = false;
    int searchRegion = 10;
    while (std::getline(modelStream, line)) {
        if (line.find("llvm.func") != std::string::npos) {
            modelStream.clear();
            modelStream.seekg(pos);
            isLLVMStream = true;
            break;
        }
        if (searchRegion-- < 0) {
            break;
        }
    }
    modelStream.clear();
    modelStream.seekg(pos);
    ASSERT_TRUE(isLLVMStream) << "CompiledStream from HostCompile mode shall has 'llvm.func' inside it";

    ov::CompiledModel importedModel;
    OV_ASSERT_NO_THROW(core->import_model(modelStream, target_device, configuration));

    ov::InferRequest reqDynamic;
    // Add shape check once npu_mlir_runtime is inside test package
    EXPECT_THROW(reqDynamic = importedModel.create_infer_request(), ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
