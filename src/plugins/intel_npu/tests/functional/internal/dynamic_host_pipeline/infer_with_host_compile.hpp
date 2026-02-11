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

// Customize a model with a dynamic shape
std::string MaxPoolModelXmlString = R"V0G0N(<?xml version="1.0"?>
        <net name="MaxPool" version="11">
                <layers>
                        <layer id="0" name="input1" type="Parameter" version="opset1">
                                <data shape="1,1,384,1024" element_type="f32" />
                                <rt_info>
                                        <attribute name="fused_names" version="0" value="input1" />
                                </rt_info>
                                <output>
                                        <port id="0" precision="FP32" names="input1">
                                                <dim>1</dim>
                                                <dim>1</dim>
                                                <dim>384</dim>
                                                <dim>1024</dim>
                                        </port>
                                </output>
                        </layer>
                        <layer id="1" name="MaxPool_2" type="MaxPool" version="opset1">
                                <data strides="1, 1" pads_begin="0, 0" pads_end="0, 0" kernel="1, 1" rounding_type="floor" auto_pad="explicit" />
                                <rt_info>
                                        <attribute name="fused_names" version="0" value="MaxPool_2" />
                                </rt_info>
                                <input>
                                        <port id="0" precision="FP32">
                                                <dim>1</dim>
                                                <dim>1</dim>
                                                <dim>384</dim>
                                                <dim>1024</dim>
                                        </port>
                                </input>
                                <output>
                                        <port id="1" precision="FP32">
                                                <dim>1</dim>
                                                <dim>1</dim>
                                                <dim>384</dim>
                                                <dim>1024</dim>
                                        </port>
                                </output>
                        </layer>
                        <layer id="2" name="output" type="Result" version="opset1">
                                <rt_info>
                                        <attribute name="fused_names" version="0" value="output" />
                                </rt_info>
                                <input>
                                        <port id="0" precision="FP32">
                                                <dim>1</dim>
                                                <dim>1</dim>
                                                <dim>384</dim>
                                                <dim>1024</dim>
                                        </port>
                                </input>
                        </layer>
                </layers>
                <edges>
                        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
                        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
                </edges>
                <rt_info />
        </net>

        )V0G0N";

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
    auto dynamicShapeModel = core->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});

    auto preprocessor = ov::preprocess::PrePostProcessor(dynamicShapeModel);
    const auto inputs = dynamicShapeModel->inputs();
    const auto outputs = dynamicShapeModel->outputs();

    for (size_t i = 0; i < inputs.size(); i++) {
        preprocessor.input(i).tensor().set_element_type(ov::element::f16);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        preprocessor.output(i).tensor().set_element_type(ov::element::f16);
    }
    auto model = preprocessor.build();

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