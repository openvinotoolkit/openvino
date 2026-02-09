// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
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

using InferWithHostCompileParams = std::tuple<std::string,      // Device name
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
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    ov::AnyMap configuration;
};


// Compile model, infer with min size
TEST_P(InferWithHostCompileTests, DISABLED_CompileModelAndInferWithMinSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    // Reshape with dynamic dimension
    model->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 10, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ov::CompiledModel execNetDynamic;
    ov::InferRequest reqDynamic;

    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile fail now
// Compile model, infer with medium size
TEST_P(InferWithHostCompileTests, DISABLED_CompileModelAndInferWithMediumSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    // Reshape with dynamic dimension
    model->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 600, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile fail now
// Compile model, infer with min size
TEST_P(InferWithHostCompileTests, DISABLED_CompileModelAndInferWithMaxSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto model = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    // Reshape with dynamic dimension
    model->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Plugin shall fix
//  Compile model, process to fp16, set size smaller than min size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithoutSet) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.infer());
}

// Plugin need fix
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithoutSetUpdateShape) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    ov::Shape newShape = {1, 16, 400, 1280};
    auto tensor = reqDynamic.get_input_tensor();
    tensor.set_shape(newShape);
    OV_ASSERT_NO_THROW(reqDynamic.infer());
}

// Test sync infers
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndSyncInfersWithoutSet) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const int inferReqNumber = 256;
    ov::InferRequest reqDynamic;
    ov::Tensor input_tensor;
    for (int i = 0; i < inferReqNumber; i++) {
        OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
        OV_ASSERT_NO_THROW(input_tensor = reqDynamic.get_input_tensor());
        OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(input_tensor));
        OV_ASSERT_NO_THROW(reqDynamic.infer());
        OV_ASSERT_NO_THROW(reqDynamic.get_output_tensor());
    }
}

// Plugin shall fix
//  Compile model, process to fp16, set size smaller than min size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithIlegalSmallSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 5, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_THROW(reqDynamic.set_tensor(inputName, inTensor), ov::Exception);
}

// Plugin shall fix
// Compile model, process to fp16, set size larger than max size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithIlegalMaxSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 800, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_THROW(reqDynamic.set_tensor(inputName, inTensor), ov::Exception);
}

// Inference fail now
// Compile model, process to fp16, infer with min size
TEST_P(InferWithHostCompileTests, DISABLED_CompileModelProcessToFp16WithRangeAndInferWithMinSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 10, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile model, process to fp16, infer with medium size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithMediumSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 600, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile model, process to fp16, infer with max size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithMaxSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile model, process to fp16, infer with size from small to large
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithIncreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 100, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 360, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape2 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 600, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape3 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";
}

// Compile model, process to fp16, infer with size from large to small
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithDecreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 700, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 650, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape2 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 200, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape3 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";
}

// Compile model, process to fp16, infer with random size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithRandomSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 700, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 350, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape2 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 500, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape3 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";

    ov::Shape shape4 = {1, 16, 100, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape4 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";
}

// compiler complain about upper bound not set
//  Compile model, process to fp16, infer with random size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithoutRangeAndInferWithRandomSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 700, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 350, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape2 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 500, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape3 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";

    ov::Shape shape4 = {1, 16, 100, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape4 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";
}

// Compile model, process to fp16, infer with random size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithN1AndInferWithRandomSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, -1, 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 700, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 350, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape2 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 500, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape3 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";

    ov::Shape shape4 = {1, 16, 100, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape4 == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";
}

// Compile model, process to fp16, infer with different inferequest
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithDifferentInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // Test request 1
    ov::Shape shape = {1, 16, 100, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    // cTest request 2
    ov::Shape shape2 = {1, 16, 360, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    ov::InferRequest reqDynamic2;
    OV_ASSERT_NO_THROW(reqDynamic2 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic2.infer());
    ASSERT_TRUE(shape2 == reqDynamic2.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic2.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    ov::Shape shape3 = {1, 16, 720, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic2.infer());
    ASSERT_TRUE(shape3 == reqDynamic2.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic2.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";

    // Test request 3
    ov::Shape shape4 = {1, 16, 700, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    ov::InferRequest reqDynamic3;
    OV_ASSERT_NO_THROW(reqDynamic3 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic3.set_tensor(inputName, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic3.infer());
    ASSERT_TRUE(shape4 == reqDynamic3.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic3.get_output_tensor().get_shape())
        << "Output tensor of third inferrequest from model does not has same shape with input tensor";

    ov::Shape shape5 = {1, 16, 360, 1280};
    ov::Tensor inTensor5 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape5, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic3.set_tensor(inputName, inTensor5));
    OV_ASSERT_NO_THROW(reqDynamic3.infer());
    ASSERT_TRUE(shape5 == reqDynamic3.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic3.get_output_tensor().get_shape())
        << "Output tensor of third inferrequest from model does not update shape to small shape";
}

// Compile model, process to fp16, infer with medium size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithMediumSizeAndHostTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    const std::string inputName = "input1";
    const std::string outputName = "output";

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 600, 1280};
    auto context = ie->get_default_context(target_device);
    ov::Tensor inTensor = context.create_host_tensor(model->input().get_element_type(), shape);
    ov::Tensor outputTensor = context.create_host_tensor(model->output().get_element_type(), shape);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.set_output_tensor(outputTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

// Compile model, process to fp16, infer with medium size
TEST_P(InferWithHostCompileTests, CompileModelProcessToFp16WithRangeAndInferWithMediumSizeAndRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    const std::string inputName = "input1";
    const std::string outputName = "output";

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 600, 1280};
    auto context = ie->get_default_context(target_device);
    ov::Tensor inTensor = context.create_tensor(model->input().get_element_type(), shape);
    ov::Tensor outputTensor = context.create_tensor(model->output().get_element_type(), shape);

    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    // OV_ASSERT_NO_THROW(reqDynamic.set_tensor(outputName, outputTensor));
    OV_ASSERT_NO_THROW(reqDynamic.set_output_tensor(outputTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

TEST_P(InferWithHostCompileTests,
       CompileModelProcessToFp16WithRangeAndInferWithDifferentInferRequestShareUserInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // Test request 1
    ov::Shape shape = {1, 16, 100, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic1;
    OV_ASSERT_NO_THROW(reqDynamic1 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic2;
    OV_ASSERT_NO_THROW(reqDynamic2 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor));

    OV_ASSERT_NO_THROW(reqDynamic.start_async());
    OV_ASSERT_NO_THROW(reqDynamic1.start_async());
    OV_ASSERT_NO_THROW(reqDynamic2.start_async());

    OV_ASSERT_NO_THROW(reqDynamic.wait());
    OV_ASSERT_NO_THROW(reqDynamic1.wait());
    OV_ASSERT_NO_THROW(reqDynamic2.wait());

    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

TEST_P(InferWithHostCompileTests,
       CompileModelProcessToFp16WithRangeAndInferWithDifferentInferRequestShareHostInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // Test request 1
    ov::Shape shape = {1, 16, 100, 1280};
    auto context = ie->get_default_context(target_device);
    ov::Tensor inTensor = context.create_host_tensor(model->input().get_element_type(), shape);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic1;
    OV_ASSERT_NO_THROW(reqDynamic1 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic2;
    OV_ASSERT_NO_THROW(reqDynamic2 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor));

    OV_ASSERT_NO_THROW(reqDynamic.start_async());
    OV_ASSERT_NO_THROW(reqDynamic1.start_async());
    OV_ASSERT_NO_THROW(reqDynamic2.start_async());

    OV_ASSERT_NO_THROW(reqDynamic.wait());
    OV_ASSERT_NO_THROW(reqDynamic1.wait());
    OV_ASSERT_NO_THROW(reqDynamic2.wait());

    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

TEST_P(InferWithHostCompileTests,
       CompileModelProcessToFp16WithRangeAndInferWithDifferentInferRequestShareRemoteInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto dynamicShapeModel = ie->read_model(MaxPoolModelXmlString, ov::Tensor());
    dynamicShapeModel->reshape({{1, 16, ov::Dimension(10, 720), 1280}});
    ov::Shape MaxShape = {1, 16, 720, 1280};

    // Have to process to fp16, otherwise compilation will fail
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

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));

    const std::string inputName = "input1";
    const std::string outputName = "output";

    // Test request 1
    ov::Shape shape = {1, 16, 100, 1280};
    auto context = ie->get_default_context(target_device);
    ov::Tensor inTensor = context.create_tensor(model->input().get_element_type(), shape);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic1;
    OV_ASSERT_NO_THROW(reqDynamic1 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(inputName, inTensor));

    ov::InferRequest reqDynamic2;
    OV_ASSERT_NO_THROW(reqDynamic2 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor));

    OV_ASSERT_NO_THROW(reqDynamic.start_async());
    OV_ASSERT_NO_THROW(reqDynamic1.start_async());
    OV_ASSERT_NO_THROW(reqDynamic2.start_async());

    OV_ASSERT_NO_THROW(reqDynamic.wait());
    OV_ASSERT_NO_THROW(reqDynamic1.wait());
    OV_ASSERT_NO_THROW(reqDynamic2.wait());

    ASSERT_TRUE(shape == reqDynamic.get_output_tensor().get_shape() ||
                MaxShape == reqDynamic.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov