// Copyright (C) 2018-2025 Intel Corporation
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

using InferRequestDynamicShapeParams = std::tuple<std::string,  // Device name
                                                  ov::AnyMap    // Config
                                                  >;

// These tests are required by the NPU plugin to verify the support of dynamic shape during
// compilation and inference on different NPU drivers
class InferRequestDynamicShapeTests : public testing::WithParamInterface<InferRequestDynamicShapeParams>,
                                      public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestDynamicShapeParams> obj) {
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
    bool compareTensorOutputs(const ov::Tensor& dynamicInferenceOutput, const ov::Tensor& undefinedInferenceOutput);

    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    ov::AnyMap configuration;
};

bool InferRequestDynamicShapeTests::compareTensorOutputs(const ov::Tensor& dynamicInferenceOutput,
                                                         const ov::Tensor& undefinedInferenceOutput) {
    const auto dynamicShape = dynamicInferenceOutput.get_shape();
    const auto undefinedShape = undefinedInferenceOutput.get_shape();

    // compare two models' element types
    if (dynamicInferenceOutput.get_element_type() != undefinedInferenceOutput.get_element_type()) {
        return false;
    }

    // compare two models' shapes
    if (dynamicShape.size() != undefinedShape.size()) {
        return false;
    }

    if (!std::equal(dynamicShape.cbegin(), dynamicShape.cend(), undefinedShape.cbegin())) {
        return false;
    }
    // compare two models' data
    for (size_t i = 0; i < undefinedInferenceOutput.get_size(); i++) {
        if (fabs(dynamicInferenceOutput.data<float>()[i] - undefinedInferenceOutput.data<float>()[i]) >
            std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
}

// Test whether the serialization and inference results of the dynamic type model and the undefined type model are the
// same
TEST_P(InferRequestDynamicShapeTests, CompareDynamicAndUndefinedTypeNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

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

    // Test whether the serialization results of the two models are the same
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

    // Test whether the inference results of the two models are the same
    // set input and output names
    const std::string inputName = "input1";
    const std::string outputName = "output";

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 600, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    ov::CompiledModel execNetDynamic;
    OV_ASSERT_NO_THROW(execNetDynamic = ie->compile_model(model, target_device, configuration));
    ov::InferRequest reqDynamic1;
    OV_ASSERT_NO_THROW(reqDynamic1 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    ASSERT_EQ(shape, reqDynamic1.get_output_tensor().get_shape())
        << "Output tensor not has same shape with input tensor";

    std::cout << __LINE__ << std::endl;
    // create input tensor match the customized models
    ov::Shape shape2 = {1, 16, 360, 1280};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    ov::InferRequest reqDynamic2;
    OV_ASSERT_NO_THROW(reqDynamic2 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic2.infer());
    ASSERT_EQ(shape2, reqDynamic2.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not has same shape with input tensor";

    std::cout << __LINE__ << std::endl;
    ov::Shape shape3 = {1, 16, 720, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic2.set_tensor(inputName, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic2.infer());
    ASSERT_EQ(shape3, reqDynamic2.get_output_tensor().get_shape())
        << "Output tensor of second inferrequest from model does not update output tensor shape with larger shape";

    std::cout << __LINE__ << std::endl;
    ov::Shape shape4 = {1, 16, 720, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    ov::InferRequest reqDynamic3;
    OV_ASSERT_NO_THROW(reqDynamic3 = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic3.set_tensor(inputName, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic3.infer());
    ASSERT_EQ(shape4, reqDynamic3.get_output_tensor().get_shape())
        << "Output tensor of third inferrequest from model does not has same shape with input tensor";

    std::cout << __LINE__ << std::endl;
    ov::Shape shape5 = {1, 16, 360, 1280};
    ov::Tensor inTensor5 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape5, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic3.set_tensor(inputName, inTensor5));
    OV_ASSERT_NO_THROW(reqDynamic3.infer());
    ASSERT_EQ(shape5, reqDynamic3.get_output_tensor().get_shape())
        << "Output tensor of third inferrequest from model does not update shape to small shape";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
