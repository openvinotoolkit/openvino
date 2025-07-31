// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
namespace ov {
namespace test {
namespace behavior {

using InferRequestElementTypeParams = std::tuple<std::string,  // Device name
                                                 ov::AnyMap    // Config
                                                 >;

// These tests are required by the NPU plugin to verify the compatibility of undefined type and dynamic type during
// compilation and inference on different NPU drivers, hence they are kept here. Compared to test in serialize it
// includes an additional comparison of inference results.
class InferRequestElementTypeTests : public testing::WithParamInterface<InferRequestElementTypeParams>,
                                     public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestElementTypeParams> obj) {
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

bool InferRequestElementTypeTests::compareTensorOutputs(const ov::Tensor& dynamicInferenceOutput,
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
TEST_P(InferRequestElementTypeTests, CompareDynamicAndUndefinedTypeNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // Customize a model with a dynamic type
    std::string dynamicTypeModelXmlString = R"V0G0N(<?xml version="1.0"?>
<net name="custom_model" version="11">
    <layers>
        <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
            <data shape="1,1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="Parameter_1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Relu_2" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ReadValue_3" type="ReadValue" version="opset6">
            <data variable_id="my_var" variable_type="dynamic" variable_shape="..." />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Assign_4" type="Assign" version="opset6">
            <data variable_id="my_var" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Squeeze_5" type="Squeeze" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="Output_5">
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Result_6" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
    </edges>
    <rt_info />
</net>
)V0G0N";

    // Customize a model with a undefined type
    std::string undefinedTypeModelXmlString = R"V0G0N(<?xml version="1.0"?>
<net name="custom_model" version="11">
    <layers>
        <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
            <data shape="1,1,128" element_type="f32" />
            <output>
                <port id="0" precision="FP32" names="Parameter_1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Relu_2" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="ReadValue_3" type="ReadValue" version="opset6">
            <data variable_id="my_var" variable_type="undefined" variable_shape="..." />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Assign_4" type="Assign" version="opset6">
            <data variable_id="my_var" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Squeeze_5" type="Squeeze" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="Output_5">
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="Result_6" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
    </edges>
    <rt_info />
</net>
)V0G0N";

    std::stringstream dynamicTypeModelXmlStream, undefinedTypeModelXmlStream, dynamicTypeModelBinStream,
        undefinedTypeModelBinStream;

    // Test whether the serialization results of the two models are the same
    auto dynamicTypeModel = ie->read_model(dynamicTypeModelXmlString, ov::Tensor());
    auto undefinedTypeModel = ie->read_model(undefinedTypeModelXmlString, ov::Tensor());

    // compile the serialized models
    ov::pass::Serialize(dynamicTypeModelXmlStream, dynamicTypeModelBinStream).run_on_model(dynamicTypeModel);
    ov::pass::Serialize(undefinedTypeModelXmlStream, undefinedTypeModelBinStream).run_on_model(undefinedTypeModel);

    ASSERT_TRUE(dynamicTypeModelXmlStream.str() == undefinedTypeModelXmlStream.str())
        << "Serialized XML files are different: dynamic type vs undefined type";

    // Test whether the inference results of the two models are the same
    // set input and output names
    const std::string inputName = "Parameter_1";
    const std::string outputName = "Output_5";

    // create input tensor match the customized models
    ov::Shape shape = {1, 1, 128};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

    auto execNetDynamic = ie->compile_model(dynamicTypeModel, target_device, configuration);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    auto execNetUndefined = ie->compile_model(undefinedTypeModel, target_device, configuration);
    ov::InferRequest reqUndefined;
    OV_ASSERT_NO_THROW(reqUndefined = execNetUndefined.create_infer_request());
    OV_ASSERT_NO_THROW(reqUndefined.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqUndefined.infer());

    // compare the reference outputs between dynamic type model and undefined type model
    ASSERT_TRUE(compareTensorOutputs(reqDynamic.get_tensor(outputName), reqUndefined.get_tensor(outputName)))
        << "Inference results are different: dynamic type vs undefined type";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
