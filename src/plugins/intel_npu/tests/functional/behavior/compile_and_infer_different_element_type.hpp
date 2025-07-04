// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

//add for init  for driver
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include <ze_api.h>
#include <ze_graph_ext.h>

// get platform
#include "common/functions.h"

//add test
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace test {
namespace behavior {

std::shared_ptr<ov::Model> getFunction() {
    const std::vector<size_t> inputShape = {1, 1, 128};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Parameter_1");
    params.front()->get_output_tensor(0).set_names({"Parameter_1"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->set_friendly_name("Relu_2");
    relu->get_output_tensor(0).set_names({"relu_output"});

    auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "my_var"});

    auto read_value = std::make_shared<ov::op::v6::ReadValue>(relu->output(0), variable);
    read_value->set_friendly_name("ReadValue_3");
    read_value->get_output_tensor(0).set_names({"readvalue_output"});

    auto assign = std::make_shared<ov::op::v6::Assign>(read_value->output(0), variable);
    assign->set_friendly_name("Assign_4");
    assign->get_output_tensor(0).set_names({"assign_output"});

    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(assign);
    squeeze->set_friendly_name("Squeeze_5");
    squeeze->get_output_tensor(0).set_names({"Output_5"});

    auto result = std::make_shared<ov::op::v0::Result>(squeeze);
    result->set_friendly_name("Result_6");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "custom_model");
}

class NPUInferRequestElementTypeTests : public OVInferRequestDynamicTests {
protected:
    std::string m_out_xml_path_1{};
    std::string m_out_bin_path_1{};
    std::string m_out_xml_path_2{};
    std::string m_out_bin_path_2{};
    std::string filePrefix{};

    std::string xmlDynamicFileName{};
    std::string xmlUndefinedFileName{};
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> m_initStruct;
    uint32_t m_graphDdiExtVersion;

    void SetupFileNames() {
        filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path_1 = filePrefix + "1" + ".xml";
        m_out_bin_path_1 = filePrefix + "1" + ".bin";
        m_out_xml_path_2 = filePrefix + "2" + ".xml";
        m_out_bin_path_2 = filePrefix + "2" + ".bin";
    }

    void RemoveFiles() {
        std::remove(m_out_xml_path_1.c_str());
        std::remove(m_out_xml_path_2.c_str());
        std::remove(m_out_bin_path_1.c_str());
        std::remove(m_out_bin_path_2.c_str());
    }

    void SetUp() override {
        const auto& var = ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM;
        if (!var.empty()) {
            std::cout << "---> setup.  This test is supported on the current platform: " << var << std::endl;
        } else {
            std::cout << "---> setup.  This test is not supported on the current platform." << std::endl;
        }

        std::cout << " ---> setUp: PlatformEnvironment::PLATFORM is: " <<  PlatformEnvironment::PLATFORM.c_str() << std::endl;
        m_initStruct = ::intel_npu::ZeroInitStructsHolder::getInstance();
        if (!m_initStruct) {
            GTEST_SKIP() << "ZeroInitStructsHolder init failed, ZeroInitStructsHolder is a nullptr. Skip due to failure on device";
            std::cout << "---> setup. init zeroinitholder." << std::endl;
        }

        ze_graph_dditable_ext_decorator& graph_ddi_table_ext = m_initStruct->getGraphDdiTable();
        uint32_t m_graphDdiExtVersion = graph_ddi_table_ext.version();

        if (m_graphDdiExtVersion < ZE_GRAPH_EXT_VERSION_1_5) {
            std::cout << "--test-- m_graphDdiExtVersion is less than ZE_GRAPH_EXT_VERSION_1_5" << std::endl;
        }

        SetupFileNames();
        OVInferRequestDynamicTests::SetUp();
    }

    void TearDown() override {
        RemoveFiles();
        OVInferRequestDynamicTests::TearDown();
    }

    bool files_equal(std::ifstream& f1, std::ifstream& f2) {
        if (!f1.good())
            return false;
        if (!f2.good())
            return false;

        while (!f1.eof() && !f2.eof()) {
            if (f1.get() != f2.get()) {
                return false;
            }
        }

        if (f1.eof() != f2.eof()) {
            return false;
        }

        return true;
    }

    bool checkTwoTypeOutput(const ov::Tensor& dynamicOutput, const ov::Tensor& undefinedOutput) {
        bool result = true;
        const auto dynamicShape = dynamicOutput.get_shape();
        const auto undefinedShape = undefinedOutput.get_shape();
        if (dynamicShape.size() != undefinedShape.size()) {
            return false;
        }
        if (!std::equal(dynamicShape.cbegin(), dynamicShape.cend(), undefinedShape.cbegin())) {
            return false;
        }
        for (int i = 0; i < undefinedOutput.get_size(); i++) {
            if (fabs(dynamicOutput.data<float>()[i] - undefinedOutput.data<float>()[i]) >
                std::numeric_limits<float>::epsilon())
                return false;
        }
        return result;
    }
};

// Test whether the serialization and inference results of the dynamic type model
// and the undefined type model are the same
TEST_P(NPUInferRequestElementTypeTests, CompareDynamicAndUndefinedTypeNetwork) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    // Customize a model with a dynamic type
    std::string dynamicModel = R"V0G0N(<?xml version="1.0"?>
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
    std::string undefinedModel = R"V0G0N(<?xml version="1.0"?>
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

    // Test whether the serialization results of the two models are the same
    auto expectedDynamic = ie->read_model(dynamicModel, ov::Tensor());
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expectedDynamic);
    auto expectedUndefined = ie->read_model(undefinedModel, ov::Tensor());
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expectedUndefined);

    std::ifstream xml_dynamic(m_out_xml_path_1, std::ios::in);
    std::ifstream xml_undefined(m_out_xml_path_2, std::ios::in);

    ASSERT_TRUE(files_equal(xml_dynamic, xml_undefined));

    // Test whether the inference results of the two models are the same
    const std::string inputName = "Parameter_1";
    const std::string outputName = "Output_5";
    ov::Shape shape = inOutShapes[0].first;
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

    auto execNetDynamic = ie->compile_model(expectedDynamic, target_device, configuration);
    ov::InferRequest reqDynamic;
    OV_ASSERT_NO_THROW(reqDynamic = execNetDynamic.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    auto execNetUndefined = ie->compile_model(expectedUndefined, target_device, configuration);
    ov::InferRequest reqUndefined;
    OV_ASSERT_NO_THROW(reqUndefined = execNetUndefined.create_infer_request());
    OV_ASSERT_NO_THROW(reqUndefined.set_tensor(inputName, inTensor));
    OV_ASSERT_NO_THROW(reqUndefined.infer());

    OV_ASSERT_NO_THROW(checkTwoTypeOutput(reqDynamic.get_tensor(outputName), reqUndefined.get_tensor(outputName)));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
