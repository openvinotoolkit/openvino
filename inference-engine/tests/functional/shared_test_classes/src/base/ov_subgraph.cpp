// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include <transformations/serialize.hpp>
#include <ngraph/opsets/opset.hpp>
#include <pugixml.hpp>
#include <common_test_utils/file_utils.hpp>
#include <thread>
#include <functional_test_utils/ov_tensor_utils.hpp>

#include "openvino/core/variant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

void SubgraphBaseTest::run() {
    auto crashHandler = [](int errCode) {
        auto &s = LayerTestsUtils::Summary::getInstance();
        s.saveReport();
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    LayerTestsUtils::PassRate::Statuses status =
            FuncTestUtils::SkipTestsConfig::currentTestIsDisabled() ?
            LayerTestsUtils::PassRate::Statuses::SKIPPED : LayerTestsUtils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    OPENVINO_ASSERT(!targetStaticShapes.empty(), "Target Static Shape is empty!!!");
    std::string errorMessage;
    try {
        compile_model();
        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                if (!inputDynamicShapes.empty()) {
                    // resize ngraph function according new target shape
                    resize_ngraph_function(targetStaticShapeVec);
                }
                generate_inputs(targetStaticShapeVec);
                infer();
                validate();
                status = LayerTestsUtils::PassRate::Statuses::PASSED;
            } catch (const std::exception &ex) {
//                OPENVINO_ASSERT("Incorrect target static shape: ", CommonTestUtils::vec2str(targetStaticShape), std::endl, ex.what());
            }
        }
    } catch (const std::exception &ex) {
        status = LayerTestsUtils::PassRate::Statuses::FAILED;
        errorMessage = ex.what();
    } catch (...) {
        status = LayerTestsUtils::PassRate::Statuses::FAILED;
        errorMessage = "Unknown failure occurred.";
    }
    summary.updateOPsStats(function, status);
    GTEST_FATAL_FAILURE_(errorMessage.c_str());
}

void SubgraphBaseTest::serialize() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::string output_name = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + GetTimestamp();

    std::string out_xml_path = output_name + ".xml";
    std::string out_bin_path = output_name + ".bin";

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(out_xml_path, out_bin_path);
    manager.run_passes(function);
    function->validate_nodes_and_infer_types();

    auto result = core->read_model(out_xml_path, out_bin_path);

    bool success;
    std::string message;
    std::tie(success, message) =
            compare_functions(result, function, false, false, false,
                              true,     // precision
                              true);    // attributes

    EXPECT_TRUE(success) << message;

    CommonTestUtils::removeIRFiles(out_xml_path, out_bin_path);
}

void SubgraphBaseTest::query_model() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    auto queryNetworkResult = core->query_model(function, targetDevice);
    std::set<std::string> expected;
    for (auto&& node : function->get_ops()) {
        expected.insert(node->get_friendly_name());
    }

    std::set<std::string> actual;
    for (auto&& res : queryNetworkResult) {
        actual.insert(res.first);
    }
    ASSERT_EQ(expected, actual);
}

//ov::runtime::Tensor SubgraphBaseTest::generate_input(const element::Type& type, const ov::Shape& shape) const {
//    return create_and_fill_tensor(type, shape);
//}
//
//void SubgraphBaseTest::compare(const std::vector<std::pair<element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
//                               const std::vector<ov::runtime::Tensor>& actualOutputs) {
//    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
//        const auto& expected = expectedOutputs[outputIndex];
//        const auto& actual = actualOutputs[outputIndex];
//        compare(expected, actual);
//    }
//}
//
//void SubgraphBaseTest::compare(const std::pair<element::Type, std::vector<std::uint8_t>>& expected,
//                               const ov::runtime::Tensor& actual) {
void SubgraphBaseTest::compare(const std::vector<ov::runtime::Tensor> &expected,
                               const std::vector<ov::runtime::Tensor> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
//    return ::ov::test::utils::compare(expected_tensor, actual, abs_threshold, rel_threshold);
}

//void SubgraphBaseTest::compare_desc(const ov::runtime::Tensor& expected, const ov::runtime::Tensor& actual) {
//    ASSERT_EQ(expected.get_element_type(), actual.get_element_type());
//    ASSERT_EQ(expected.get_shape(), actual.get_shape());
//}

//void SubgraphBaseTest::configure_model() {
//    // configure input precision
//    {
//        auto params = function->get_parameters();
//        for (auto& param : params) {
//            param->get_output_tensor(0).set_element_type(inPrc);
//        }
//    }
//
//    // configure output precision
//    {
//        auto results = function->get_results();
//        for (auto& result : results) {
//            result->get_output_tensor(0).set_element_type(outPrc);
//        }
//    }
//}

void SubgraphBaseTest::compile_model() {
//    configure_model();
    executableNetwork = core->compile_model(function, targetDevice, configuration);
}

void SubgraphBaseTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    for (int i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        ov::runtime::Tensor tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[i]);
        inputs.insert({param->get_friendly_name(), tensor});
    }
}

void SubgraphBaseTest::infer() {
    inferRequest = executableNetwork.create_infer_request();
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }
    inferRequest.infer();
}
//
std::vector<ov::runtime::Tensor> SubgraphBaseTest::calculate_refs() {
    // nGraph interpreter does not support f16/bf16
    ngraph::pass::ConvertPrecision<element::Type_t::f16, element::Type_t::f32>().run_on_function(function);
    ngraph::pass::ConvertPrecision<element::Type_t::bf16, element::Type_t::f32>().run_on_function(function);

    function->validate_nodes_and_infer_types();

//    auto referenceInputs = std::vector<std::vector<uint8_t>>(inputs.size());
//    auto refInputsTypes = std::vector<element::Type>(inputs.size());
//    for (const auto& inputTensor : inputs) {
//        const auto &input = inputTensor.second;
//        const auto &inputSize = input.get_byte_size();
//        auto &referenceInput = referenceInputs[i];
//        referenceInput.resize(inputSize);
//        const auto buffer = static_cast<uint8_t*>(input.data());
//        std::copy(buffer, buffer + inputSize, referenceInput.data());
//        refInputsTypes[i] = input.get_element_type();
//    }
    auto referenceInputs = inputs;

//    auto expectedOutputs = ngraph::helpers::interpreterFunction(function, referenceInputs, refInputsTypes);
    auto expectedOutputs = ngraph::helpers::interpreterFunction(function, referenceInputs);
    return expectedOutputs;
}
//
std::vector<ov::runtime::Tensor> SubgraphBaseTest::get_outputs() {
    auto outputs = std::vector<ov::runtime::Tensor>{};
    for (const auto& output : executableNetwork.get_results()) {
        const auto& name = output->input_value(0).get_node()->get_friendly_name();
        outputs.push_back(inferRequest.get_tensor(name));
    }
    return outputs;
}
//
void SubgraphBaseTest::validate() {
    auto expectedOutputs = calculate_refs();
    const auto& actualOutputs = get_outputs();

    if (expectedOutputs.empty()) {
        return;
    }

    OPENVINO_ASSERT(actualOutputs.size() == expectedOutputs.size(),
                    "nGraph interpreter has ", expectedOutputs.size(), " outputs, while IE ", actualOutputs.size());

    compare(expectedOutputs, actualOutputs);
}

//std::string SubgraphBaseTest::get_runtime_precision(const std::string& layerName) {
//    const auto function = executableNetwork.get_runtime_function();
//    for (const auto& op : function->get_ops()) {
//        const auto name = op->get_friendly_name();
//        if (name == layerName) {
//            const auto& rtInfo = op->get_rt_info();
//            const auto& it = rtInfo.find("runtimePrecision");
//
//            OPENVINO_ASSERT(it != rtInfo.end(), "Runtime precision is not found for node: ", name);
//
//            const auto rtPrecisionPtr = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(it->second);
//            return rtPrecisionPtr->get();
//        }
//    }
//    return {};
//}

//std::string SubgraphBaseTest::get_runtime_precision_by_type(const std::string& layerType) {
//    const auto function = executableNetwork.get_runtime_function();
//
//    for (const auto& op : function->get_ops()) {
//        const auto& rtInfo = op->get_rt_info();
//        const auto& typeIt = rtInfo.find("layerType");
//
//        OPENVINO_ASSERT(typeIt != rtInfo.end(), "Layer is not found for type: ", layerType);
//
//        const auto type = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(typeIt->second)->get();
//        if (type == layerType) {
//            const auto& it = rtInfo.find("runtimePrecision");
//
//            OPENVINO_ASSERT(it != rtInfo.end(), "Runtime precision is not found for node: ", type);
//
//            const auto rtPrecisionPtr = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(it->second);
//            return rtPrecisionPtr->get();
//        }
//    }
//
//    return {};
//}

//#ifndef NDEBUG
//void SubgraphBaseTest::show_runtime_precision() {
//    const auto function = executableNetwork.get_runtime_function();
//
//    for (const auto& op : function->get_ops()) {
//        const auto& rtInfo = op->get_rt_info();
//        const auto& typeIt = rtInfo.find("layerType");
//
//        const auto type = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(typeIt->second)->get();
//        const auto& it = rtInfo.find("runtimePrecision");
//
//        const auto rtPrecisionPtr = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(it->second);
//        std::cout << type << ": " << rtPrecisionPtr->get() << std::endl;
//    }
//}
//#endif

//std::shared_ptr<ngraph::Function> SubgraphBaseTest::get_function() {
//    return function;
//}
//
//std::map<std::string, std::string>& SubgraphBaseTest::get_configuration() {
//    return configuration;
//}

void SubgraphBaseTest::resize_ngraph_function(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    auto params = function->get_parameters();
    std::map<std::string, ngraph::PartialShape> shapes;
    ASSERT_LE(params.size(), targetInputStaticShapes.size());
    for (size_t i = 0; i < params.size(); i++) {
        shapes.insert({*params[i]->get_output_tensor(0).get_names().begin(), targetInputStaticShapes[i]});
    }
    function->reshape(shapes);
//    functionRefs->reshape(shapes);
}

void SubgraphBaseTest::init_input_shapes(const std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>& shapes) {
    targetStaticShapes = shapes.second;
    if (!shapes.first.empty()) {
        inputDynamicShapes = shapes.first;
    } else {
        OPENVINO_ASSERT(targetStaticShapes.size() == 1, "Incorrect size of targetStaticShapes for static scenario");
        for (const auto& targetStaticShape : targetStaticShapes.front()) {
            inputDynamicShapes.emplace_back(targetStaticShape);
        }
    }
}

void SubgraphBaseTest::init_input_shapes(const std::pair<ov::PartialShape, std::vector<ov::Shape>>& shapes) {
    std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>> tmpShapeObj;
    if (shapes.first.rank() != 0) {
        tmpShapeObj.first = {shapes.first};
    } else {
        tmpShapeObj.first = {};
    }
    tmpShapeObj.second = {shapes.second};
    init_input_shapes(tmpShapeObj);
}

}  // namespace test
}  // namespace ov