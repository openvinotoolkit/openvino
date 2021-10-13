// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "openvino/pass/serialize.hpp"

#include "graph_comparator.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

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
                    resize_function(targetStaticShapeVec);
                }
                generate_inputs(targetStaticShapeVec);
                infer();
                validate();
            } catch (const std::exception &ex) {
                OPENVINO_ASSERT("Incorrect target static shape: ", ex.what());
            }
        }
        status = LayerTestsUtils::PassRate::Statuses::PASSED;
    } catch (const std::exception &ex) {
        status = LayerTestsUtils::PassRate::Statuses::FAILED;
        errorMessage = ex.what();
    } catch (...) {
        status = LayerTestsUtils::PassRate::Statuses::FAILED;
        errorMessage = "Unknown failure occurred.";
    }
    summary.updateOPsStats(function, status);
    if (status != LayerTestsUtils::PassRate::Statuses::PASSED) {
        GTEST_FATAL_FAILURE_(errorMessage.c_str());
    }
}

void SubgraphBaseTest::serialize() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::string output_name = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + GetTimestamp();

    std::string out_xml_path = output_name + ".xml";
    std::string out_bin_path = output_name + ".bin";

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(out_xml_path, out_bin_path);
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

void SubgraphBaseTest::compare(const std::vector<ov::runtime::Tensor> &expected,
                               const std::vector<ov::runtime::Tensor> &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

void SubgraphBaseTest::configure_model() {
    // configure input precision
    {
        auto params = function->get_parameters();
        for (auto& param : params) {
            if (inType != ov::element::Type_t::undefined) {
                param->get_output_tensor(0).set_element_type(inType);
            }
        }
    }

    // configure output precision
    {
        auto results = function->get_results();
        for (auto& result : results) {
            if (outType != ov::element::Type_t::undefined) {
                result->get_output_tensor(0).set_element_type(outType);
            }
        }
    }
}

void SubgraphBaseTest::compile_model() {
    configure_model();
    if (functionRefs == nullptr) {
        functionRefs = ov::clone_function(*function);
    }
    executableNetwork = core->compile_model(function, targetDevice, configuration);
}

void SubgraphBaseTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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

std::vector<ov::runtime::Tensor> SubgraphBaseTest::calculate_refs() {
    // nGraph interpreter does not support f16/bf16
    ngraph::pass::ConvertPrecision<element::Type_t::f16, element::Type_t::f32>().run_on_function(functionRefs);
    ngraph::pass::ConvertPrecision<element::Type_t::bf16, element::Type_t::f32>().run_on_function(functionRefs);

    functionRefs->validate_nodes_and_infer_types();

    return ngraph::helpers::interpreterFunction(functionRefs, inputs);
}

std::vector<ov::runtime::Tensor> SubgraphBaseTest::get_plugin_outputs() {
    auto outputs = std::vector<ov::runtime::Tensor>{};
    for (const auto& output : executableNetwork.outputs()) {
        const auto& name = *output.get_tensor().get_names().begin();
        outputs.push_back(inferRequest.get_tensor(name));
    }
    return outputs;
}

void SubgraphBaseTest::validate() {
    auto expectedOutputs = calculate_refs();
    const auto& actualOutputs = get_plugin_outputs();

    if (expectedOutputs.empty()) {
        return;
    }

    OPENVINO_ASSERT(actualOutputs.size() == expectedOutputs.size(),
                    "nGraph interpreter has ", expectedOutputs.size(), " outputs, while IE ", actualOutputs.size());

    compare(expectedOutputs, actualOutputs);
}

void SubgraphBaseTest::resize_function(const std::vector<ov::Shape>& targetInputStaticShapes) {
    auto params = function->get_parameters();
    std::map<std::string, ov::PartialShape> shapes;
    ASSERT_LE(params.size(), targetInputStaticShapes.size());
    for (size_t i = 0; i < params.size(); i++) {
        shapes.insert({*params[i]->get_output_tensor(0).get_names().begin(), targetInputStaticShapes[i]});
    }
    function->reshape(shapes);
    functionRefs->reshape(shapes);
}

void SubgraphBaseTest::init_input_shapes(const InputShapes& shapes) {
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

void SubgraphBaseTest::init_input_shapes(const InputShape& shapes) {
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