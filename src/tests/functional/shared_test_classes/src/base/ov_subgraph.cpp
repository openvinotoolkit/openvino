// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <fstream>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_precision.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

#ifdef _WIN32
#include <process.h>
#endif

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/serialize.hpp"

#include "graph_comparator.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

std::ostream& operator <<(std::ostream& os, const InputShape& inputShape) {
    os << CommonTestUtils::partialShape2str({inputShape.first}) << "_" << CommonTestUtils::vec2str(inputShape.second);
    return os;
}

void SubgraphBaseTest::run() {
    auto crashHandler = [](int errCode) {
        auto& s = LayerTestsUtils::Summary::getInstance();
        s.saveReport();
        std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
        std::abort();
    };
    signal(SIGSEGV, crashHandler);

    LayerTestsUtils::PassRate::Statuses status = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()
                                                     ? LayerTestsUtils::PassRate::Statuses::SKIPPED
                                                     : LayerTestsUtils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    ASSERT_FALSE(targetStaticShapes.empty()) << "Target Static Shape is empty!!!";
    std::string errorMessage;
    try {
        compile_model();
        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                if (!inputDynamicShapes.empty()) {
                    // resize ngraph function according new target shape
                    // Note: output shapes of some nodes depend on the input data
                    // so for some tests we need to override this function and replace parameter with constant node to get correct output shapes
                    init_ref_function(functionRefs, targetStaticShapeVec);
                }
                generate_inputs(targetStaticShapeVec);
            } catch (const std::exception& ex) {
                throw std::runtime_error("Incorrect target static shape: " +
                                         CommonTestUtils::vec2str(targetStaticShapeVec) + " " + ex.what());
            }
            infer();
            validate();
        }
        status = LayerTestsUtils::PassRate::Statuses::PASSED;
    } catch (const std::exception& ex) {
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
    std::tie(success, message) = compare_functions(result,
                                                   function,
                                                   false,
                                                   false,
                                                   false,
                                                   true,   // precision
                                                   true);  // attributes

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

void SubgraphBaseTest::compare(const std::vector<ov::runtime::Tensor>& expected,
                               const std::vector<ov::runtime::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

void SubgraphBaseTest::configure_model() {
    // configure input precision
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (inType != ov::element::Type_t::undefined) {
                p.input(i).tensor().set_element_type(inType);
            }
        }
    }

    // configure output precision
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType != ov::element::Type_t::undefined) {
                p.output(i).tensor().set_element_type(outType);
            }
        }
    }
    function = p.build();
}

void SubgraphBaseTest::compile_model() {
    configure_model();
    if (functionRefs == nullptr) {
        functionRefs = ov::clone_model(*function);
    }
    executableNetwork = core->compile_model(function, targetDevice, configuration);
}

void SubgraphBaseTest::init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) {
    ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
}

void SubgraphBaseTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::runtime::Tensor tensor;
        if (funcInput.get_element_type().is_real()) {
            tensor = ov::test::utils::create_and_fill_tensor(
                funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
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
    using InputsMap = std::map<std::shared_ptr<ov::Node>, ov::runtime::Tensor>;

    auto functionToProcess = ov::clone_model(*functionRefs);
    //TODO: remove this conversions as soon as function interpreter fully support bf16 and f16
    static const precisions_array precisions = {
            { ngraph::element::bf16, ngraph::element::f32 },
            { ngraph::element::f16, ngraph::element::f32}
    };

    pass::Manager manager;
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions);
    manager.run_passes(functionToProcess);
    functionToProcess->validate_nodes_and_infer_types();

    ov::preprocess::PrePostProcessor p(functionToProcess);
    const auto& inputNodes = functionToProcess->inputs();
    for (size_t i = 0; i < inputNodes.size(); ++i) {
        auto itr = std::find_if(inputs.begin(), inputs.end(),
                                [&](const InputsMap::value_type& item) {
                                    return item.first->get_friendly_name() == inputNodes[i].get_node_shared_ptr()->get_friendly_name();
                                });
        if (itr != inputs.end()) {
            auto elementType = itr->second.get_element_type();
            if (inputNodes[i].get_element_type() != elementType) {
                p.input(i).tensor().set_element_type(elementType);
            }
        } else {
            std::stringstream errMsg;
            errMsg << "Couldn't find input with name " << inputNodes[i].get_node_shared_ptr()->get_friendly_name();
            errMsg << " in the inputs map";
            throw std::runtime_error(errMsg.str());
        }
    }

    const auto& outputs = functionToProcess->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outType != ElementType::undefined && outType != outputs[i].get_element_type()) {
            p.output(i).tensor().set_element_type(outType);
        }
    }

    functionToProcess = p.build();

    return ngraph::helpers::interpretFunction(functionToProcess, inputs);
}

std::vector<ov::runtime::Tensor> SubgraphBaseTest::get_plugin_outputs() {
    auto outputs = std::vector<ov::runtime::Tensor>{};
    for (const auto& output : function->outputs()) {
        outputs.push_back(inferRequest.get_tensor(output));
    }
    return outputs;
}

void SubgraphBaseTest::validate() {
    auto expectedOutputs = calculate_refs();
    const auto& actualOutputs = get_plugin_outputs();

    if (expectedOutputs.empty()) {
        return;
    }

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    compare(expectedOutputs, actualOutputs);
}

void SubgraphBaseTest::init_input_shapes(const std::vector<InputShape>& shapes) {
    size_t targetStaticShapeSize = shapes.front().second.size();
    for (size_t i = 1; i < shapes.size(); ++i) {
        if (targetStaticShapeSize < shapes[i].second.size()) {
            targetStaticShapeSize = shapes[i].second.size();
        }
    }
    targetStaticShapes.resize(targetStaticShapeSize);

    for (const auto& shape : shapes) {
        auto dynShape = shape.first;
        if (dynShape.rank() == 0) {
            ASSERT_EQ(targetStaticShapeSize, 1) << "Incorrect number of static shapes for static case";
            dynShape = shape.second.front();
        }
        inputDynamicShapes.push_back(dynShape);
        for (size_t i = 0; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(i < shape.second.size() ? shape.second.at(i) : shape.second.back());
        }
    }
}
}  // namespace test
}  // namespace ov
