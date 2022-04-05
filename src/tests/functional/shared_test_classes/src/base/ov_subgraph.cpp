// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <fstream>
#include "transformations/convert_precision.hpp"

#ifdef _WIN32
#include <process.h>
#endif

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/serialize.hpp"

#include "common_test_utils/graph_comparator.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/crash_handler.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"

#include <setjmp.h>

namespace ov {
namespace test {

std::ostream& operator <<(std::ostream& os, const InputShape& inputShape) {
    os << CommonTestUtils::partialShape2str({inputShape.first}) << "_" << CommonTestUtils::vec2str(inputShape.second);
    return os;
}

void SubgraphBaseTest::run() {
    bool isCurrentTestDisabled = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled();

    LayerTestsUtils::PassRate::Statuses status = isCurrentTestDisabled ?
        LayerTestsUtils::PassRate::Statuses::SKIPPED :
        LayerTestsUtils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);

    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(CommonTestUtils::env);
#else
    jmpRes = sigsetjmp(CommonTestUtils::env, 1);
#endif
    if (jmpRes == CommonTestUtils::JMP_STATUS::ok) {
        crashHandler->StartTimer();

        ASSERT_FALSE(targetStaticShapes.empty() && !function->get_parameters().empty()) << "Target Static Shape is empty!!!";
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
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::anyError) {
        IE_THROW() << "Crash happens";
    } else if (jmpRes == CommonTestUtils::JMP_STATUS::alarmErr) {
        summary.updateOPsStats(function, LayerTestsUtils::PassRate::Statuses::HANGED);
        IE_THROW() << "Crash happens";
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

void SubgraphBaseTest::compare(const std::vector<ov::Tensor>& expected,
                               const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(expected.size(), function->get_results().size());
    auto compareMap = utils::getCompareMap();
    const auto& results = function->get_results();
    for (size_t j = 0; j < results.size(); j++) {
        const auto result = results[j];
        for (size_t i = 0; i < result->get_input_size(); ++i) {
            std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
            if (std::dynamic_pointer_cast<ov::op::v0::Convert>(inputNode)) {
                std::shared_ptr<ov::Node> nextNodePtr = inputNode->get_input_node_shared_ptr(0);
                if (!ngraph::is_type<ov::op::v0::Result>(nextNodePtr)) {
                    inputNode = nextNodePtr;
                }
            }
            auto it = compareMap.find(inputNode->get_type_info());
            it->second(inputNode, i, expected[j], actual[j], abs_threshold, rel_threshold);
        }
    }
}

void SubgraphBaseTest::init_inputs_and_outputs() {
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        if (inType.size() == 1) {
            for (size_t i = 1; i < params.size(); ++i) {
                inType.push_back(inType.front());
            }
        }
    }
    {
        auto& results = function->get_results();
        if (outType.size() == 1) {
            for (size_t i = 1; i < results.size(); ++i) {
                outType.push_back(outType.front());
            }
        }
    }
}

void SubgraphBaseTest::configure_model() {
    // align number of inputs outputs attributes to tensor inputs outputs
    init_inputs_and_outputs();

    // configure input precision
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (inType[i] != ov::element::Type_t::undefined) {
                p.input(i).tensor().set_element_type(inType[i]);
            }
        }
    }

    // configure output precision
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType[i] != ov::element::Type_t::undefined) {
                p.output(i).tensor().set_element_type(outType[i]);
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
    compiledModel = core->compile_model(function, targetDevice, configuration);
}

void SubgraphBaseTest::init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) {
    ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
}

void SubgraphBaseTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto inputMap = utils::getInputMap();
    auto itTargetShape = targetInputStaticShapes.begin();
    for (const auto &param : function->get_parameters()) {
        std::shared_ptr<ov::Node> inputNode = param;
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto &node : param->get_output_target_inputs(i)) {
                std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ov::op::v0::Convert>(nodePtr)) {
                    std::shared_ptr<ov::Node> nextNodePtr = nodePtr->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
                    if (!ngraph::is_type<ov::op::v0::Result>(nextNodePtr)) {
                        inputNode = nodePtr;
                        nodePtr = nextNodePtr;
                    }
                }
                auto it = inputMap.find(nodePtr->get_type_info());
                for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                    if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                        inputs.insert({param, it->second(nodePtr, port, param->get_element_type(), *itTargetShape++)});
                        break;
                    }
                }
            }
        }
    }
}

void SubgraphBaseTest::infer() {
    inferRequest = compiledModel.create_infer_request();
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }
    inferRequest.infer();
}

std::vector<ov::Tensor> SubgraphBaseTest::calculate_refs() {
    using InputsMap = std::map<std::shared_ptr<ov::Node>, ov::Tensor>;

    auto functionToProcess = ov::clone_model(*functionRefs);
    //TODO: remove this conversions as soon as function interpreter fully support bf16 and f16
    precisions_array precisions = {
            { ngraph::element::bf16, ngraph::element::f32 }
    };
    auto convert_added = false;
    for (const auto &param : function->get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto &node : param->get_output_target_inputs(i)) {
                std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ov::op::v0::Convert>(nodePtr)) {
                    convert_added = true;
                    break;
                }
            }
        }
    }
    if (!convert_added) {
        precisions.push_back({ ngraph::element::f16, ngraph::element::f32});
    }
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
        if (outType[i] != ElementType::undefined && outType[i] != outputs[i].get_element_type()) {
            p.output(i).tensor().set_element_type(outType[i]);
        }
    }

    functionToProcess = p.build();

    return ngraph::helpers::interpretFunction(functionToProcess, inputs);
}

std::vector<ov::Tensor> SubgraphBaseTest::get_plugin_outputs() {
    auto outputs = std::vector<ov::Tensor>{};
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
    if (shapes.empty()) {
        targetStaticShapes = {{}};
        return;
    }
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
