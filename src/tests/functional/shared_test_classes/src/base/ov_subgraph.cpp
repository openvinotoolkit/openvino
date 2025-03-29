// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <setjmp.h>
#include <signal.h>

#include <chrono>
#include <fstream>
#include <thread>

#ifdef _WIN32
#include <process.h>
#endif

#include "openvino/pass/manager.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/convert_precision.hpp"

#include "template/properties.hpp"

#include "common_test_utils/graph_comparator.hpp"


#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "functional_test_utils/crash_handler.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "shared_test_classes/base/utils/calculate_thresholds.hpp"

#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {

std::ostream& operator <<(std::ostream& os, const InputShape& inputShape) {
    auto shape_str = ov::test::utils::vec2str(inputShape.second);
    std::replace(shape_str.begin(), shape_str.end(), ',', '.');
    os << ov::test::utils::partialShape2str({inputShape.first}) << "_" << shape_str;
    return os;
}

void SubgraphBaseTest::run() {
    is_reported = true;
    bool isCurrentTestDisabled = ov::test::utils::current_test_is_disabled();

    ov::test::utils::PassRate::Statuses status = isCurrentTestDisabled ?
         ov::test::utils::PassRate::Statuses::SKIPPED :
         ov::test::utils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status, rel_influence_coef);

    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();

        ASSERT_FALSE(targetStaticShapes.empty() && !function->get_parameters().empty()) << "Target Static Shape is empty!!!";
        std::string errorMessage;
        try {
            compile_model();
            for (const auto& targetStaticShapeVec : targetStaticShapes) {
                generate_inputs(targetStaticShapeVec);
                validate();
            }
            status = ov::test::utils::PassRate::Statuses::PASSED;
        } catch (const std::exception& ex) {
            if (callback_exception != nullptr) {
                // exception will be checked by callback.
                callback_exception(ex);
                return;
            } else {
                status = ov::test::utils::PassRate::Statuses::FAILED;
                errorMessage = ex.what();
            }
        } catch (...) {
            status = ov::test::utils::PassRate::Statuses::FAILED;
            errorMessage = "Unknown failure occurred.";
        }
        summary.updateOPsStats(function, status, rel_influence_coef);
        if (status != ov::test::utils::PassRate::Statuses::PASSED) {
            GTEST_FATAL_FAILURE_(errorMessage.c_str());
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        OPENVINO_THROW("Crash happens");
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
        OPENVINO_THROW("Crash happens");
    }
}

void SubgraphBaseTest::serialize() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::string output_name = ov::test::utils::generateTestFilePrefix();

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

    ov::test::utils::removeIRFiles(out_xml_path, out_bin_path);
}

void SubgraphBaseTest::query_model() {
    bool isCurrentTestDisabled = ov::test::utils::current_test_is_disabled();

    ov::test::utils::PassRate::Statuses status = isCurrentTestDisabled ?
         ov::test::utils::PassRate::Statuses::SKIPPED :
         ov::test::utils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status, rel_influence_coef);

    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        std::string errorMessage;
        try {
            auto queryNetworkResult = core->query_model(function, targetDevice);
            std::set<std::string> expected;
            for (auto&& node : function->get_ops()) {
                expected.insert(node->get_friendly_name());
            }

            std::set<std::string> actual;
            for (auto&& res : queryNetworkResult) {
                actual.insert(res.first);
            }
            if (expected != actual) {
                OPENVINO_THROW("Expected and actual are different");
            }
            status = ov::test::utils::PassRate::Statuses::PASSED;
        } catch (const std::exception& ex) {
            status = ov::test::utils::PassRate::Statuses::FAILED;
            errorMessage = ex.what();
        } catch (...) {
            status = ov::test::utils::PassRate::Statuses::FAILED;
            errorMessage = "Unknown failure occurred.";
        }
        summary.updateOPsStats(function, status, rel_influence_coef);
        if (status != ov::test::utils::PassRate::Statuses::PASSED) {
            GTEST_FATAL_FAILURE_(errorMessage.c_str());
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        OPENVINO_THROW("Crash happens");
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
        OPENVINO_THROW("Crash happens");
    }
}

void SubgraphBaseTest::import_export() {
    bool isCurrentTestDisabled = ov::test::utils::current_test_is_disabled();

    ov::test::utils::PassRate::Statuses status = isCurrentTestDisabled ?
         ov::test::utils::PassRate::Statuses::SKIPPED :
         ov::test::utils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status, rel_influence_coef);

    if (isCurrentTestDisabled)
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;

    // in case of crash jump will be made and work will be continued
    auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler->StartTimer();
        std::string errorMessage;
        try {
            compile_model();

            std::stringstream strm;
            compiledModel.export_model(strm);
            ov::CompiledModel importedModel = core->import_model(strm, targetDevice, configuration);
            const auto importedFunction = importedModel.get_runtime_model()->clone();
            const auto runtimeModel = compiledModel.get_runtime_model()->clone();
            compare_models_param_res(importedFunction, runtimeModel);
            status = ov::test::utils::PassRate::Statuses::PASSED;
        } catch (const std::exception& ex) {
            status = ov::test::utils::PassRate::Statuses::FAILED;
            errorMessage = ex.what();
        } catch (...) {
            status = ov::test::utils::PassRate::Statuses::FAILED;
            errorMessage = "Unknown failure occurred.";
        }
        summary.updateOPsStats(function, status, rel_influence_coef);
        if (status != ov::test::utils::PassRate::Statuses::PASSED) {
            GTEST_FATAL_FAILURE_(errorMessage.c_str());
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        OPENVINO_THROW("Crash happens");
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED, rel_influence_coef);
        OPENVINO_THROW("Crash happens");
    }
}

void SubgraphBaseTest::compare(const std::vector<ov::Tensor>& expected,
                               const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(expected.size(), function->get_results().size());
    init_thresholds();
    auto compareMap = utils::getCompareMap();
    const auto& results = function->get_results();
    for (size_t j = 0; j < results.size(); j++) {
        const auto result = results[j];
        for (size_t i = 0; i < result->get_input_size(); ++i) {
            std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
            auto it = compareMap.find(inputNode->get_type_info());
            ASSERT_NE(it, compareMap.end());
            it->second(inputNode, i, inference_precision,
                       expected[j], actual[j],
                       abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
        }
    }
}

void SubgraphBaseTest::configure_model() {
    // configure input precision
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (inType != ov::element::Type_t::dynamic) {
                p.input(i).tensor().set_element_type(inType);
            }
        }
    }

    // configure output precision
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType != ov::element::Type_t::dynamic) {
                p.output(i).tensor().set_element_type(outType);
            }
        }
    }
    function = p.build();
}

void SubgraphBaseTest::compile_model() {
    if (is_report_stages) {
        std::cout << "[ PLUGIN      ] `SubgraphBaseTest::compile_model()` is started" << std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    configure_model();
    core_configuration(this);
    compiledModel = core->compile_model(function, targetDevice, configuration);
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ PLUGIN      ] `SubgraphBaseTest::compile_model()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    try {
        inference_precision = compiledModel.get_property(ov::hint::inference_precision);
    } catch (std::exception& e) {
        std::cout << "[ WARNING ] Impossible to get Inference Precision with exception: " << e.what() << std::endl;
    }
}

void SubgraphBaseTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(function);

    auto itTargetShape = targetInputStaticShapes.begin();
    for (const auto &param : function->get_parameters()) {
        std::shared_ptr<ov::Node> inputNode = param;
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto &node : param->get_output_target_inputs(i)) {
                std::shared_ptr<ov::Node> nodePtr = node.get_node()->shared_from_this();
                for (size_t port = 0; port < nodePtr->get_input_size(); ++port) {
                    if (nodePtr->get_input_node_ptr(port)->shared_from_this() == inputNode->shared_from_this()) {
                        inputs.insert({param, modelRange.generate_input(nodePtr, port, *itTargetShape)});
                        break;
                    }
                }
            }
        }
        itTargetShape++;
    }
}

void SubgraphBaseTest::infer() {
    inferRequest = compiledModel.create_infer_request();
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }
    inferRequest.infer();
}

void SubgraphBaseTest::update_ref_model() {
    if (functionRefs == nullptr) {
        functionRefs = function->clone();
    }
    using InputsMap = std::map<std::shared_ptr<ov::Node>, ov::Tensor>;

    if (!convert_precisions.empty()) {
        pass::Manager manager;
        manager.register_pass<ov::pass::ConvertPrecision>(convert_precisions, type_to_fuse_map{}, false, false);
        manager.run_passes(functionRefs);
        functionRefs->validate_nodes_and_infer_types();
    }

    ov::preprocess::PrePostProcessor p(functionRefs);
    const auto& inputNodes = functionRefs->inputs();
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
    const auto& outputs = functionRefs->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outType != ElementType::dynamic && outType != outputs[i].get_element_type()) {
            p.output(i).tensor().set_element_type(outType);
        }
    }
    functionRefs = p.build();
}

void SubgraphBaseTest::match_parameters(const ov::ParameterVector& params, const ov::ParameterVector& ref_params) {
    matched_parameters.clear();
    size_t param_size = params.size(), ref_param_size = ref_params.size();
    if (params.size() < ref_params.size()) {
        throw std::runtime_error("Incompatible parameters in original and reference model!");
    }
    if (params.size() == ref_params.size()) {
        for (size_t in_idx = 0; in_idx < params.size(); ++in_idx) {
            matched_parameters.insert({ ref_params[in_idx], params[in_idx] });
        }
    } else {
        auto it = params.begin();
        auto it_ref = ref_params.begin();
        while (it_ref != ref_params.end() && it != params.end()) {
            bool is_match_in = true;
            if ((*it_ref)->get_output_partial_shape(0).is_static()) {
                if (inputs.at(*it).get_shape() != (*it_ref)->get_output_shape(0)) {
                    is_match_in = false;
                }
            } else if ((*it)->get_output_partial_shape(0) != (*it_ref)->get_output_partial_shape(0)) {
                is_match_in = false;
            }
            if ((*it)->get_output_element_type(0) != ((*it_ref)->get_output_element_type(0))) {
                is_match_in = false;
            }
            if (is_match_in) {
                matched_parameters.insert({ *it_ref, *it });
                ++it_ref;
            }
            ++it;
        }
        if (matched_parameters.size() != ref_params.size()) {
            throw std::runtime_error("Incompatible parameters in original and reference model!");
        }
    }
}

std::vector<ov::Tensor> SubgraphBaseTest::calculate_refs() {
    if (is_report_stages) {
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    update_ref_model();
    match_parameters(function->get_parameters(), functionRefs->get_parameters());

    std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs_ref;
    for (const auto& param : functionRefs->get_parameters()) {
        inputs_ref[param] = inputs.at(matched_parameters[param]);
    }

    auto outputs = ov::test::utils::infer_on_template(functionRefs, inputs_ref);

    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    return outputs;
}

std::vector<ov::Tensor> SubgraphBaseTest::get_plugin_outputs() {
    if (is_report_stages) {
        std::cout << "[ PLUGIN      ] `SubgraphBaseTest::get_plugin_outputs()` is started"<< std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    infer();
    auto outputs = std::vector<ov::Tensor>{};
    for (const auto& output : function->outputs()) {
        outputs.push_back(inferRequest.get_tensor(output));
    }
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ PLUGIN      ] `SubgraphBaseTest::get_plugin_outputs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
    return outputs;
}

void SubgraphBaseTest::validate() {
    std::vector<ov::Tensor> expectedOutputs, actualOutputs;
    std::exception_ptr expected_outputs_error, actual_output_error;

#ifndef NDEBUG
    actualOutputs = get_plugin_outputs();
    expectedOutputs = calculate_refs();
#else
    std::thread t_device([this, &actualOutputs, &actual_output_error] {
        // The try ... catch block is required to handle exceptions during output calculations and report as test fail.
        // If exception is not caught then application would be terminated with crash. (CVS-133676)
        try {
            actualOutputs = get_plugin_outputs();
        } catch (...) {
            actual_output_error = std::current_exception();
        }
    });
    std::thread t_ref([this, &expectedOutputs, &expected_outputs_error] {
        try {
            expectedOutputs = calculate_refs();
        } catch (...) {
            expected_outputs_error = std::current_exception();
        }
    });
    t_device.join();
    t_ref.join();

    if (actual_output_error) {
        std::rethrow_exception(actual_output_error);
    }
    if (expected_outputs_error) {
        std::rethrow_exception(expected_outputs_error);
    }
#endif

    if (expectedOutputs.empty()) {
        return;
    }

    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
        << "TEMPLATE plugin has " << expectedOutputs.size() << " outputs, while " << targetDevice << " " << actualOutputs.size();
    if (is_report_stages) {
        std::cout << "[ COMPARATION ] `ov_tensor_utils.hpp::compare()` is started"<< std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    compare(expectedOutputs, actualOutputs);
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ COMPARATION ] `ov_tensor_utils.hpp::compare()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
    }
}

void SubgraphBaseTest::init_thresholds() {
    double max_abs_threshold = 0.f, max_rel_threshold = 0.f;
    std::tie(max_abs_threshold, max_rel_threshold) = ov::test::utils::calculate_thresholds_by_model(function, functionRefs, inference_precision);
    if (abs_threshold == disable_threshold) {
        abs_threshold = max_abs_threshold;
    }
    if (rel_threshold == disable_threshold) {
        rel_threshold = max_rel_threshold;
    }
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
            dynShape = shape.second.front();
        }
        inputDynamicShapes.push_back(dynShape);
        for (size_t i = 0; i < targetStaticShapeSize; ++i) {
            targetStaticShapes[i].push_back(i < shape.second.size() ? shape.second.at(i) : shape.second.back());
        }
    }
}

void SubgraphBaseTest::compare_nodes(const std::shared_ptr<ov::Node>& node1, const std::shared_ptr<ov::Node>& node2, std::ostream& err_log) {
    // compare inputs size, element_type and constant's values
    if (node1->get_input_size() != node2->get_input_size()) {
        err_log << "Number of inputs is different: " << to_str(node1->get_input_size()) << " for "
                << node1->get_friendly_name() << " and " << to_str(node2->get_input_size()) << " for " + node2->get_friendly_name();
    }

    for (size_t i = 0; i < node1->get_input_size(); ++i) {
        if (node1->input(i).get_element_type() != node2->input(i).get_element_type()) {
            err_log << "Different element type detected\n"
                    << node1->get_friendly_name() << " Input(" << i << ") " << node1->input(i).get_element_type() << " and "
                    << node2->get_friendly_name() << " Input(" << i << ") " << node2->input(i).get_element_type() << std::endl;
        }

        const auto const_in_1 = ov::as_type_ptr<ov::op::v0::Constant>(node1->get_input_node_shared_ptr(i));
        const auto const_in_2 = ov::as_type_ptr<ov::op::v0::Constant>(node2->get_input_node_shared_ptr(i));
        const auto equal_value = ::attributes::detail::equal::Equal<std::shared_ptr<ov::op::v0::Constant>>::equal_value;
        if (const_in_1 && const_in_2 && !equal_value(const_in_1, const_in_2)) {
            err_log << "Different Constant values detected\n"
                    << const_in_1->get_friendly_name() << " & " << const_in_2->get_friendly_name() << "\n"
                    << node1->description() << " Input(" << i << ") and " << node2->description() << " Input(" << i
                    << ")" << std::endl;
        }
    }

    // compare outputs size, shape, element_type
    if (node1->get_output_size() != node2->get_output_size()) {
        err_log << "Number of outputs is different: " << to_str(node1->get_output_size()) << " for "
                << node1->get_friendly_name() << " and " << to_str(node2->get_output_size()) << " for " << node2->get_friendly_name();
    }

    for (int i = 0; i < node1->get_output_size(); ++i) {
        if (!node1->output(i).get_partial_shape().same_scheme(node2->output(i).get_partial_shape())) {
            err_log << "Different shape detected\n"
                    << node1->get_friendly_name() << " Output(" << i << ") " << node1->output(i).get_partial_shape() << " and "
                    << node2->get_friendly_name() << " Output(" << i << ") " << node2->output(i).get_partial_shape() << std::endl;
        }

        if (node1->output(i).get_element_type() != node2->output(i).get_element_type()) {
            err_log << "Different element type detected\n"
                    << node1->get_friendly_name() << " Input(" << i << ") " << node1->output(i).get_element_type() << " and "
                    << node2->get_friendly_name() << " Input(" << i << ") " << node2->output(i).get_element_type() << std::endl;
        }
    }
}

void SubgraphBaseTest::compare_models_param_res(const std::shared_ptr<ov::Model>& f, const std::shared_ptr<ov::Model>& f_ref) {
    if (is_report_stages) {
        std::cout << "[ COMPARATION ] `compare_models_param_res(f, ref_f)` is started"<< std::endl;
    }
    auto start_time = std::chrono::system_clock::now();

    std::queue<std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>> queue;
    auto parameters = f->get_parameters();
    auto ref_parameters = f_ref->get_parameters();
    match_parameters(parameters, ref_parameters);

    for (auto& matched : matched_parameters) {
        queue.push({matched.first, matched.second});
    }

    auto f_results = f->get_results();
    auto f_ref_results = f_ref->get_results();
    if (f_results.size() != f_ref_results.size()) {
        throw std::runtime_error("Number of results is different: " + to_str(f_results.size()) + " and " +
                                to_str(f_ref_results.size()));
    }

    for (size_t i = 0; i < f_results.size(); ++i) {
        queue.push({f_results[i], f_ref_results[i]});
    }

    std::stringstream errors;
    while (!queue.empty()) {
        auto nodes = queue.front();
        compare_nodes(nodes.first, nodes.second, errors);
        queue.pop();
    }
    if (is_report_stages) {
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        std::cout << "[ COMPARATION ] `compare_models_param_res(f, ref_f)` is finished. Duration is " << duration.count() << "s" << std::endl;
    }

    if (!errors.str().empty()) {
        throw std::runtime_error(errors.str());
    }
}

}  // namespace test
}  // namespace ov
