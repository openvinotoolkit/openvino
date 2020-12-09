// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <sstream>
// TODO: c++17 code
//#include <filesystem>
#include <ngraph/validation_util.hpp>
#include <ops_cache.hpp>
#include "inference_engine.hpp"
#include "common_test_utils/file_utils.hpp"


using namespace SubgraphsDumper;

// TODO: c++17 code
//namespace fs = std::filesystem;

void OPCache::update_ops_cache(const std::shared_ptr<ngraph::Node> &op,
                               const std::string &source_model) {
    bool op_found = false;
    if (!m_ops_cache.empty()) {
        for (auto &it : m_ops_cache) {
            // TODO: Extend for subgraphs comparison if num_neighbors_to_cache != 0
            if (ngraph::ops_identical(it.first, op)) {
                op_found = true;
                it.second += 1;
                break;
            }
        }
    }
    if (!op_found) {
        // TODO: Extend for subgraphs caching if num_neighbors_to_cache != 0
        ngraph::OutputVector op_inputs;
        bool input_shapes_static = true;
        for (const auto &input : op->inputs()) {
            if (ngraph::op::is_constant(input.get_source_output().get_node_shared_ptr())) {
                op_inputs.push_back(input.get_source_output().get_node_shared_ptr()->clone_with_new_inputs({}));
            } else {
                if (input.get_source_output().get_tensor().get_partial_shape().is_dynamic()) {
                    // TODO: Handle dynamic shapes properly
                    input_shapes_static = false;
                    break;
                }
                op_inputs.push_back(std::make_shared<ngraph::op::Parameter>(input.get_element_type(),
                                                                            input.get_source_output().get_shape()));
            }
        }
        if (input_shapes_static) {
            auto op_clone = op->clone_with_new_inputs(op_inputs);
            if (!source_model.empty()) {
// TODO: c++17 code
//                auto source_model_name = fs::path(source_model).filename();
                auto source_model_name = source_model;
                ngraph::Node::RTMap &rt_info = op_clone->get_rt_info();
                // TODO: Store both list of model where OP appears and the model from which it cached
                rt_info["source_model"] = std::make_shared<ngraph::VariantWrapper<std::string>>(source_model_name);
            }
            m_ops_cache.push_back({op_clone, 1});
        }
    }
}

void OPCache::update_ops_cache(const std::shared_ptr<ngraph::Function> &func, const std::string &source_model) {
    func->validate_nodes_and_infer_types();
    for (const auto &op : func->get_ordered_ops()) {
        if (ngraph::is_type<ngraph::op::Parameter>(op) ||
            ngraph::is_type<ngraph::op::Constant>(op) ||
            ngraph::is_type<ngraph::op::Result>(op)) {
            continue;
        }
        update_ops_cache(op, source_model);
    }
}

void OPCache::serialize_cached_ops(const std::string &serialization_dir) {
// TODO: c++17 code
//    if (!fs::is_directory(serialization_dir)) {
//        fs::create_directories(serialization_dir);
//    }
    if (!CommonTestUtils::directoryExists(serialization_dir)) {
        CommonTestUtils::createDirectoryRecursive(serialization_dir);
    }
    for (const auto &op : m_ops_cache) {
        try {
            auto rt_info = op.first->get_rt_info();
            std::cout << "Serializing function wrapping op " << op.first->get_type_name() << std::endl;
            if (rt_info.find("source_model") != rt_info.end()) {
                auto val = rt_info["source_model"];
                auto source_model = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(val);
                if (source_model != nullptr) {
                    std::cout << " Taken from model: " << source_model->get() << std::endl;
                }
            }

            ngraph::ParameterVector params;
            for (size_t i = 0; i < op.first->get_input_size(); ++i) {
                if (ngraph::op::is_parameter(op.first->get_input_node_ptr(i))) {
                    auto param = std::dynamic_pointer_cast<ngraph::op::Parameter>(
                            op.first->get_input_node_shared_ptr(i));
                    params.push_back(param);
                }
            }
            ngraph::ResultVector results;
            for (auto &out : op.first->outputs()) {
                results.push_back(std::make_shared<ngraph::op::Result>(out));
            }
            auto function = std::make_shared<ngraph::Function>(results, params);
            bool output_shape_is_dynamic = false;
            for (size_t i = 0; i < function->get_output_size(); ++i) {
                if (function->get_output_partial_shape(i).is_dynamic()) {
                    std::cerr << "Can't serialize function related to op: " << std::endl << op.first << std::endl <<
                              "Output shape on port " << i << " is dynamic" << std::endl;
                    output_shape_is_dynamic = true;
                    break;
                }
            }
//            if (output_shape_is_dynamic) {
//                continue;
//            }
            function->validate_nodes_and_infer_types();
            // TODO: How to define element type for multi-output ops
            auto op_el_type = op.first->get_output_element_type(0).get_type_name();
// TODO: c++17 code
//            auto current_op_folder = fs::path(serialization_dir) / op.first->get_type_info().name / op_el_type;
//            if (!fs::is_directory(current_op_folder)) {
//                fs::create_directories(current_op_folder);
//            }
            auto current_op_folder =
                    serialization_dir + CommonTestUtils::FileSeparator + op.first->get_type_info().name + CommonTestUtils::FileSeparator + op_el_type;
            std::cout << current_op_folder << std::endl;
// TODO: c++17 code
//            if (!fs::is_directory(current_op_folder)) {
//                fs::create_directories(current_op_folder);
//            }
            if (!CommonTestUtils::directoryExists(current_op_folder)) {
                CommonTestUtils::createDirectoryRecursive(current_op_folder);
            }
            auto op_name = op.first->get_name();
            std::replace(op_name.begin(), op_name.end(), '/', '_');
            std::replace(op_name.begin(), op_name.end(), '\\', '_');
            // TODO: Possible names collision
            auto xml_path = current_op_folder + CommonTestUtils::FileSeparator + (op_name + std::string(".xml"));
            auto bin_path = current_op_folder + CommonTestUtils::FileSeparator + (op_name + std::string(".bin"));
            auto cnn_net = InferenceEngine::CNNNetwork(function);
            // TODO: Visitor API is not supported in v0::TensorIterator TensorIterator_623065
            //  (Parameter_623062[0]:f16{1,16,512}, Constant_623063[0]:f16{1,256}, Constant_623064[0]:f16{1,256}) ->
            //  (f16{1,16,256})
            // TODO: Runtime Info doesn't serialized
            cnn_net.serialize(xml_path, bin_path);
        } catch (std::exception &e) {
            std::cerr << "Failed to serialize function related to op" << op.first << std::endl
                      << "Exception occurred: " << e.what() << std::endl;
        }
    }
}
