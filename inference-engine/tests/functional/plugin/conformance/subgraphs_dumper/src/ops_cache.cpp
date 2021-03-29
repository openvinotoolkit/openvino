// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <ngraph/validation_util.hpp>
#include <ops_cache.hpp>
#include <op_cloner.hpp>
#include "inference_engine.hpp"
#include "common_test_utils/file_utils.hpp"


using namespace SubgraphsDumper;


void OPCache::update_ops_cache(const std::shared_ptr<ngraph::Node> &op,
                               const std::string &source_model) {
    const bool op_found = [&] {
        for (auto &&it : m_ops_cache) {
            if (manager.match_any(it.first, op)) {
                it.second.found_in_models[source_model] += 1;
                return true;
            }
        }
        return false;
    }();
    if (!op_found) {
        // TODO: Extend for subgraphs caching
        const auto& clone_fn = SubgraphsDumper::ClonersMap::cloners.at(op->get_type_info());
        m_ops_cache.emplace_back(clone_fn(op), OPInfo(source_model));
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
    if (!CommonTestUtils::directoryExists(serialization_dir)) {
        CommonTestUtils::createDirectoryRecursive(serialization_dir);
    }
    for (const auto &op : m_ops_cache) {
        try {
            auto rt_info = op.first->get_rt_info();
            std::cout << "Serializing function wrapping op " << op.first << std::endl;
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
            // TODO: Check 'o.get_partial_shape().is_static()' failed at
            //  inference-engine/src/transformations/src/transformations/serialize.cpp:680:
            for (size_t i = 0; i < function->get_output_size(); ++i) {
                if (function->get_output_partial_shape(i).is_dynamic()) {
                    std::cerr << "Can't serialize function related to op: " << std::endl << op.first << std::endl <<
                              "Output shape on port " << i << " is dynamic" << std::endl;
                    output_shape_is_dynamic = true;
                    break;
                }
            }
            if (output_shape_is_dynamic) {
                continue;
            }
            function->validate_nodes_and_infer_types();
            // TODO: How to define element type for multi-output ops
            auto op_el_type = op.first->get_output_element_type(0).get_type_name();
            auto current_op_folder = serialization_dir + CommonTestUtils::FileSeparator +
                                     op.first->get_type_info().name + CommonTestUtils::FileSeparator + op_el_type;
            std::cout << current_op_folder << std::endl;
            if (!CommonTestUtils::directoryExists(current_op_folder)) {
                CommonTestUtils::createDirectoryRecursive(current_op_folder);
            }
            auto op_name = op.first->get_name();
            std::replace(op_name.begin(), op_name.end(), '/', '_');
            std::replace(op_name.begin(), op_name.end(), '\\', '_');
            // TODO: Possible names collision
            auto xml_path = current_op_folder + CommonTestUtils::FileSeparator + op_name + ".xml";
            auto bin_path = current_op_folder + CommonTestUtils::FileSeparator + op_name + ".bin";
            auto mapping = current_op_folder + CommonTestUtils::FileSeparator + op_name + ".csv";
            auto cnn_net = InferenceEngine::CNNNetwork(function);
            cnn_net.serialize(xml_path, bin_path);

            std::string delimiter = ",";
            std::ofstream out(mapping);
            out << "Model" << delimiter << "counters\n";
            // TODO: rethink format of mapping -
            //  how to store both initial source model and usage statistics in one file?
            out << op.second.source_model << delimiter << "source\n";
            for (const auto &m : op.second.found_in_models) {
                out << m.first << delimiter << m.second << "\n";
            }
            out.close();
        } catch (std::exception &e) {
            std::cerr << "Failed to serialize function related to op" << op.first << std::endl
                      << "Exception occurred: " << e.what() << std::endl;
        }
    }
}
