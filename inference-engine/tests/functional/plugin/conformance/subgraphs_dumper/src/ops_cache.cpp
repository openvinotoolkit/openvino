// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <ngraph/validation_util.hpp>
#include <ops_cache.hpp>
#include <op_cloner.hpp>
#include "inference_engine.hpp"
#include "common_test_utils/file_utils.hpp"
#include "pugixml.hpp"


using namespace SubgraphsDumper;

void OPCache::update_ops_cache(const std::shared_ptr<ngraph::Node> &op,
                               const std::string &source_model) {
    const bool op_found = [&] {
        for (auto &&it : m_ops_cache) {
            if (manager.match_any(it.first, op, it.second)) {
                it.second.found_in_models[source_model] += 1;
                return true;
            }
        }
        return false;
    }();
    if (!op_found) {
        const auto &clone_fn = SubgraphsDumper::ClonersMap::cloners.at(op->get_type_info());
        LayerTestsUtils::OPInfo meta(source_model);
        try {
            const std::shared_ptr<ngraph::Node> op_clone = clone_fn(op, meta);
            op_clone->set_friendly_name(op_clone->get_friendly_name() + "_cached");
            m_ops_cache.emplace_back(std::make_pair(op_clone, meta));
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }
}

void OPCache::update_ops_cache(const std::shared_ptr<ngraph::Function> &func, const std::string &source_model) {
    size_t cached_ops_count = m_ops_cache.size();
    for (const auto &op : func->get_ordered_ops()) {
        if (ngraph::is_type<ngraph::op::Parameter>(op) ||
            ngraph::is_type<ngraph::op::Constant>(op) ||
            ngraph::is_type<ngraph::op::Result>(op) ||
            // ReadValue and Assign have to be handled in pair
            // Will be handled as part of 48838
            ngraph::is_type<ngraph::op::AssignBase>(op) ||
            ngraph::is_type<ngraph::op::ReadValueBase>(op)
                    ) {
            continue;
        }
        update_ops_cache(op, source_model);
    }
    std::cout << "\t" << m_ops_cache.size() - cached_ops_count << " new OPs were cached." << std::endl;
}

void OPCache::serialize_cached_ops(const std::string &serialization_dir) {
    if (!CommonTestUtils::directoryExists(serialization_dir)) {
        CommonTestUtils::createDirectoryRecursive(serialization_dir);
    }
    for (const auto &op : m_ops_cache) {
        auto res = serialize_function(op, serialization_dir);
        if (res != OPCache::SerializationStatus::RETRY) {
            continue;
        } else {
            for (size_t i = 1; i <= 5; ++i) {
                std::cout << "Serialization retry #" << i << std::endl;
                res = serialize_function(op, serialization_dir);
                if (res != OPCache::SerializationStatus::RETRY) {
                    break;
                }
            }
        }
    }
}

void OPCache::serialize_meta_info(const LayerTestsUtils::OPInfo &info, const std::string &path) {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("meta_info");
    pugi::xml_node models = root.append_child("models");
    models.append_child("initial_model").append_attribute("name").set_value(info.source_model.c_str());
    for (const auto &model : info.found_in_models) {
        pugi::xml_node model_node = models.append_child("model");
        model_node.append_attribute("name").set_value(model.first.c_str());
        model_node.append_attribute("count").set_value(model.second);
    }
    auto ports_info = root.append_child("ports_info");
    for (const auto &port : info.ports_info) {
        auto port_node = ports_info.append_child("port");
        port_node.append_attribute("id").set_value(port.first);
        if (port.second.min == std::numeric_limits<double>::min()) {
            port_node.append_attribute("max").set_value("undefined");
            port_node.append_attribute("min").set_value("undefined");
        } else {
            port_node.append_attribute("max").set_value(port.second.max);
            port_node.append_attribute("min").set_value(port.second.min);
        }
        port_node.append_attribute("convert_to_const").set_value(port.second.convert_to_const);
    }
    doc.save_file(path.c_str());
}

float OPCache::get_size_of_cached_ops() {
    float size = 0;
    for (const auto &op : m_ops_cache) {
        for (size_t i = 0; i < op.first->get_input_size(); ++i) {
            const auto constant = std::dynamic_pointer_cast<ngraph::opset6::Constant>(
                    op.first->get_input_node_shared_ptr(i));
            if (constant != nullptr) {
                size += static_cast<float>(ngraph::shape_size(constant->get_shape()) *
                                           constant->get_element_type().size()) / (1024 * 1024);
            }
        }
    }
    return size;
}

OPCache::SerializationStatus
OPCache::serialize_function(const std::pair<std::shared_ptr<ngraph::Node>, LayerTestsUtils::OPInfo> &op,
                            const std::string &serialization_dir) {
    try {
        if (op.first->get_friendly_name() == "Relu_8793_cached") {
            std::cout << std::endl;
        }
        std::cout << "Serializing function wrapping op " << op.first << std::endl;
        std::cout << "Taken from model: " << op.second.source_model << std::endl;

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
        auto meta_info = current_op_folder + CommonTestUtils::FileSeparator + op_name + ".meta";
        auto cnn_net = InferenceEngine::CNNNetwork(function);
        cnn_net.serialize(xml_path, bin_path);
        serialize_meta_info(op.second, meta_info);
        return OPCache::SerializationStatus::OK;
    } catch (std::exception &e) {
        std::cout << "Failed to serialize function related to op" << op.first << std::endl
                  << "Exception occurred: " << e.what() << std::endl;
        if (std::string(e.what()).find("Can't open") != std::string::npos) {
            return OPCache::SerializationStatus::RETRY;
        }
        return OPCache::SerializationStatus::FAILED;
    }
}
