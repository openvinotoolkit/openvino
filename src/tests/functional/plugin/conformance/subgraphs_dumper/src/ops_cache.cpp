// Copyright (C) 2018-2023 Intel Corporation
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

void OPCache::update_ops_cache(const std::shared_ptr<ov::Node> &op,
                               const std::string &source_model) {
    const std::shared_ptr<ov::Node> cachedOp = [&] {
        for (auto &&it : m_ops_cache) {
            if (manager.match_any(it.first, op, it.second)) {
                it.second.found_in_models[source_model] += 1;
                return it.first;
            }
        }
        return std::shared_ptr<ov::Node>{};
    }();

    auto saveOpToCash = [&] {
        try {
            const auto& clone_fn = SubgraphsDumper::ClonersMap::cloners.at(op->get_type_info());
            LayerTestsUtils::OPInfo meta(source_model);
            const std::shared_ptr<ov::Node> op_clone = clone_fn(op, meta);
            if (!op_clone) {
                return;
            }
            op_clone->set_friendly_name(op_clone->get_friendly_name() + "_cached");
            m_ops_cache.insert({op_clone, meta});
        } catch (std::out_of_range& e) {
            std::cout << "WARNING: Cloner for " << op->get_type_name() << " (" << op->get_type_info().get_version()
                      << ") isn't found: " << e.what() << std::endl;
        } catch (std::exception& e) {
            std::cout << "ERROR: " << e.what() << std::endl;
        }
    };

    if (!cachedOp.get()) {
        saveOpToCash();
    } else {
        for (int i = 0; i < op->get_input_size(); i++) {
            auto shape = op->get_input_shape(i);
            unsigned long shapeSize = ov::shape_size(shape) * op->get_output_element_type(0).size();

            auto cachedOpShape = cachedOp->get_input_shape(i);
            unsigned long cachedOpShapeSize =
                ov::shape_size(cachedOpShape) * cachedOp->get_output_element_type(0).size();

            if (shapeSize < cachedOpShapeSize) {
                m_ops_cache.erase(cachedOp);
                saveOpToCash();
            }
        }
    }
}

void OPCache::update_ops_cache(const std::shared_ptr<ov::Model> &func, const bool extract_body, const std::string &source_model) {
    size_t cached_ops_count = m_ops_cache.size();
    for (const auto &op : func->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Constant>(op) ||
            std::dynamic_pointer_cast<ov::op::v0::Result>(op) ||
            // ReadValue and Assign have to be handled in pair
            // Will be handled as part of 48838
            std::dynamic_pointer_cast<ov::op::util::AssignBase>(op) ||
            std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(op)
                    ) {
            continue;
        }
        if (extract_body) {
            if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
                auto if_op = std::dynamic_pointer_cast<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    update_ops_cache(if_body, extract_body, source_model);
                }
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = std::dynamic_pointer_cast<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                update_ops_cache(loop_body, extract_body, source_model);
            } else if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_body();
                update_ops_cache(ti_body, extract_body, source_model);
            }
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
        model_node.append_attribute("count").set_value(static_cast<unsigned long long>(model.second));
    }
    auto ports_info = root.append_child("ports_info");
    for (const auto &port : info.ports_info) {
        auto port_node = ports_info.append_child("port");
        port_node.append_attribute("id").set_value(static_cast<unsigned long long>(port.first));
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
            const auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    op.first->get_input_node_shared_ptr(i));
            if (constant != nullptr) {
                size += static_cast<float>(ov::shape_size(constant->get_shape()) *
                                           constant->get_output_element_type(0).size()) / (1024 * 1024);
            }
        }
    }
    return size;
}

OPCache::SerializationStatus
OPCache::serialize_function(const std::pair<std::shared_ptr<ov::Node>, LayerTestsUtils::OPInfo> &op,
                            const std::string &serialization_dir) {
    try {
        std::cout << "Serializing function wrapping op " << op.first << std::endl;
        std::cout << "Taken from model: " << op.second.source_model << std::endl;

        ov::ParameterVector params;
        bool is_dynamic = false;
        for (size_t i = 0; i < op.first->get_input_size(); ++i) {
            if (ov::op::util::is_parameter(op.first->get_input_node_ptr(i))) {
                auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
                        op.first->get_input_node_shared_ptr(i));
                if (param->get_partial_shape().is_dynamic()) {
                    is_dynamic = true;
                }
                params.push_back(param);
            }
        }
        ov::ResultVector results;
        for (auto &out : op.first->outputs()) {
            if (out.get_partial_shape().is_dynamic()) {
                is_dynamic = true;
            }
            results.push_back(std::make_shared<ov::op::v0::Result>(out));
        }
        auto function = std::make_shared<ov::Model>(results, params);

        // TODO: How to define element type for multi-output ops
        std::string op_folder_name = op.first->get_type_info().name;
        std::string opset_version = op.first->get_type_info().get_version();
        std::string opset_name = "opset";
        auto pos = opset_version.find(opset_name);
        if (pos != std::string::npos) {
            op_folder_name += "-" + opset_version.substr(pos + opset_name.size());
        }
        auto op_el_type = op.first->get_output_element_type(0).get_type_name();
        auto current_op_folder = serialization_dir + CommonTestUtils::FileSeparator +
                                 (is_dynamic ? "dynamic" : "static") + CommonTestUtils::FileSeparator +
                                 op_folder_name + CommonTestUtils::FileSeparator + op_el_type;
        auto op_name = op.first->get_name();
        op_name = op_name.substr(op_name.find("_") + 1);

        std::cout << op_name << " will be serialized to " << current_op_folder << std::endl;
        if (!CommonTestUtils::directoryExists(current_op_folder)) {
            CommonTestUtils::createDirectoryRecursive(current_op_folder);
        }
        std::replace(op_name.begin(), op_name.end(), '/', '_');
        std::replace(op_name.begin(), op_name.end(), '\\', '_');
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
