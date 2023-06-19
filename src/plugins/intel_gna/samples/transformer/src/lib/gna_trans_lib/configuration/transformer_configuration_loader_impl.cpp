// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer_configuration_loader_impl.hpp"

#include <fstream>
#include <openvino/runtime/intel_gna/properties.hpp>

namespace transformation_sample {

const std::string TransformerConfigurationLoaderImpl::kConfigurationFieldName = "configuration";
const std::string TransformerConfigurationLoaderImpl::kTransformationsListFieldName = "transformations_list";
const std::string TransformerConfigurationLoaderImpl::kTransformationFieldName = "name";

TransformerConfiguration TransformerConfigurationLoaderImpl::parse_configuration(
    const std::string& path_to_file) const {
    std::ifstream file_stream(path_to_file);
    if (!file_stream) {
        throw JSONParsingException("File loading error. Check if file exists: " + path_to_file);
    }

    try {
        return parse_configuration(file_stream);
    } catch (const std::exception& exc) {
        std::string error_message = "Error when parsing file: " + path_to_file + exc.what();
        throw JSONParsingException(error_message);
    }
}

TransformerConfiguration TransformerConfigurationLoaderImpl::parse_configuration(std::istream& stream_with_json) const {
    try {
        auto parsed_json = nlohmann::json::parse(stream_with_json);
        if (!parsed_json.is_object()) {
            throw JSONParsingException("Config file should contain at least one json object!: ");
        }

        TransformerConfiguration config;

        config.gna_configuration = retrieve_gna_configuration(parsed_json);
        config.transformations_names = retrieve_transformations_names(parsed_json);
        return config;
    } catch (const JSONParsingException& exc) {
        // to rethrow current exception
        throw exc;
    } catch (const std::exception& exc) {
        // to catch json library exception and add extra message.
        throw JSONParsingException(std::string("Error when parsing configuration file: ") + exc.what());
    }
}

std::map<std::string, std::string> TransformerConfigurationLoaderImpl::retrieve_gna_configuration(
    const nlohmann::json& root) const {
    if (!root.contains(kConfigurationFieldName)) {
        throw JSONParsingException(std::string("Configuration file does not contain: ") + kConfigurationFieldName);
    }

    auto configuration = root.at(kConfigurationFieldName);
    if (!configuration.is_object()) {
        throw JSONParsingException(std::string("Configuration field should be an object: ") + kConfigurationFieldName);
    }

    std::map<std::string, std::string> config;

    // to set execution target
    std::string field_name = ov::intel_gna::execution_target.name();
    std::string value = retrieve_string(configuration, field_name, kConfigurationFieldName);
    config.emplace(field_name, value);

    // to set gna_precision
    field_name = ov::hint::inference_precision.name();
    value = retrieve_string(configuration, field_name, kConfigurationFieldName);
    config.emplace(field_name, value);

    // pwl_max_error_precentage
    field_name = ov::intel_gna::pwl_max_error_percent.name();
    value = retrieve_string(configuration, field_name, kConfigurationFieldName);
    config.emplace(field_name, value);

    // Transformations uses also input_low_preciosion from config, but there is no possibility
    // to change its value, so default value is used.

    return config;
}

std::vector<std::string> TransformerConfigurationLoaderImpl::retrieve_transformations_names(
    const nlohmann::json& root) const {
    if (!root.contains(kTransformationsListFieldName)) {
        throw JSONParsingException(std::string("Configuration file does not contain: ") +
                                   kTransformationsListFieldName);
    }

    auto transformations_list = root.at(kTransformationsListFieldName);

    if (!transformations_list.is_array()) {
        throw JSONParsingException(std::string("Configuration field should be an array: ") +
                                   kTransformationsListFieldName);
    }

    std::vector<std::string> transformations;
    for (const auto& transformation : transformations_list) {
        transformations.push_back(
            retrieve_string(transformation, kTransformationFieldName, kTransformationsListFieldName));
    }
    return transformations;
}

std::string TransformerConfigurationLoaderImpl::retrieve_string(const nlohmann::json& root,
                                                                const std::string& field_name,
                                                                const std::string& parent_name) const {
    if (!root.contains(field_name)) {
        std::string error_message = parent_name + " does not contain field: " + field_name;
        throw JSONParsingException(error_message);
    }

    if (!root.at(field_name).is_string()) {
        std::string error_message = field_name + " of " + parent_name + " should be string!";
        throw JSONParsingException(error_message);
    }

    return root.at(field_name);
}

}  // namespace transformation_sample