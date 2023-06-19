// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <nlohmann/json.hpp>

#include "transformer_configuration_loader.hpp"

namespace transformation_sample {

// TODO write tests
class TransformerConfigurationLoaderImpl : public TransformerConfigurationLoader {
public:
    TransformerConfiguration parse_configuration(const std::string& path_to_file) const override;
    TransformerConfiguration parse_configuration(std::istream& stream_with_json) const override;

private:
    std::map<std::string, std::string> retrieve_gna_configuration(const nlohmann::json& root) const;
    std::vector<std::string> retrieve_transformations_names(const nlohmann::json& root) const;
    std::string retrieve_string(const nlohmann::json& root,
                                const std::string& field_name,
                                const std::string& parent_name) const;

    static const std::string kConfigurationFieldName;
    static const std::string kTransformationsListFieldName;
    static const std::string kTransformationFieldName;
};

}  // namespace transformation_sample
