// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension/json_config.hpp"

#include "openvino/core/deprecated.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
#include "nlohmann/json-schema.hpp"
OPENVINO_SUPPRESS_DEPRECATED_END

#include "extension/json_transformation.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "so_extension.hpp"

namespace {
static const nlohmann::json validation_schema =
    R"(
{
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Root",
        "type": "array",
        "default": [],
        "items": {
            "$id": "#root/items",
                    "title": "Items",
                    "type": "object",
                    "required": [
            "id",
                    "match_kind"
            ],
            "properties": {
                "custom_attributes": {
                    "$id": "#root/items/custom_attributes",
                            "title": "Custom_attributes",
                            "type": "object",
                            "properties": {
                    }
                },
                "id": {
                    "$id": "#root/items/id",
                            "title": "Id",
                            "type": "string",
                            "pattern": "^.*$",
                            "minLength": 1
                },
                "inputs": {
                    "$id": "#root/items/inputs",
                            "title": "Inputs",
                            "type": "array",
                            "default": [],
                            "items": {
                        "$id": "#root/items/inputs/items",
                                "title": "Items",
                                "type": "array",
                                "default": [],
                                "items": {
                            "$id": "#root/items/inputs/items/items",
                                    "title": "Items",
                                    "type": "object",
                                    "properties": {
                                "node": {
                                    "$id": "#root/items/inputs/items/items/node",
                                            "title": "Node",
                                            "type": "string",
                                            "default": "",
                                            "pattern": "^.*$"
                                },
                                "port": {
                                    "$id": "#root/items/inputs/items/items/port",
                                            "title": "Port",
                                            "type": "integer",
                                            "default": 0
                                }
                            },
                            "required": ["node", "port"]
                        }

                    }
                },
                "instances": {
                    "$id": "#root/items/instances",
                            "title": "Instances",
                            "type": ["array", "object"],
                    "items": {
                        "$id": "#root/items/instances/items",
                                "title": "Items",
                                "type": "string",
                                "default": "",
                                "pattern": "^.*$"
                    }
                },
                "match_kind": {
                    "$id": "#root/items/match_kind",
                            "title": "Match_kind",
                            "type": "string",
                            "enum": ["points", "scope", "general"],
                    "default": "points",
                            "pattern": "^.*$"
                },
                "outputs": {
                    "$id": "#root/items/outputs",
                            "title": "Outputs",
                            "type": "array",
                            "default": [],
                            "items": {
                        "$id": "#root/items/outputs/items",
                                "title": "Items",
                                "type": "object",
                                "properties": {
                            "node": {
                                "$id": "#root/items/outputs/items/node",
                                        "title": "Node",
                                        "type": "string",
                                        "default": "",
                                        "pattern": "^.*$"
                            },
                            "port": {
                                "$id": "#root/items/outputs/items/port",
                                        "title": "Port",
                                        "type": "integer",
                                        "default": 0
                            }
                        },
                        "required": ["node", "port"]
                    }

                },
                "include_inputs_to_sub_graph": {
                    "$id": "#root/items/include_inputs_to_sub_graph",
                            "title": "Include_inputs_to_sub_graph",
                            "type": "boolean",
                            "default": false
                },
                "include_outputs_to_sub_graph": {
                    "$id": "#root/items/include_outputs_to_sub_graph",
                            "title": "Include_outputs_to_sub_graph",
                            "type": "boolean",
                            "default": false
                }
            }
        }
}
)"_json;
}  //  namespace

using namespace ov;
using namespace ov::frontend;

JsonConfigExtension::JsonConfigExtension(const std::string& config_path)
    : DecoderTransformationExtension([this](std::shared_ptr<ov::Model> f) {
          bool res = false;
          for (const auto& target_extension : m_target_extensions) {
              if (auto extension = std::dynamic_pointer_cast<JsonTransformationExtension>(target_extension.first)) {
                  res |= extension->transform(f, target_extension.second);
              }
          }
          return res;
      }) {
    // Load JSON config
    nlohmann::json config_json;
    std::ifstream config_file(config_path);
    config_file >> config_json;

    // Validate JSON config
    nlohmann::json_schema::json_validator validator;
    try {
        validator.set_root_schema(validation_schema);
    } catch (const std::exception& e) {
        OPENVINO_ASSERT(false, "Invalid json schema : ", e.what());
    }

    try {
        validator.validate(config_json);
    } catch (const std::exception& e) {
        OPENVINO_ASSERT(false, "Json schema validation failed: ", e.what());
    }

    // Parse JSON Extensions

    // Group sections describing transformations by library.
    std::unordered_map<std::string, nlohmann::json> lib_to_sections;
    for (const auto& section : config_json) {
        lib_to_sections[section["library"]].push_back(section);
    }

    // Load all extensions in each library and select required
    for (const auto& it : lib_to_sections) {
        const auto& lib = it.first;
        const auto& sections = it.second;

        auto extensions = ov::detail::load_extensions(lib);
        m_loaded_extensions.insert(m_loaded_extensions.end(), extensions.begin(), extensions.end());
        for (const auto& ext : extensions) {
            auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext);
            OPENVINO_ASSERT(so_extension, "Unexpected extension type loaded from shared library.");
            auto extension = so_extension->extension();
            if (auto json_ext = std::dynamic_pointer_cast<JsonTransformationExtension>(extension)) {
                for (const auto& section : sections) {
                    if (section["id"] == json_ext->id()) {
                        m_target_extensions.push_back({json_ext, section.dump()});
                    }
                }
            }
        }
    }
}