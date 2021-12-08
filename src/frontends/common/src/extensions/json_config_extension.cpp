// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/extensions/json_config_extension.hpp"

#include "common/extensions/decoder_transformation_extension.hpp"
#include "common/extensions/json_schema.hpp"
#include "common/extensions/json_transformation_extension.hpp"
#include "so_extension.hpp"

using namespace ov;
using namespace ov::frontend;

JsonConfigExtension::JsonConfigExtension(const std::string& config_path)
    : DecoderTransformationExtension([this](std::shared_ptr<ov::Function> f) {
          bool res = true;
          for (const auto& target_extension : m_target_extensions) {
              auto extension = std::dynamic_pointer_cast<JsonTransformationExtension>(target_extension.first);
              res &= extension->transform(f, target_extension.second);
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
        validator.set_root_schema(json_schema);
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
    std::map<std::string, nlohmann::json> lib_to_sections;
    for (const auto& section : config_json) {
        lib_to_sections[section["library"]].push_back(section);
    }

    // Load all extensions in each library and select required
    for (const auto& it : lib_to_sections) {
        const auto& lib = it.first;
        const auto& sections = it.second;

        auto extensions = detail::load_extensions(lib);
        m_loaded_extensions.insert(m_loaded_extensions.end(), extensions.begin(), extensions.end());
        for (const auto& ext : extensions) {
            auto so_extension = std::dynamic_pointer_cast<ov::detail::SOExtension>(ext);
            OPENVINO_ASSERT(so_extension, "Unexpected extension type loaded from shared library.");
            auto extension = so_extension->extension();
            if (auto json_ext = std::dynamic_pointer_cast<JsonTransformationExtension>(extension)) {
                for (const auto& section : sections) {
                    if (section["id"] == json_ext->id()) {
                        m_target_extensions.push_back({json_ext, section});
                    }
                }
            }
        }
    }
}

JsonConfigExtension::~JsonConfigExtension() {
    // Reset is required here prior unload_extensions, because
    // there shouldn't be any alive references before the unloading the library.
    // Doing it here explicitly to avoid relying on order of class fields definition
    /*    for (const auto& target_extension : m_target_extensions) {
            target_extension.first.reset();
        }*/
}