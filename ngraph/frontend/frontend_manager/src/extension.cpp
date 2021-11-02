// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/extension.hpp"
#include "../../../../thirdparty/nlohmann/json/json.hpp"


using namespace ngraph;
using namespace ngraph::frontend;

// Helper class to register user function as a FunctionPass
class CustomFunctionPass : public ov::pass::FunctionPass {
public:

    CustomFunctionPass (std::function<bool(std::shared_ptr<ov::Function>)> pass) :
            m_pass(pass)
    {}

    bool run_on_function (std::shared_ptr<ov::Function> f) override
    {
        return m_pass(f);
    }
private:

    std::function<bool(std::shared_ptr<ov::Function>)> m_pass;
};

// Helper class to register user matcher pass initialization as a MatcherPass
class CustomMatcherPass : public ov::pass::MatcherPass {
public:

    CustomMatcherPass (std::function<void(ov::pass::MatcherPass*)> matcher_pass_initializer) {
        matcher_pass_initializer(this);
    }
};

DecoderTransformationExtension::DecoderTransformationExtension (
        std::function<bool(std::shared_ptr<ov::Function>)> function_pass) :
    m_registration(
            [function_pass](ov::pass::Manager& manager) {
                manager.register_pass<CustomFunctionPass>(function_pass);
            }) {}

// Create a custom matcher pass where the code of matcher pass initialization is a given function.
DecoderTransformationExtension::DecoderTransformationExtension (
        std::function<void(ov::pass::MatcherPass*)> matcher_pass_initializer) :
    m_registration(
            [matcher_pass_initializer](ov::pass::Manager& manager) {
                manager.register_pass<CustomMatcherPass>(matcher_pass_initializer);
            }) {}

void DecoderTransformationExtension::register_pass (ov::pass::Manager& manager) const {
    m_registration(manager);
}

JsonConfigExtension::JsonConfigExtension(const std::string& config_path) :
    DecoderTransformationExtension([this](std::shared_ptr<ov::Function> f) {
        auto extension = std::dynamic_pointer_cast<JsonTransformationExtension>(m_target_extension);
        return extension->transform(f, m_replacement_descriptions);
    }) {
    nlohmann::json config_json;
    std::ifstream config_file(config_path);
    config_file >> config_json;
    OPENVINO_ASSERT(config_json.size() == 1, "JSON config with only one section can be processed. Found none or multiple sections, not one.");
    auto library = config_json[0]["library"];
    auto extension_id = config_json[0]["id"];
    std::cerr << "Trying to load library " << library;
    m_loaded_extensions = ov::detail::load_extensions(library);
    std::cerr << "Loaded extensions successfully";

    for(auto extension: m_loaded_extensions) {
        if(auto json_ext = std::dynamic_pointer_cast<JsonTransformationExtension>(extension)){
            if(json_ext->id() == extension_id) {
                OPENVINO_ASSERT(!m_target_extension, "Multiple sections with the same ID were found, the only one is allowed.");
                m_target_extension = json_ext;
            }
        }
    }
    OPENVINO_ASSERT(m_target_extension, "No extension with a given ID from JSON config was found.");
}

JsonConfigExtension::~JsonConfigExtension () {
    // reset is required here prior unload_extensions, because
    // there shouldn't be any alive references before the unloading
    m_target_extension.reset();

    // TODO: Delete this call and rework initialization part as master version has changed
    //ov::detail::unload_extensions(m_loaded_extensions);
}