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

DecoderTransformationExtension::DecoderTransformationExtension (std::function<bool(std::shared_ptr<ov::Function>)> function_pass) :
    m_registration(
            [function_pass](ov::pass::Manager& manager) {
                manager.register_pass<CustomFunctionPass>(function_pass);
            }) {}

// Create a custom matcher pass where the code of matcher pass initialization is a given function.
DecoderTransformationExtension::DecoderTransformationExtension (std::function<void(ov::pass::MatcherPass*)> matcher_pass_initializer) :
    m_registration(
            [matcher_pass_initializer](ov::pass::Manager& manager) {
                manager.register_pass<CustomMatcherPass>(matcher_pass_initializer);
            }) {}

void DecoderTransformationExtension::register_pass (ov::pass::Manager& manager) const {
    m_registration(manager);
}

JsonConfigExtension::JsonConfigExtension (const std::string& config_path) :
    DecoderTransformationExtension([config_path](std::shared_ptr<ov::Function> f){
        // TODO: Implement real code here
        // This is a placeholder that just demonstrates basic steps of config-enabled transformation.
        // Real transformation should do some pre-work to load a target transformation identified
        // in json config file from a library file and then here just to call it.
        // In this placeholder we do all the steps (partially synthetic) as a part of the transformation function itself
        nlohmann::json config_json;
        std::ifstream config_file(config_path);
        config_file >>
                    config_json;
        std::cerr << "++++++++++++++ Read json: ++++++++++++++++\n" <<
                  config_json;

        auto ops = f->get_ordered_ops();
        std::cerr << "HELLO! Run on function with " << ops.size() << " nodes\n";
        return true;
    }) {}
