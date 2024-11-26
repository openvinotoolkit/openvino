// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiled_model_factory.hpp"
#include "logging.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "compiled_model.hpp"
#include "llm_compiled_model.hpp"

std::shared_ptr<ov::ICompiledModel>
ov::npuw::CompiledModelFactory::create(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const ov::AnyMap& properties) {
    LOG_VERB(__PRETTY_FUNCTION__);
    LOG_BLOCK();
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    auto use_dynamic_llm_key = ov::intel_npu::npuw::dynamic_llm::enabled.name();
    if (properties.count(use_dynamic_llm_key) &&
        properties.at(use_dynamic_llm_key).as<bool>() == true) {
        LOG_DEBUG("ov::npuw::LLMCompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, properties);
    } else {
        LOG_DEBUG("ov::npuw::CompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::CompiledModel>(model, plugin, properties);
    }
    LOG_DEBUG("Done");
    return compiled_model;
}
