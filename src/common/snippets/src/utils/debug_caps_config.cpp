// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include "snippets/utils/debug_caps_config.hpp"

namespace ov {
namespace snippets {

void DebugCapsConfig::readProperties() {
    auto readEnv = [](const char* envVar) {
        const char* env = std::getenv(envVar);
        if (env && *env)
            return env;

        return (const char*)nullptr;
    };

    const char* envVarValue = nullptr;
    if ((envVarValue = readEnv("OV_SNIPPETS_DUMP_LIR"))) {
        dumpLIR.parseAndSet(envVarValue);
        OPENVINO_ASSERT(!dumpLIR.passes.empty(), "Passes option in OV_SNIPPETS_DUMP_LIR must be provided.");
    }
}

void DebugCapsConfig::PropertyGroup::parseAndSet(const std::string& str) {
    const auto& options = ov::util::split(str, ' ');
    const auto& propertySetters = getPropertySetters();
    bool failed = false;
    auto getHelp = [propertySetters]() {
        std::string help;
        for (const auto& property : propertySetters)
            help.append('\t' + property->getPropertyName() + "=<" + property->getPropertyValueDescription() + ">\n");
        return help;
    };

    for (const auto& option : options) {
        const auto& parts = ov::util::split(option, '=');
        if (parts.size() > 2) {
            failed = true;
            break;
        }
        const auto& propertyName = ov::util::to_lower(parts.front());
        const auto& foundSetter = std::find_if(propertySetters.begin(),
                                               propertySetters.end(),
                                               [propertyName](const PropertySetterPtr& setter) {
                                                   return setter->getPropertyName() == propertyName;
                                               });
        if (foundSetter == propertySetters.end() ||
            !(*foundSetter)->parseAndSet(parts.size() == 1 ? "" : parts.back())) {
            failed = true;
            break;
        }
    }

    if (failed)
        OPENVINO_THROW("Wrong syntax: ",
                       str,
                       "\n",
                       "The following space separated options are supported (option names are case insensitive):",
                       "\n",
                       getHelp());
}

}  // namespace snippets
}  // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
