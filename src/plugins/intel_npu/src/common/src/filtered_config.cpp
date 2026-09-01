// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/filtered_config.hpp"

namespace intel_npu {

bool FilteredConfig::hasOpt(std::string_view key) const {
    return _desc->has(key);
}

details::OptionConcept FilteredConfig::getOpt(std::string_view key) const {
    return _desc->get(key);
}

bool FilteredConfig::isOptPublic(std::string_view key) const {
    auto log = Logger::global().clone("Config");
    if (_desc->has(key)) {
        return _desc->get(key).isPublic();
    } else {
        log.warning("Option '%s' not registered in config", key.data());
        return true;
    }
}

void FilteredConfig::walkInternals(std::function<void(const std::string&)> cb) const {
    for (const auto& itr : _internal_compiler_configs) {
        cb(itr.first);
    }
}

void FilteredConfig::addOrUpdateInternal(std::string key, std::string value) {
    auto log = Logger::global().clone("Config");
    if (_internal_compiler_configs.count(key) != 0) {
        log.warning("Internal compiler option '%s' was already registered! Updating value only!", key.c_str());
        _internal_compiler_configs.at(key) = std::move(value);
    } else {
        // manual insert
        log.trace("Store internal compiler option %s: %s", key.c_str(), value.c_str());
        _internal_compiler_configs.emplace(key, std::move(value));
    }
}

bool FilteredConfig::hasInternal(std::string_view key) const {
    return _internal_compiler_configs.count(std::string(key)) != 0;
}

void FilteredConfig::removeCompileTimeConfigs() {
    for (auto it = _impl.begin(); it != _impl.end();) {
        if (_desc->get(it->first).mode() == OptionMode::CompileTime) {
            it = _impl.erase(it);
        } else {
            ++it;
        }
    }

    _internal_compiler_configs.clear();
}

std::string FilteredConfig::getInternal(std::string key) const {
    if (_internal_compiler_configs.count(key) == 0) {
        OPENVINO_THROW(std::string("Internal compiler option " + key + " does not exist! "));
    }
    return _internal_compiler_configs.at(key);
}

std::string FilteredConfig::toStringForCompiler(const std::function<bool(const std::string&)>& isSupported) const {
    if (!isSupported) {
        OPENVINO_THROW("FilteredConfig::toStringForCompiler requires a valid support predicate");
    }

    std::stringstream resultStream;
    bool hasSerializedValue = false;

    const auto append = [&](const std::string& key, const std::string& serializedValue) {
        if (hasSerializedValue) {
            resultStream << " ";
        }
        resultStream << key << "=\"" << serializedValue << "\"";
        hasSerializedValue = true;
    };

    for (const auto& [key, value] : _impl) {
        if (!_desc->has(key)) {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + std::string(key) +
                           "' is not supported for current configuration");
        }

        const auto mode = _desc->get(key).mode();
        if (mode != OptionMode::CompileTime && mode != OptionMode::Both) {
            continue;
        }
        if (mode == OptionMode::CompileTime && !isSupported(std::string(key))) {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + std::string(key) +
                           "' is not supported for current configuration");
        }
        if (mode == OptionMode::Both && !isSupported(std::string(key))) {
            continue;
        }

        append(std::string(key), value->toString());
    }

    for (const auto& [key, value] : _internal_compiler_configs) {
        if (!isSupported(std::string(key))) {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + std::string(key) +
                           "' is not supported for current configuration");
        }
        append(std::string(key), value);
    }

    return resultStream.str();
}

}  // namespace intel_npu
