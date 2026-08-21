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

void FilteredConfig::update(const ConfigMap& options) {
    auto log = Logger::global().clone("Config");

    for (const auto& p : options) {
        log.trace("Update option '%s' to value '%s'", p.first.c_str(), p.second.c_str());
        if (_desc->has(p.first)) {
            const auto opt = _desc->get(p.first);
            _impl[opt.key().data()] = opt.validateAndParseFromString(p.second);
        } else {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + p.first + "' is not supported for current configuration");
        }
    }
}

void FilteredConfig::updateAny(const ov::AnyMap& options) {
    auto log = Logger::global().clone("Config");

    for (const auto& p : options) {
        log.trace("Update option '%s' to given 'ov::Any' value", p.first.c_str());
        if (_desc->has(p.first)) {
            const auto opt = _desc->get(p.first);
            _impl[opt.key().data()] = opt.validateAndParseFromAny(p.second);
        } else {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + p.first + "' is not supported for current configuration");
        }
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

std::string FilteredConfig::getInternal(std::string key) const {
    if (_internal_compiler_configs.count(key) == 0) {
        OPENVINO_THROW(std::string("Internal compiler option " + key + " does not exist! "));
    }
    return _internal_compiler_configs.at(key);
}

std::string FilteredConfig::toStringForCompiler(const std::function<bool(std::string_view)>& isSupported) const {
    std::stringstream resultStream;
    bool hasSerializedValue = false;

    for (const auto& [key, value] : _impl) {
        if (_desc->has(key) && _desc->get(key).mode() != OptionMode::RunTime && isSupported(key)) {
            if (hasSerializedValue) {
                resultStream << " ";
            }
            resultStream << key << "=\"" << value->toString() << "\"";
            hasSerializedValue = true;
        }
    }

    for (const auto& [key, value] : _internal_compiler_configs) {
        if (isSupported(key)) {
            if (hasSerializedValue) {
                resultStream << " ";
            }
            resultStream << key << "=\"" << value << "\"";
            hasSerializedValue = true;
        }
    }

    return resultStream.str();
}

}  // namespace intel_npu
