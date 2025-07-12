// Copyright (C) 2018-2025 Intel Corporation
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

void FilteredConfig::update(const ConfigMap& options, OptionMode mode) {
    auto log = Logger::global().clone("Config");

    for (const auto& p : options) {
        log.trace("Update option '%s' to value '%s'", p.first.c_str(), p.second.c_str());

        if (isAvailable(p.first)) {
            const auto opt = _desc->get(p.first, mode);
            _impl[opt.key().data()] = opt.validateAndParse(p.second);
        } else {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + p.first + "' is not supported for current configuration");
        }
    }
}

bool FilteredConfig::isAvailable(std::string key) const {
    // NPUW properties are requested by OV Core during caching and have no effect on the NPU plugin. But we still need
    // to enable those for OV Core to query.
    if (key.find("NPUW") != key.npos) {
        return true;  // always available
    }
    auto it = _enabled.find(key);
    if (it != _enabled.end() && hasOpt(key)) {
        return it->second;
    }
    // if doesnt exist = not available
    return false;
}

void FilteredConfig::enable(std::string key, bool enabled) {
    // we insert for all cases - no need to check if exists
    _enabled[key] = enabled;
}

void FilteredConfig::enableAll() {
    _desc->walk([&](const details::OptionConcept& opt) {
        enable(opt.key().data(), true);
    });
}

void FilteredConfig::walkEnables(std::function<void(const std::string&)> cb) const {
    for (const auto& itr : _enabled) {
        cb(itr.first);
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
        _internal_compiler_configs.insert(std::make_pair(key, value));  // insert new
    }
}

std::string FilteredConfig::getInternal(std::string key) const {
    auto log = Logger::global().clone("Config");
    if (_internal_compiler_configs.count(key) == 0) {
        OPENVINO_THROW(std::string("Internal compiler option " + key + " does not exist! "));
    }
    return _internal_compiler_configs.at(key);
}

std::string FilteredConfig::toStringForCompilerInternal() const {
    std::stringstream resultStream;

    for (auto it = _internal_compiler_configs.cbegin(); it != _internal_compiler_configs.cend(); ++it) {
        resultStream << " " << it->first << "=\"" << it->second << "\"";
    }

    return resultStream.str();
}

std::string FilteredConfig::toStringForCompiler() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto& key = it->first;

        // Only include available configs which options have OptionMode::Compile or OptionMode::Both
        if (isAvailable(key)) {
            if (_desc->has(key)) {
                if (_desc->get(key).mode() != OptionMode::RunTime) {
                    resultStream << key << "=\"" << it->second->toString() << "\"";
                    if (std::next(it) != _impl.end()) {
                        resultStream << " ";
                    }
                }
            }
        }
    }

    return resultStream.str();
}

}  // namespace intel_npu
