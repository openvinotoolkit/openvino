// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/config.hpp"

namespace intel_npu {

// Splits the `str` string onto separate elements using `delim` as delimiter and
// call `callback` for each element.
void splitAndApply(const std::string& str, char delim, std::function<void(std::string_view)> callback) {
    const auto begin = str.begin();
    const auto end = str.end();

    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(std::string_view(&(*curBegin), static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}

//
// OptionParser
//

bool OptionParser<bool>::parse(std::string_view val) {
    if (val == "YES") {
        return true;
    } else if (val == "NO") {
        return false;
    }

    OPENVINO_THROW("Value '", val.data(), "' is not a valid BOOL option");
}

int32_t OptionParser<int32_t>::parse(std::string_view val) {
    try {
        return std::stol(val.data());
    } catch (...) {
        OPENVINO_THROW("Value '%s' is not a valid INT32 option", val.data());
    }
}

uint32_t OptionParser<uint32_t>::parse(std::string_view val) {
    try {
        return std::stoul(val.data());
    } catch (...) {
        OPENVINO_THROW("Value '%s' is not a valid UINT32 option", val.data());
    }
}

int64_t OptionParser<int64_t>::parse(std::string_view val) {
    try {
        return std::stoll(val.data());
    } catch (...) {
        OPENVINO_THROW("Value '", val.data(), "' is not a valid INT64 option");
    }
}

uint64_t OptionParser<uint64_t>::parse(std::string_view val) {
    try {
        return std::stoull(val.data());
    } catch (...) {
        OPENVINO_THROW("Value '", val.data(), "' is not a valid UINT64 option");
    }
}

double OptionParser<double>::parse(std::string_view val) {
    try {
        return std::stod(val.data());
    } catch (...) {
        OPENVINO_THROW("Value '", val.data(), "' is not a valid FP64 option");
    }
}

ov::log::Level OptionParser<ov::log::Level>::parse(std::string_view val) {
    std::string strVal(val);
    std::istringstream is(strVal);
    ov::log::Level level;
    is >> level;
    return level;
}

ov::hint::ExecutionMode OptionParser<ov::hint::ExecutionMode>::parse(std::string_view val) {
    std::string strVal(val);
    std::istringstream is(strVal);
    ov::hint::ExecutionMode mode;
    is >> mode;
    return mode;
}

//
// OptionPrinter
//

std::string OptionPrinter<bool>::toString(bool val) {
    return val ? "YES" : "NO";
}

std::string OptionPrinter<ov::log::Level>::toString(ov::log::Level val) {
    std::ostringstream os;
    os << val;
    return os.str();
}

std::string OptionPrinter<ov::hint::ExecutionMode>::toString(ov::hint::ExecutionMode val) {
    std::ostringstream os;
    os << val;
    return os.str();
}

//
// OptionMode
//

std::string_view stringifyEnum(OptionMode val) {
    switch (val) {
    case OptionMode::Both:
        return "Both";
    case OptionMode::CompileTime:
        return "CompileTime";
    case OptionMode::RunTime:
        return "RunTime";
    default:
        return "<UNKNOWN>";
    }
}

//
// OptionValue
//

details::OptionValue::~OptionValue() = default;

//
// OptionsDesc
//

details::OptionConcept OptionsDesc::get(std::string_view key, OptionMode mode) const {
    auto log = Logger::global().clone("OptionsDesc");

    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(std::string(key));
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        log.warning("Deprecated option '%s' was used, '%s' should be used instead", key.data(), searchKey.c_str());
    }

    const auto itMain = _impl.find(searchKey);
    OPENVINO_ASSERT(itMain != _impl.end(),
                    "[ NOT_FOUND ] Option '",
                    key.data(),
                    "' is not supported for current configuration");

    const auto& desc = itMain->second;

    if (mode == OptionMode::RunTime) {
        if (desc.mode() == OptionMode::CompileTime) {
            log.warning("%s option '%s' was used in %s mode",
                        stringifyEnum(desc.mode()).data(),
                        key.data(),
                        stringifyEnum(mode).data());
        }
    }

    return desc;
}

void OptionsDesc::remove(std::string_view key) {
    std::string searchKey{key};
    auto it = _impl.find(searchKey);
    if (it != _impl.end()) {
        _impl.erase(it);
    }
}

void OptionsDesc::reset() {
    _impl.clear();
}

bool OptionsDesc::has(std::string_view key) const {
    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(std::string(key));
    if (itDeprecated != _deprecated.end()) {
        return true;
    }
    const auto itMain = _impl.find(searchKey);
    if (itMain != _impl.end()) {
        return true;
    }
    return false;
}

std::vector<std::string> OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back(p.first);
        }
    }

    return res;
}

std::vector<ov::PropertyName> OptionsDesc::getSupportedOptions(bool includePrivate) const {
    std::vector<ov::PropertyName> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back({p.first, p.second.mutability()});
        }
    }

    return res;
}

std::string OptionsDesc::getSupportedAsString(bool includePrivate) const {
    std::string res;

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res += p.first;
            res += " ";
        }
    }

    return res;
}

void OptionsDesc::walk(std::function<void(const details::OptionConcept&)> cb) const {
    for (const auto& itr : _impl) {
        cb(itr.second);
    }
}

//
// Config
//

Config::Config(const std::shared_ptr<const OptionsDesc>& desc) : _desc(desc) {
    OPENVINO_ASSERT(_desc != nullptr, "Got NULL OptionsDesc");
}

bool Config::hasOpt(std::string_view key) const {
    return _desc->has(key);
}

details::OptionConcept Config::getOpt(std::string_view key) const {
    return _desc->get(key);
}

bool Config::isOptPublic(std::string_view key) const {
    auto log = Logger::global().clone("Config");
    if (_desc->has(key)) {
        return _desc->get(key).isPublic();
    } else {
        log.warning("Option '%s' not registered in config", key.data());
        return true;
    }
}

void Config::parseEnvVars() {
    auto log = Logger::global().clone("Config");

    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                log.trace("Update option '%s' to value '%s' parsed from environment variable '%s'",
                          opt.key().data(),
                          envVar,
                          opt.envVar().data());

                _impl[opt.key().data()] = opt.validateAndParse(envVar);
            }
        }
    });
}

void Config::update(const ConfigMap& options, OptionMode mode) {
    auto log = Logger::global().clone("Config");

    for (const auto& p : options) {
        log.trace("Update option '%s' to value '%s'", p.first.c_str(), p.second.c_str());

        if (isAvailable(p.first)) {
            const auto opt = _desc->get(p.first, mode);
            _impl[opt.key().data()] = opt.validateAndParse(p.second);
        } else {
            OPENVINO_ASSERT("[ NOT_FOUND ] Option '", p.first.c_str(), "' is not supported for current configuration");
        }
    }
}

void Config::fromString(const std::string& str) {
    std::map<std::string, std::string> config;
    std::string str_cfg(str);

    auto parse_token = [&](const std::string& token) {
        auto pos_eq = token.find('=');
        auto key = token.substr(0, pos_eq);
        auto value = token.substr(pos_eq + 2, token.size() - pos_eq - 3);
        config[key] = std::move(value);
    };

    size_t pos = 0;
    std::string token, key, value;
    while ((pos = str_cfg.find(' ')) != std::string::npos) {
        token = str_cfg.substr(0, pos);
        parse_token(token);
        str_cfg.erase(0, pos + 1);
    }

    // Process tail
    parse_token(str_cfg);

    update(config);
}

bool Config::isAvailable(std::string key) const {
    auto it = _enabled.find(key);
    if (it != _enabled.end()) {
        return it->second;
    }
    // if doesnt exist = not available
    return false;
};

void Config::enable(std::string key, bool enabled) {
    // we insert for all cases - no need to check if exists
    _enabled[key] = enabled;
};

void Config::enableAll() {
    _desc->walk([&](const details::OptionConcept& opt) {
        enable(opt.key().data(), true);
    });
}

void Config::walkEnables(std::function<void(const std::string&)> cb) const {
    for (const auto& itr : _enabled) {
        cb(itr.first);
    }
}

std::string Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto& key = it->first;

        // include only enabled configs
        if (isAvailable(key)) {
            resultStream << key << "=\"" << it->second->toString() << "\"";
            if (std::next(it) != _impl.end()) {
                resultStream << " ";
            }
        }
    }

    return resultStream.str();
}

std::string Config::toStringForCompiler() const {
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

void Config::addOrUpdateInternal(std::string key, std::string value) {
    auto log = Logger::global().clone("Config");
    if (_internal_compiler_configs.count(key) != 0) {
        log.warning("Internal compiler option '%s' was already registered! Updating value only!", key.c_str());
        _internal_compiler_configs.at(key) = value;
    } else {
        // manual insert_or_assign
        auto it = _internal_compiler_configs.find(key);
        if (it != _internal_compiler_configs.end()) {
            it->second = value;  // only update
        } else {
            _internal_compiler_configs.insert(std::make_pair(key, value));  // insert new
        }
    }
};

std::string Config::getInternal(std::string key) const {
    auto log = Logger::global().clone("Config");
    if (_internal_compiler_configs.count(key) == 0) {
        OPENVINO_THROW(std::string("Internal compiler option " + key + " does not exist! "));
    }
    return _internal_compiler_configs.at(key);
};

std::string Config::toStringForCompilerInternal() const {
    std::stringstream resultStream;

    for (auto it = _internal_compiler_configs.cbegin(); it != _internal_compiler_configs.cend(); ++it) {
        resultStream << it->first << "=\"" << it->second << "\"";
    }

    return resultStream.str();
};

//
// envVarStrToBool
//

bool envVarStrToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        OPENVINO_THROW(std::string("Environment variable ") + varName + " has wrong value : " + e.what());
    }
}

}  // namespace intel_npu
