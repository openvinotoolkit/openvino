// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/config.hpp"

#include <cctype>
#include <limits>

namespace intel_npu {

namespace {

// `std::sto*` stops at the first character which is not part of the number, so without this check a valid
// prefix would be enough to accept malformed values such as "12oops". Only trailing whitespace is tolerated.
void assertFullyConsumed(const std::string& str, size_t pos) {
    while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos]))) {
        ++pos;
    }

    OPENVINO_ASSERT(pos == str.size());
}

}  // namespace

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
    std::string strVal(val);
    std::transform(strVal.begin(), strVal.end(), strVal.begin(), [](char c) {
        return std::toupper(c);
    });
    // Note: the spellings accepted here must stay a superset of the ones `ov::util::Read<bool>` accepts, since
    // both paths lead to the same options (`update()` vs `updateAny()` with a string payload)
    if (strVal == "YES" || strVal == "TRUE" || strVal == "ON" || strVal == "1") {
        return true;
    } else if (strVal == "NO" || strVal == "FALSE" || strVal == "OFF" || strVal == "0") {
        return false;
    }

    OPENVINO_THROW("Value '", val.data(), "' is not a valid BOOL option");
}

int32_t OptionParser<int32_t>::parse(std::string_view val) {
    try {
        const std::string str(val);
        size_t pos = 0;
        const auto parsed = std::stoll(str, &pos);
        assertFullyConsumed(str, pos);
        OPENVINO_ASSERT(parsed >= std::numeric_limits<int32_t>::min() && parsed <= std::numeric_limits<int32_t>::max());
        return static_cast<int32_t>(parsed);
    } catch (...) {
        OPENVINO_THROW("Value '", val, "' is not a valid INT32 option");
    }
}

uint32_t OptionParser<uint32_t>::parse(std::string_view val) {
    try {
        // Note: "std::stoul" silently wraps negative values around, hence the signed intermediate
        const std::string str(val);
        size_t pos = 0;
        const auto parsed = std::stoll(str, &pos);
        assertFullyConsumed(str, pos);
        OPENVINO_ASSERT(parsed >= 0 && parsed <= std::numeric_limits<uint32_t>::max());
        return static_cast<uint32_t>(parsed);
    } catch (...) {
        OPENVINO_THROW("Value '", val, "' is not a valid UINT32 option");
    }
}

int64_t OptionParser<int64_t>::parse(std::string_view val) {
    try {
        const std::string str(val);
        size_t pos = 0;
        const auto parsed = std::stoll(str, &pos);
        assertFullyConsumed(str, pos);
        return parsed;
    } catch (...) {
        OPENVINO_THROW("Value '", val, "' is not a valid INT64 option");
    }
}

uint64_t OptionParser<uint64_t>::parse(std::string_view val) {
    try {
        // Note: "std::stoull" silently wraps negative values around, hence the explicit check
        const std::string str(val);
        OPENVINO_ASSERT(str.find('-') == std::string::npos);
        size_t pos = 0;
        const auto parsed = std::stoull(str, &pos);
        assertFullyConsumed(str, pos);
        return parsed;
    } catch (...) {
        OPENVINO_THROW("Value '", val, "' is not a valid UINT64 option");
    }
}

double OptionParser<double>::parse(std::string_view val) {
    try {
        const std::string str(val);
        size_t pos = 0;
        const auto parsed = std::stod(str, &pos);
        assertFullyConsumed(str, pos);
        return parsed;
    } catch (...) {
        OPENVINO_THROW("Value '", val, "' is not a valid FP64 option");
    }
}

//
// OptionPrinter
//

std::string OptionPrinter<bool>::toString(bool val) {
    return val ? "YES" : "NO";
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

details::OptionConcept OptionsDesc::get(std::string_view key) const {
    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(std::string(key));
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        _log.warning("Deprecated option '%s' was used, '%s' should be used instead", key.data(), searchKey.c_str());
    }

    const auto itMain = _impl.find(searchKey);
    OPENVINO_ASSERT(itMain != _impl.end(),
                    "[ NOT_FOUND ] Option '",
                    key.data(),
                    "' is not supported for current configuration");

    return itMain->second;
}

void OptionsDesc::reset() {
    _impl.clear();
}

bool OptionsDesc::has(std::string_view key) const {
    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(searchKey);
    if (itDeprecated != _deprecated.end()) {
        return true;
    }
    const auto itMain = _impl.find(searchKey);
    if (itMain != _impl.end()) {
        return true;
    }
    return false;
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

void Config::parseEnvVars() {
    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                _log.trace("Update option '%s' to value '%s' parsed from environment variable '%s'",
                           opt.key().data(),
                           envVar,
                           opt.envVar().data());

                try {
                    _impl[opt.key().data()] = opt.validateAndParse(std::string(envVar));
                } catch (const std::exception& e) {
                    _log.warning(
                        "Environment variable '%s' with value '%s' was ignored for option '%s' due to error:\n%s",
                        opt.envVar().data(),
                        envVar,
                        opt.key().data(),
                        e.what());
                }
            }
        }
    });
}

bool Config::has(std::string key) const {
    return _impl.count(key) != 0;
}

void Config::remove(std::string key) {
    _impl.erase(key);
}

void Config::update(const ConfigMap& options) {
    for (const auto& p : options) {
        _log.trace("Update option '%s' to value '%s'", p.first.c_str(), p.second.c_str());

        const auto opt = _desc->get(p.first);
        _impl[opt.key().data()] = opt.validateAndParse(p.second);
    }
}

void Config::update(std::string_view key, const ov::Any& value) {
    _log.trace("Update option '%s'", std::string(key).c_str());

    const auto opt = _desc->get(key);
    _impl[opt.key().data()] = opt.validateAndParse(value);
}

void Config::updateAny(std::string_view key, const ov::Any& value) {
    _log.trace("Update option '%s' to given 'ov::Any' value", std::string(key).c_str());

    const auto opt = _desc->get(key);
    _impl[opt.key().data()] = opt.validateAndParse(value);
}

std::string Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto& key = it->first;

        // include only enabled configs
        resultStream << key << "=\"" << it->second->toString() << "\"";
        if (std::next(it) != _impl.end()) {
            resultStream << " ";
        }
    }

    return resultStream.str();
}

void Config::fromString(const std::string& str) {
    std::string str_cfg(str);

    auto parse_token = [&](const std::string& token) {
        auto pos_eq = token.find('=');
        auto key = token.substr(0, pos_eq);
        auto value = token.substr(pos_eq + 2, token.size() - pos_eq - 3);
        update(key, value);
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
}

bool Config::hasOpt(std::string_view key) const {
    return _desc->has(key);
}

details::OptionConcept Config::getOpt(std::string_view key) const {
    return _desc->get(key);
}

void Config::addOrUpdateInternal(std::string key, std::string value) {
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

bool Config::hasInternal(std::string_view key) const {
    return _internal_compiler_configs.count(std::string(key)) != 0;
}

void Config::removeCompileTimeConfigs() {
    for (auto it = _impl.begin(); it != _impl.end();) {
        if (_desc->get(it->first).mode() == OptionMode::CompileTime) {
            it = _impl.erase(it);
        } else {
            ++it;
        }
    }

    _internal_compiler_configs.clear();
}

std::string Config::getInternal(std::string key) const {
    if (_internal_compiler_configs.count(key) == 0) {
        OPENVINO_THROW(std::string("Internal compiler option " + key + " does not exist! "));
    }
    return _internal_compiler_configs.at(key);
}

std::string Config::toStringForCompiler(const std::function<bool(const std::string&)>& isSupported) const {
    if (!isSupported) {
        OPENVINO_THROW("Config::toStringForCompiler requires a valid support predicate");
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
