// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/config.hpp"

#include <cctype>
#include <charconv>
#include <string_view>

#include "openvino/util/common_util.hpp"

namespace intel_npu {

namespace {

// `std::from_chars` does the range checking for the exact target type on its own (unlike the `std::sto*`
// functions, which either need a wider signed intermediate or silently wrap negative values around), but it
// neither skips leading whitespace nor reports anything about the characters left after the number. Hence
// the surrounding whitespace is trimmed upfront and the parse is required to reach the end of the value, so
// that a valid prefix alone ("12oops") is not enough to accept it.
template <typename T>
T parseNumber(std::string_view val, std::string_view typeName) {
    auto trimmed = ov::util::trim(val);
    // Unlike `std::from_chars`, the `std::sto*` functions used before accepted an explicit plus sign. Only a
    // plus directly followed by a digit is dropped, so that "+-1" is still rejected instead of being turned
    // into a well formed "-1".
    if (trimmed.size() > 1 && trimmed.front() == '+' && std::isdigit(static_cast<unsigned char>(trimmed[1]))) {
        trimmed.remove_prefix(1);
    }

    T parsed{};
    // `ov::util::trim` hands back a default constructed view for a blank value, whose `data()` is null
    auto errorCode = std::errc::invalid_argument;
    auto parseEnd = trimmed.data();
    if (!trimmed.empty()) {
        const auto parseResult = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), parsed);
        parseEnd = parseResult.ptr;
        errorCode = parseResult.ec;
    }

    OPENVINO_ASSERT(errorCode == std::errc{} && parseEnd == trimmed.data() + trimmed.size(),
                    "Value '",
                    val,
                    "' is not a valid ",
                    typeName,
                    " option");

    return parsed;
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
        return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    });
    if (strVal == "YES" || strVal == "TRUE" || strVal == "ON" || strVal == "1") {
        return true;
    } else if (strVal == "NO" || strVal == "FALSE" || strVal == "OFF" || strVal == "0") {
        return false;
    }

    OPENVINO_THROW("Value '", val.data(), "' is not a valid BOOL option");
}

int32_t OptionParser<int32_t>::parse(std::string_view val) {
    return parseNumber<int32_t>(val, "INT32");
}

uint32_t OptionParser<uint32_t>::parse(std::string_view val) {
    // Note: `std::from_chars` rejects negative values for unsigned types instead of wrapping them around
    return parseNumber<uint32_t>(val, "UINT32");
}

int64_t OptionParser<int64_t>::parse(std::string_view val) {
    return parseNumber<int64_t>(val, "INT64");
}

uint64_t OptionParser<uint64_t>::parse(std::string_view val) {
    return parseNumber<uint64_t>(val, "UINT64");
}

double OptionParser<double>::parse(std::string_view val) {
    // `std::from_chars` has no floating point overload in every supported standard library, so `std::stod` is
    // used here. Same as in `parseNumber`, the surrounding whitespace is trimmed upfront and the parse is
    // required to reach the end of the value, so that a valid prefix alone ("1.5oops") is not enough.
    try {
        const auto trimmed = ov::util::trim(val);
        // `ov::util::trim` hands back a default constructed view for a blank value, whose `data()` is null
        OPENVINO_ASSERT(!trimmed.empty(), "Value '", val, "' holds no number at all");

        const std::string str(trimmed);
        size_t pos = 0;
        const auto parsed = std::stod(str, &pos);
        OPENVINO_ASSERT(pos == str.size(), "Value '", val, "' has leftover characters after the number");
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
    _deprecated.clear();
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

bool Config::has(std::string_view key) const {
    return _impl.count(key) != 0;
}

void Config::remove(std::string_view key) {
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
    const auto [it, inserted] = _internal_compiler_configs.insert_or_assign(std::move(key), std::move(value));
    if (inserted) {
        _log.trace("Store internal compiler option %s: %s", it->first.c_str(), it->second.c_str());
    } else {
        _log.warning("Internal compiler option '%s' was already registered! Updating value only!", it->first.c_str());
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

std::string Config::getInternal(std::string_view key) const {
    // `ConfigMap` is keyed by `std::string` and uses the default comparator, so it has no heterogeneous lookup
    const auto it = _internal_compiler_configs.find(std::string(key));
    if (it == _internal_compiler_configs.end()) {
        OPENVINO_THROW("Internal compiler option ", key, " does not exist!");
    }

    return it->second;
}

std::string Config::toStringForCompiler(const std::function<bool(const std::string&)>& isSupported) const {
    if (!isSupported) {
        OPENVINO_THROW("Config::toStringForCompiler requires a valid support predicate");
    }

    std::stringstream resultStream;
    bool hasSerializedValue = false;

    const auto append = [&](std::string_view key, std::string_view serializedValue) {
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

        append(key, value->toString());
    }

    for (const auto& [key, value] : _internal_compiler_configs) {
        if (!isSupported(key)) {
            OPENVINO_THROW("[ NOT_FOUND ] Option '" + key + "' is not supported for current configuration");
        }
        append(key, value);
    }

    return resultStream.str();
}

}  // namespace intel_npu
