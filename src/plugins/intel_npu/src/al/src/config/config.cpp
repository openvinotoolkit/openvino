// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/config.hpp"

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

        const auto opt = _desc->get(p.first, mode);
        _impl[opt.key().data()] = opt.validateAndParse(p.second);
    }
}

std::string Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto key = it->first;

        resultStream << key << "=\"" << it->second->toString() << "\"";
        if (std::next(it) != _impl.end()) {
            resultStream << " ";
        }
    }

    return resultStream.str();
}

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
