// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <stdexcept>
#include <chrono>

#include <vpu/vpu_plugin_config.hpp>

#include <vpu/utils/logger.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

VPU_DECLARE_ENUM(ConfigMode,
    Any,
    RunTime,
)

class ParsedConfigBase {
public:
    LogLevel logLevel() const {
        return _logLevel;
    }

    bool exclusiveAsyncRequests() const {
        return _exclusiveAsyncRequests;
    }

public:
    ParsedConfigBase();
    virtual ~ParsedConfigBase();

    void update(
            const std::map<std::string, std::string>& config,
            ConfigMode mode = ConfigMode::Any);

protected:
    virtual const std::unordered_set<std::string>& getCompileOptions() const;
    virtual const std::unordered_set<std::string>& getRunTimeOptions() const;
    virtual const std::unordered_set<std::string>& getDeprecatedOptions() const;
    virtual void parse(const std::map<std::string, std::string>& config);

protected:
    static std::unordered_set<std::string> merge(
                const std::unordered_set<std::string>& set1,
                const std::unordered_set<std::string>& set2);

    static void setOption(
                std::string& dst,
                const std::map<std::string, std::string>& config,
                const std::string& key);

    template <typename T, class SupportedMap>
    static void setOption(
                T& dst,
                const SupportedMap& supported,
                const std::map<std::string, std::string>& config,
                const std::string& key) {
        const auto value = config.find(key);
        if (value != config.end()) {
            const auto parsedValue = supported.find(value->second);
            if (parsedValue == supported.end()) {
                THROW_IE_EXCEPTION
                        << "Unsupported value " << "\"" << value->second << "\""
                        << " for key " << key;
            }

            dst = parsedValue->second;
        }
    }

    template <typename T, class PreprocessFunc>
    static void setOption(
                T& dst,
                const std::map<std::string, std::string>& config,
                const std::string& key,
                const PreprocessFunc& preprocess) {
        const auto value = config.find(key);
        if (value != config.end()) {
            try {
                dst = preprocess(value->second);
            } catch(const std::exception& e) {
                THROW_IE_EXCEPTION
                        << "Invalid value " << "\"" << value->second << "\""
                        << " for key " << key
                        << " : " << e.what();
            }
        }
    }

    static std::chrono::seconds parseSeconds(const std::string& src) {
        try {
            return std::chrono::seconds(std::stoi(src));
        } catch (const std::exception& e) {
            THROW_IE_EXCEPTION
                        << "Can not convert string:"
                        << src << " to seconds. "
                        << "Message : " << e.what();
        }
    }

    static int parseInt(const std::string& src) {
        const auto val = std::stoi(src);

        return val;
    }

    static float parseFloat(const std::string& src) {
        return std::stof(src);
    }

    static float parseFloatReverse(const std::string& src) {
        const auto val = std::stof(src);
        if (val == 0.0f) {
            throw std::invalid_argument("Zero value");
        }
        return 1.0f / val;
    }

protected:
    static const std::unordered_map<std::string, bool> switches;

    Logger::Ptr _log;

private:
    LogLevel _logLevel = LogLevel::None;
    bool _exclusiveAsyncRequests = false;
};

}  // namespace vpu
