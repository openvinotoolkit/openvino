// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_istreams_executor.hpp>
#include <ie_performance_hints.hpp>
#include <ie/ie_common.h>
#include <openvino/util/common_util.hpp>
#include "utils/debug_capabilities.h"

#include <bitset>
#include <string>
#include <map>

namespace ov {
namespace intel_cpu {

struct Config {
    Config();

    enum LPTransformsMode {
        Off,
        On,
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    int batchLimit = 0;
    size_t rtCacheCapacity = 5000ul;
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
#if defined(__arm__) || defined(__aarch64__)
    // Currently INT8 mode is not optimized on ARM, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
    bool enforceBF16 = false;
#else
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
    bool enforceBF16 = true;
    bool manualEnforceBF16 = false;
#endif

    std::string cache_dir{};

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;

#ifdef CPU_DEBUG_CAPS

private:
    struct PropertySetter {
        virtual bool parseAndSet(const std::string& str) = 0;
        virtual std::string getPropertyValueDescription(void) const = 0;

        PropertySetter(const std::string&& name) : propertyName(name) {}
        const std::string& getPropertyName(void) const { return propertyName; }

    private:
        const std::string propertyName;
    };
    using PropertySetterPtr = std::shared_ptr<PropertySetter>;
    struct StringPropertySetter : PropertySetter {
        StringPropertySetter(const std::string&& name, std::string& ref, const std::string&& valueDescription)
            : property(ref), propertyValueDescription(valueDescription), PropertySetter(std::move(name)) {}
        bool parseAndSet(const std::string& str) override {
            property = str;
            return true;
        }
        std::string getPropertyValueDescription(void) const override { return propertyValueDescription; }

    private:
        std::string& property;
        const std::string propertyValueDescription;
    };
    template<std::size_t NumOfBits>
    struct BitsetFilterPropertySetter : PropertySetter {
        struct Token {
            std::string name;
            std::vector<size_t> bits;
        };

        BitsetFilterPropertySetter(const std::string&& name, std::bitset<NumOfBits>& ref, const std::vector<Token>&& tokens)
            : property(ref), propertyTokens(tokens), PropertySetter(std::move(name)) {}
        bool parseAndSet(const std::string& str) override {
            const auto& tokens = str.empty() ?
                std::vector<std::string>{"all"} : ov::util::split(ov::util::to_lower(str), ',');
            property.reset();
            for (const auto& token : tokens) {
                const bool tokenVal = (token.front() != '-');
                const auto& tokenName = tokenVal ? token : token.substr(1);
                const auto& foundToken = std::find_if(propertyTokens.begin(), propertyTokens.end(),
                    [tokenName] (const Token& token) { return token.name == tokenName; });
                if (foundToken == propertyTokens.end())
                    return false;

                for (const auto& bit : foundToken->bits) {
                    property.set(bit, tokenVal);
                }
            }
            return true;
        }
        std::string getPropertyValueDescription(void) const override {
            std::string supportedTokens = "comma separated filter tokens: ";
            for (auto i = 0; i < propertyTokens.size(); i++) {
                if (i)
                    supportedTokens.push_back(',');
                supportedTokens.append(propertyTokens[i].name);
            }
            supportedTokens.append("; -'token' is used for exclusion, case does not matter, no tokens is treated as 'all'");
            return supportedTokens;
        }

    private:
        std::bitset<NumOfBits>& property;
        const std::vector<Token> propertyTokens;
    };
    struct PropertyGroup {
        virtual std::vector<PropertySetterPtr> getPropertySetters(void) = 0;

        void parseAndSet(const std::string& str) {
            const auto& options = ov::util::split(str, ' ');
            const auto& propertySetters = getPropertySetters();
            bool failed = false;
            auto getHelp = [propertySetters] (void) {
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
                const auto& foundSetter = std::find_if(propertySetters.begin(), propertySetters.end(),
                    [propertyName] (const PropertySetterPtr& setter) { return setter->getPropertyName() == propertyName; });
                if (foundSetter == propertySetters.end() ||
                    !(*foundSetter)->parseAndSet(parts.size() == 1 ? "" : parts.back())) {
                    failed = true;
                    break;
                }
            }

            if (failed)
                IE_THROW() << "Wrong syntax: " << str << std::endl
                           << "The following space separated options are supported (option names are case insensitive):" << std::endl
                           << getHelp();
        }
    };

public:
    struct TransformationFilter {
        enum Type : uint8_t {
            PreLpt = 0, Lpt, PostLpt, Snippets, Specific, NumOfTypes
        };
        std::bitset<NumOfTypes> filter;

        PropertySetterPtr getPropertySetter() {
            return PropertySetterPtr(new BitsetFilterPropertySetter<NumOfTypes>("transformations", filter,
                {{"all", {PreLpt, Lpt, PostLpt, Snippets, Specific}},
                 {"common", {PreLpt, PostLpt}},
                 {"prelpt", {PreLpt}},
                 {"lpt", {Lpt}},
                 {"postlpt", {PostLpt}},
                 {"snippets", {Snippets}},
                 {"specific", {Specific}}
                }));
        }
    };
    struct IrFormatFilter {
        enum Type : uint8_t {
            Xml = 0, Dot, Svg, NumOfTypes
        };
        std::bitset<NumOfTypes> filter;

        PropertySetterPtr getPropertySetter() {
            return PropertySetterPtr(new BitsetFilterPropertySetter<NumOfTypes>("formats", filter,
                {{"all", {Xml, Dot, Svg}},
                 {"xml", {Xml}},
                 {"dot", {Dot}},
                 {"svg", {Svg}},
                }));
        }
    };

    enum FILTER {
        BY_PORTS,
        BY_EXEC_ID,
        BY_TYPE,
        BY_NAME,
    };

    enum class FORMAT {
        BIN,
        TEXT,
    };

    std::string execGraphPath;
    std::string verbose;
    std::string blobDumpDir = "cpu_dump";
    FORMAT blobDumpFormat = FORMAT::TEXT;
    // std::hash<int> is necessary for Ubuntu-16.04 (gcc-5.4 and defect in C++11 standart)
    std::unordered_map<FILTER, std::string, std::hash<int>> blobDumpFilters;

    struct : PropertyGroup {
        TransformationFilter transformations;

        std::vector<PropertySetterPtr> getPropertySetters(void) override {
            return { transformations.getPropertySetter() };
        }
    } disable;

    struct : PropertyGroup {
        std::string dir = "intel_cpu_dump";
        IrFormatFilter format = { 1 << IrFormatFilter::Dot };
        TransformationFilter transformations;

        std::vector<PropertySetterPtr> getPropertySetters(void) override {
            return { PropertySetterPtr(new StringPropertySetter("dir", dir, "path to dumped IRs")),
                     format.getPropertySetter(),
                     transformations.getPropertySetter() };
        }
    } dumpIR;

    void readDebugCapsProperties();
#endif

    bool isNewApi = true;
};

}   // namespace intel_cpu
}   // namespace ov
