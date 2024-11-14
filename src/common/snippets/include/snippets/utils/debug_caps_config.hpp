// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "openvino/util/common_util.hpp"
#include "openvino/core/except.hpp"

#include <bitset>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ov {
namespace snippets {

class DebugCapsConfig {
private:
    struct PropertySetter;
    using PropertySetterPtr = std::shared_ptr<PropertySetter>;

public:
    DebugCapsConfig() {
        readProperties();
    }

    struct LIRFormatFilter {
        enum Type : uint8_t { controlFlow = 0, dataFlow, NumOfTypes };
        std::bitset<NumOfTypes> filter;

        PropertySetterPtr getPropertySetter() {
            return PropertySetterPtr(new BitsetFilterPropertySetter<NumOfTypes>("formats",
                                                                                filter,
                                                                                {
                                                                                    {"all", {controlFlow, dataFlow}},
                                                                                    {"control_flow", {controlFlow}},
                                                                                    {"data_flow", {dataFlow}},
                                                                                }));
        }
    };

    struct PropertyGroup {
        virtual std::vector<PropertySetterPtr> getPropertySetters() = 0;
        void parseAndSet(const std::string& str);
    };

    struct : PropertyGroup {
        std::string dir = "snippets_LIR_dump";
        LIRFormatFilter format = {1 << LIRFormatFilter::controlFlow};
        std::vector<std::string> passes;

        std::vector<PropertySetterPtr> getPropertySetters() override {
            return {PropertySetterPtr(new StringPropertySetter("dir", dir, "path to dumped LIRs")),
                    format.getPropertySetter(),
                    PropertySetterPtr(new MultipleStringPropertySetter("passes", passes,
                    "indicate dump LIRs around the passes. Support multiple passes with comma separated and case insensitive. 'all' means dump all passes"))};
        }
    } dumpLIR;

    // Snippets performance count mode
    // Disabled - default, w/o perf count for snippets
    // Chrono - perf count with chrono call. This is a universal method, and support multi-thread case to output perf
    // count data for each thread. BackendSpecific - perf count provided by backend. This is for device specific
    // requirment. For example, in sake of more light overhead and more accurate result, x86 CPU specific mode via read
    // RDTSC register is implemented, which take ~50ns, while Chrono mode take 260ns for a pair of perf count start and
    // perf count end execution, on ICX. This mode only support single thread.
    enum PerfCountMode {
        Disabled,
        Chrono,
        BackendSpecific,
    };
    PerfCountMode perf_count_mode = PerfCountMode::Disabled;

private:
    struct PropertySetter {
        PropertySetter(std::string name) : propertyName(std::move(name)) {}
        virtual ~PropertySetter() = default;
        virtual bool parseAndSet(const std::string& str) = 0;
        virtual std::string getPropertyValueDescription() const = 0;
        const std::string& getPropertyName() const {
            return propertyName;
        }

    private:
        const std::string propertyName;
    };

    struct StringPropertySetter : PropertySetter {
        StringPropertySetter(const std::string& name, std::string& ref, const std::string&& valueDescription)
            : PropertySetter(name),
              property(ref),
              propertyValueDescription(valueDescription) {}
        ~StringPropertySetter() override = default;

        bool parseAndSet(const std::string& str) override {
            property = str;
            return true;
        }
        std::string getPropertyValueDescription() const override {
            return propertyValueDescription;
        }

    private:
        std::string& property;
        const std::string propertyValueDescription;
    };

    struct MultipleStringPropertySetter : PropertySetter {
        MultipleStringPropertySetter(const std::string& name, std::vector<std::string>& ref, const std::string&& valueDescription)
            : PropertySetter(name),
              propertyValues(ref),
              propertyValueDescription(valueDescription) {}
        ~MultipleStringPropertySetter() override = default;

        bool parseAndSet(const std::string& str) override {
            propertyValues = ov::util::split(ov::util::to_lower(str), ',');
            return true;
        }

        std::string getPropertyValueDescription() const override {
            return propertyValueDescription;
        }

    private:
        std::vector<std::string>& propertyValues;
        const std::string propertyValueDescription;
    };

    template <std::size_t NumOfBits>
    struct BitsetFilterPropertySetter : PropertySetter {
        struct Token {
            std::string name;
            std::vector<size_t> bits;
        };

        BitsetFilterPropertySetter(const std::string& name,
                                   std::bitset<NumOfBits>& ref,
                                   const std::vector<Token>&& tokens)
            : PropertySetter(name),
              property(ref),
              propertyTokens(tokens) {}

        ~BitsetFilterPropertySetter() override = default;

        bool parseAndSet(const std::string& str) override {
            const auto& tokens =
                str.empty() ? std::vector<std::string>{"all"} : ov::util::split(ov::util::to_lower(str), ',');
            property.reset();
            for (const auto& token : tokens) {
                const bool tokenVal = (token.front() != '-');
                const auto& tokenName = tokenVal ? token : token.substr(1);
                const auto& foundToken =
                    std::find_if(propertyTokens.begin(), propertyTokens.end(), [tokenName](const Token& token) {
                        return token.name == tokenName;
                    });
                if (foundToken == propertyTokens.end())
                    return false;

                for (const auto& bit : foundToken->bits) {
                    property.set(bit, tokenVal);
                }
            }
            return true;
        }
        std::string getPropertyValueDescription() const override {
            std::string supportedTokens = "comma separated filter tokens: ";
            for (size_t i = 0; i < propertyTokens.size(); i++) {
                if (i)
                    supportedTokens.push_back(',');
                supportedTokens.append(propertyTokens[i].name);
            }
            supportedTokens.append(
                "; -'token' is used for exclusion, case does not matter, no tokens is treated as 'all'");
            return supportedTokens;
        }

    private:
        std::bitset<NumOfBits>& property;
        const std::vector<Token> propertyTokens;
    };

    void readProperties();
};

}  // namespace snippets
}  // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
