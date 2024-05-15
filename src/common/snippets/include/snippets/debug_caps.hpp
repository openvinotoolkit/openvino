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
        enum Type : uint8_t {
            controlFlow = 0, dataFlow, NumOfTypes
        };
        std::bitset<NumOfTypes> filter;

        PropertySetterPtr getPropertySetter() {
            return PropertySetterPtr(new BitsetFilterPropertySetter<NumOfTypes>("formats", filter, {{"all", {controlFlow, dataFlow}},
                                                                                                    {"control_flow", {controlFlow}},
                                                                                                    {"data_flow", {dataFlow}},
                                                                                                   }));
        }
    };

    struct PropertyGroup {
        virtual std::vector<PropertySetterPtr> getPropertySetters() = 0;

        void parseAndSet(const std::string& str) {
            const auto& options = ov::util::split(str, ' ');
            const auto& propertySetters = getPropertySetters();
            bool failed = false;
            auto getHelp = [propertySetters]() {
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
                OPENVINO_THROW(
                    "Wrong syntax: ",
                    str,
                    "\n",
                    "The following space separated options are supported (option names are case insensitive):",
                    "\n",
                    getHelp());
        }
    };

    struct : PropertyGroup {
        std::string dir = "intel_snippets_LIR_dump";
        LIRFormatFilter format = { 1 << LIRFormatFilter::controlFlow };
        std::string passes = "";

        std::vector<PropertySetterPtr> getPropertySetters() override {
            return { PropertySetterPtr(new StringPropertySetter("dir", dir, "path to dumped LIRs")),
                     format.getPropertySetter(),
                     PropertySetterPtr(new StringPropertySetter("passes", passes, "indicate dumped LIRs around these passes"))};
        }
    } dumpLIR;

private:
    struct PropertySetter {
        virtual bool parseAndSet(const std::string& str) = 0;
        virtual std::string getPropertyValueDescription() const = 0;

        PropertySetter(std::string name) : propertyName(std::move(name)) {}

        virtual ~PropertySetter() = default;

        const std::string& getPropertyName() const { return propertyName; }

    private:
        const std::string propertyName;
    };

    struct StringPropertySetter : PropertySetter {
        StringPropertySetter(const std::string& name, std::string& ref, const std::string&& valueDescription)
            : PropertySetter(name), property(ref), propertyValueDescription(valueDescription) {}

        ~StringPropertySetter() override = default;

        bool parseAndSet(const std::string& str) override {
            property = str;
            return true;
        }
        std::string getPropertyValueDescription() const override { return propertyValueDescription; }

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

        BitsetFilterPropertySetter(const std::string& name, std::bitset<NumOfBits>& ref, const std::vector<Token>&& tokens)
            : PropertySetter(name), property(ref), propertyTokens(tokens) {}

        ~BitsetFilterPropertySetter() override = default;

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
        std::string getPropertyValueDescription() const override {
            std::string supportedTokens = "comma separated filter tokens: ";
            for (size_t i = 0; i < propertyTokens.size(); i++) {
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

    void readProperties();
};

}   // namespace snippets
}   // namespace ov

#endif // SNIPPETS_DEBUG_CAPS
