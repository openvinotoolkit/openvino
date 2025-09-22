// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <intel_npu/utils/logger/logger.hpp>
#include <regex>
#include <string>
#include <vector>

#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/util/xml_parse_utils.hpp"

class BackendName {
public:
    BackendName() {
        const auto corePtr = ov::test::utils::PluginCache::get().core();
        if (corePtr != nullptr) {
            _name = getBackendName(*corePtr);
        } else {
            _log.error("Failed to get OpenVINO Core from cache!");
        }
    }

    std::string getName() const {
        return _name;
    }

    bool isEmpty() const noexcept {
        return _name.empty();
    }

    bool isZero() const {
        return _name == "LEVEL0";
    }

private:
    std::string _name;
    intel_npu::Logger _log = intel_npu::Logger("BackendName", ov::log::Level::INFO);
};

class AvailableDevices {
public:
    AvailableDevices() {
        const auto corePtr = ov::test::utils::PluginCache::get().core();
        if (corePtr != nullptr) {
            _availableDevices = ::getAvailableDevices(*corePtr);
        } else {
            _log.error("Failed to get OpenVINO Core from cache!");
        }

        // Private device names may be registered via environment variables
        const std::string environmentDevice =
            ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::intel_npu::Platform::AUTO_DETECT.data());
        const std::string standardizedEnvironmentDevice = ov::intel_npu::Platform::standardize(environmentDevice);

        if (std::all_of(_availableDevices.begin(), _availableDevices.end(), [&](const std::string& deviceName) {
                return deviceName.find(standardizedEnvironmentDevice) == std::string::npos;
            })) {
            _availableDevices.push_back(standardizedEnvironmentDevice);
        }
    }

    const auto& getAvailableDevices() const {
        return _availableDevices;
    }

    auto count() const {
        return _availableDevices.size();
    }

    bool has3720() const {
        return std::any_of(_availableDevices.begin(), _availableDevices.end(), [](const std::string& deviceName) {
            return deviceName.find("3720") != std::string::npos;
        });
    }

private:
    std::vector<std::string> _availableDevices;
    intel_npu::Logger _log = intel_npu::Logger("AvailableDevices", ov::log::Level::INFO);
};

class CurrentOS {
public:
    CurrentOS() {
#ifdef WIN32
        _name = "windows";
#elif defined(__linux__)
        _name = "linux";
#endif
    }

    std::string getName() const {
        return _name;
    }

    bool isLinux() const {
        return _name == "linux";
    }

    bool isWindows() const {
        return _name == "windows";
    }

private:
    std::string _name;
};

class SkipRegistry {
public:
    void addPatterns(std::string&& comment, std::vector<std::string>&& patternsToSkip) {
        _registry.emplace_back(std::move(comment), std::move(patternsToSkip));
    }

    void addPatterns(bool conditionFlag, std::string&& comment, std::vector<std::string>&& patternsToSkip) {
        if (conditionFlag) {
            addPatterns(std::move(comment), std::move(patternsToSkip));
        }
    }

    /** Searches for the skip pattern to which passed test name matches.
     * Prints the message onto console if pattern is found and the test is to be skipped
     *
     * @param testName name of the current test being matched against skipping
     * @return Suitable skip pattern or empty string if none
     */
    std::string getMatchingPattern(const std::string& testName) const {
        for (const auto& entry : _registry) {
            for (const auto& pattern : entry._patterns) {
                std::regex re(pattern);
                if (std::regex_match(testName, re)) {
                    _log.info("%s; Pattern: %s", entry._comment.c_str(), pattern.c_str());
                    return pattern;
                }
            }
        }

        return std::string{};
    }

private:
    struct Entry {
        Entry(std::string&& comment, std::vector<std::string>&& patterns)
            : _comment{std::move(comment)},
              _patterns{std::move(patterns)} {}

        std::string _comment;
        std::vector<std::string> _patterns;
    };

    std::vector<Entry> _registry;
    intel_npu::Logger _log = intel_npu::Logger("SkipRegistry", ov::log::Level::INFO);
};

std::string getCurrentTestName();

std::string getCurrentTestName() {
    const auto* currentTestInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto currentTestName = currentTestInfo->test_case_name() + std::string(".") + currentTestInfo->name();
    return currentTestName;
}

/** Checks if string containing rule has a "!" character
 * If "!" is found a flag will be set and the rule will
 * have the character erased to be used in further conditions
 *
 * @param rule Input string
 * @return true if "!" is found
 */
bool isRuleInverted(std::string& rule);

bool isRuleInverted(std::string& rule) {
    auto pos = rule.find("!");
    if (pos != std::string::npos) {
        // Delete negation character from rule string
        rule.erase(pos, 1);
        return true;
    }
    return false;
}

/** Reads multiple rules from specified categories:
 *      - "Backend" rule category
 *      - "Device" rule category
 *      - "Operating System" rule category
 *
 *  When a rule is found it will get inverted if it starts with "!"
 *  it will then be checked agains the current system config
 *
 *  If the rule is true,then the skip will be enabled and the test will not run.
 *  If the rule is false, then the skip will be disabled and the test will run.
 *
 *  No rule means skip remains enabled
 *
 * @param category Input category that will be searched for rules
 * @param localSettings Input current system setting, by category
 * @param enableRules xml node to the category that will be checked and read
 * @return true if a rule is found to match current system config
 */
bool categoryRuleEnabler(const std::string& category,
                         const std::vector<std::string>& localSettings,
                         const pugi::xml_node& enableRules);

bool categoryRuleEnabler(const std::string& category,
                         const std::vector<std::string>& localSettings,
                         const pugi::xml_node& enableRules) {
    if (enableRules.child(category.c_str()).empty()) {
        return true;
    }

    FOREACH_CHILD (enableRule, enableRules, category.c_str()) {
        auto categoryRule = enableRule.text().get();

        std::string categoryRuleString(categoryRule);
        bool invert = isRuleInverted(categoryRuleString);
        for (auto& localSetting : localSettings) {
            // Perform logical XOR to invert condition
            if (!(categoryRuleString == localSetting) != !invert) {
                return true;
            }
        }
    }

    return false;
}

std::vector<std::string> disabledTestPatterns();

std::vector<std::string> disabledTestPatterns() {
    // Initialize skip registry
    static const auto skipRegistry = []() {
        SkipRegistry _skipRegistry;

        intel_npu::Logger _log = intel_npu::Logger("SkipConfig", ov::log::Level::INFO);

        const BackendName backendName;
        const AvailableDevices devices;
        const CurrentOS currentOS;

        try {
            const auto& filePath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_SKIP_CONFIG_FILE;
            _log.info("Using %s as skip config", filePath.c_str());

            auto xmlResult = ov::util::pugixml::parse_xml(filePath.c_str());
            // Error returned from pugixml, fallback to legacy skips
            if (!xmlResult.error_msg.empty()) {
                _log.error(xmlResult.error_msg.c_str());
                throw std::runtime_error("No skip filters are applied");
            }

            pugi::xml_document& xmlSkipConfig = *xmlResult.xml;

            // Select the parent node
            pugi::xml_node skipConfigsList = xmlSkipConfig.child("skip_configs");

            // Iterate through each skip rule
            FOREACH_CHILD (skipConfigRule, skipConfigsList, "skip_config") {
                // Extract skip message, it will get printed in the test logs
                auto skipMessageEntry = skipConfigRule.child("message").text().get();

                // Read enable/disable conditions
                // There can be multiple rules for each category
                // If "!" is found, then rule is inverted
                pugi::xml_node enableRules = skipConfigRule.child("enable_rules");
                bool ruleFlag = true;
                if (!enableRules.empty()) {
                    // Accumulate rule for each category
                    ruleFlag &= categoryRuleEnabler("backend", {backendName.getName()}, enableRules);
                    ruleFlag &= categoryRuleEnabler("device", devices.getAvailableDevices(), enableRules);
                    ruleFlag &= categoryRuleEnabler("operating_system", {currentOS.getName()}, enableRules);
                }

                // Select individual filters and add them to the skipRegistry
                pugi::xml_node skipFiltersList = skipConfigRule.child("filters");
                FOREACH_CHILD (skipFilter, skipFiltersList, "filter") {
                    auto skipFilterEntry = skipFilter.text().get();
                    // Add skip to registry
                    _skipRegistry.addPatterns(ruleFlag, skipMessageEntry, {skipFilterEntry});
                }
            }

        } catch (const std::runtime_error& e) {
            // No skip filters to apply
            _log.warning(e.what());
        }        
        return _skipRegistry;
    }();
    // clang-format on

    std::vector<std::string> matchingPatterns;
    const auto currentTestName = getCurrentTestName();
    matchingPatterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matchingPatterns;
}
