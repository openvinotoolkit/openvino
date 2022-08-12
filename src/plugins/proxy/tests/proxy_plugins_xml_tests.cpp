// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <regex>

#include "file_utils.h"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

class PluginXMLGenerator {
public:
    PluginXMLGenerator()
        : file_path(InferenceEngine::getIELibraryPath() + ov::util::FileTraits<char>::file_separator +
                    "test_proxy_plugin.xml") {}
    PluginXMLGenerator(const std::string& file_name)
        : file_path(InferenceEngine::getIELibraryPath() + ov::util::FileTraits<char>::file_separator + file_name +
                    ".xml") {}

    ~PluginXMLGenerator() {
        if (generated) {
            std::remove(file_path.c_str());
        }
    }

    static std::string get_plugin_location(const std::string& plugin_name) {
        return ov::util::FileTraits<char>::library_prefix() + plugin_name + std::string(IE_BUILD_POSTFIX) +
               ov::util::FileTraits<char>::dot_symbol + ov::util::FileTraits<char>::library_ext();
    }

    void build_and_load(ov::Core& core) {
        {
            std::ofstream xml_config(file_path);
            xml_config << "<ie>\n    <plugins>\n";
            for (const auto& config : plugin_configs) {
                xml_config << config;
            }
            xml_config << "    </plugins>\n</ie>";
            generated = true;
        }
        core.register_plugins(file_path);
    }

    void add_plugin_config(const std::string& plugin_name,
                           const std::string& plugin_location,
                           const std::vector<std::string>& plugin_config = {}) {
        std::string config = "        <plugin name=\"" + plugin_name + "\" location=\"" + plugin_location + "\">\n";
        if (!plugin_config.empty()) {
            config += "            <properties>\n";
            for (const auto& property : plugin_config) {
                config += "                " + property + "\n";
            }
            config += "            </properties>\n";
        }
        config += "        </plugin>\n";
        plugin_configs.emplace_back(config);
    }

private:
    std::string file_path;
    bool generated = false;
    std::vector<std::string> plugin_configs;
};

class PluginsXmlProxyTests : public ProxyTests {
public:
    void SetUp() override {}
    PluginXMLGenerator xml_builder;
};

TEST_F(PluginsXmlProxyTests, MatchingModeWithFallback) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="MOCK" location="libmock_abc_plugind.so">
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="FALLBACK" value="MOCK">
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    xml_builder.add_plugin_config("MOCK", PluginXMLGenerator::get_plugin_location("mock_abc_plugin"));
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"FALLBACK\" value=\"MOCK\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    // No ALIAS HERE
    std::set<std::string> mock_reference_dev = {"MOCK.abc_a", "MOCK.abc_b", "MOCK.abc_c", "BDE.0", "BDE.1", "BDE.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
        EXPECT_NE(dev, "BDE.3");
        EXPECT_NE(dev, "BDE.4");
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(PluginsXmlProxyTests, MatchingModeWithFallbackWithAlias) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="ABC" location="libmock_abc_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //             </properties>
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //                 <property key="FALLBACK" value="ABC">
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    // default key="" value=""
    xml_builder.add_plugin_config("ABC", PluginXMLGenerator::get_plugin_location("mock_abc_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>"
    });
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>",
        "<property key=\"FALLBACK\" value=\"ABC\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());

    auto model = create_model_with_subtract();
    EXPECT_THROW(core.compile_model(model, "MOCK.0"), ov::Exception);
}

TEST_F(PluginsXmlProxyTests, MatchingModeWithFallbackWithAliasAndMainPriority) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="ABC" location="libmock_abc_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //                 <property key="DEVICE_PRIORITY" value="0">
    //             </properties>
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //                 <property key="FALLBACK" value="ABC">
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    // default key="" value=""
    xml_builder.add_plugin_config("ABC", PluginXMLGenerator::get_plugin_location("mock_abc_plugin"), {
        "<property key=\"DEVICE_PRIORITY\" value=\"0\"/>",
        "<property key=\"ALIAS\" value=\"MOCK\"/>"
    });
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>",
        "<property key=\"FALLBACK\" value=\"ABC\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());

    auto model = create_model_with_subtract();
    EXPECT_THROW(core.compile_model(model, "MOCK.0"), ov::Exception);
}

TEST_F(PluginsXmlProxyTests, MatchingModeWithFallbackWithAliasAndPriority) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="ABC" location="libmock_abc_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //                 <property key="DEVICE_PRIORITY" value="3">
    //             </properties>
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK">
    //                 <property key="FALLBACK" value="ABC">
    //                 <property key="DEVICE_PRIORITY" value="2">
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    // default key="" value=""
    xml_builder.add_plugin_config("ABC", PluginXMLGenerator::get_plugin_location("mock_abc_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>",
        "<property key=\"DEVICE_PRIORITY\" value=\"3\"/>"
    });
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>",
        "<property key=\"FALLBACK\" value=\"ABC\"/>",
        "<property key=\"DEVICE_PRIORITY\" value=\"2\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());

    auto model = create_model_with_subtract();
    EXPECT_NO_THROW(core.compile_model(model, "MOCK.0"));
}

TEST_F(PluginsXmlProxyTests, MatchingModeWithFallbackWithAliasForPluginName) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="ABC" location="libmock_abc_plugind.so">
    //             <properties>
    //                 <property key="ALIAS" value="MOCK"/>
    //             </properties>
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="FALLBACK" value="ABC"/>
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    // default key="" value=""
    xml_builder.add_plugin_config("ABC", PluginXMLGenerator::get_plugin_location("mock_abc_plugin"), {
        "<property key=\"ALIAS\" value=\"MOCK\"/>"
    });
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"FALLBACK\" value=\"ABC\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "BDE.0", "BDE.1", "BDE.2"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(PluginsXmlProxyTests, MatchingModeWithHetero) {
    // clang-format off
    // <ie>
    //     <plugins>
    //         <plugin name="FGH" location="libmock_fgh_plugind.so">
    //         </plugin>
    //         <plugin name="BDE" location="libmock_bde_plugind.so">
    //             <properties>
    //                 <property key="FALLBACK" value="FGH">
    //             </properties>
    //         </plugin>
    //     </plugins>
    // </ie>
    // FGH - no uuid, BDE devices ->Fallback FGH
    // default key="" value=""
    xml_builder.add_plugin_config("FGH", PluginXMLGenerator::get_plugin_location("mock_fgh_plugin")
    );
    xml_builder.add_plugin_config("BDE", PluginXMLGenerator::get_plugin_location("mock_bde_plugin"), {
        "<property key=\"FALLBACK\" value=\"FGH\"/>"
    });
    // clang-format on
    xml_builder.build_and_load(core);

    auto available_devices = core.get_available_devices();
    // 0, 1, 2 is ABC plugin
    // 1, 3, 4 is BDE plugin
    // ABC doesn't support subtract operation
    std::set<std::string> mock_reference_dev = {"BDE.0", "BDE.1", "BDE.2", "FGH.fgh_f", "FGH.fgh_g", "FGH.fgh_h"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}
// clang-format off
// <ie>
//     <plugins>
//         <plugin name="ABC" location="libmock_abc_plugind.so">
//             <properties>
//                 <property key="ALIAS" value="MOCK"/>
//             </properties>
//         </plugin>
//         <plugin name="BDE" location="libmock_bde_plugind.so">
//             <properties>
//                 <property key="ALIAS" value="MOCK_2"/>
//                 <property key="FALLBACK" value="ABC"/>
//             </properties>
//         </plugin>
//     </plugins>
// </ie>
// default key="" value=""
//
// CUDA -> Intel_GPU -> Intel_CPU
//
// GPU  -> GPU       -> CPU        // aliases
// (aliases-equal? match uuid) -> Hetero CPU because alias is different
// Hide all GPUs under alias, but don't hide CPU
//
// {CUDA Intel_GPU} -> GPU, {Intel_CPU, ARM} -> CPU hide all devices with aliases
//
//
// If we want to to run only CUDA without callback
//
// Introduce new property for plugin if option is not supported it means native
//
// Add runtime property of proxy plugin to configure fallback queue
//
// Proxy plugin should provide an option for devices_priority
