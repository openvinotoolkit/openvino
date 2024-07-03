// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <thread>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "dev/core_impl.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/util/file_util.hpp"

using namespace ov::util;

TEST(CoreTests, Throw_on_register_plugin_twice) {
    ov::Core core;
    core.register_plugin("test_plugin", "TEST_DEVICE");
    OV_EXPECT_THROW(core.register_plugin("test_plugin", "TEST_DEVICE"),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests, Throw_on_register_plugins_twice) {
    ov::Core core;

    auto get_plugin_xml = [&]() -> std::string {
        std::string plugins_xml = "test_plugins.xml";
        std::ofstream file(plugins_xml);
        file << "<ie><plugins><plugin location=\"libtest_plugin.so\" name=\"TEST_DEVICE\"></plugin></plugins></ie>";
        file.flush();
        file.close();
        return plugins_xml;
    };

    core.register_plugins(get_plugin_xml());
    OV_EXPECT_THROW(core.register_plugins(get_plugin_xml()),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests_get_plugin_path_from_xml, Use_abs_path_as_is) {
    auto xml_path = "path_to_plugins.xml";
    auto lib_path = ov::util::get_absolute_file_path("test_name.ext");  // CWD/test_name.ext
    for (auto as_abs_only : std::vector<bool>{true, false}) {
        auto abs_path = from_file_path(get_plugin_path(lib_path, xml_path, as_abs_only));
        EXPECT_TRUE(is_absolute_file_path(abs_path));
        EXPECT_STREQ(abs_path.c_str(), lib_path.c_str());
    }
}

TEST(CoreTests_get_plugin_path_from_xml, Convert_relative_path_as_relative_to_xmldir) {
    auto xml_path = "path_to_plugins.xml";
    auto lib_path = ov::util::make_path(std::string("."), std::string("test_name.ext"));  // ./test_name.ext
    for (auto as_abs_only : std::vector<bool>{true, false}) {
        auto abs_path = from_file_path(get_plugin_path(lib_path, xml_path, as_abs_only));  // XMLDIR/test_name.ext
        EXPECT_TRUE(is_absolute_file_path(abs_path));

        auto ref_path = ov::util::get_absolute_file_path(lib_path);
        EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());  // XMLDIR/test_name.ext == CWD/test_name.ext
    }
}

TEST(CoreTests_get_plugin_path_from_xml, Convert_filename_to_abs_path_if_as_abs_only) {
    auto xml_path = "path_to_plugins.xml";
    auto name = "test_name.ext";                                            // test_name.ext
    auto abs_path = from_file_path(get_plugin_path(name, xml_path, true));  // XMLDIR/libtest_name.ext.so
    EXPECT_TRUE(is_absolute_file_path(abs_path));

    auto lib_name = ov::util::make_plugin_library_name({}, std::string(name));
    auto ref_path = ov::util::get_absolute_file_path(lib_name);
    EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());  // XMLDIR/libtest_name.ext.so == CWD/libtest_name.ext.so
}

TEST(CoreTests_get_plugin_path_from_xml, Use_filename_if_not_as_abs_only) {
    auto xml_path = "path_to_plugins.xml";
    auto name = "test_name.ext";                                      // test_name.ext
    auto lib_name = from_file_path(get_plugin_path(name, xml_path));  // libtest_name.ext.so
    auto ref_name = ov::util::make_plugin_library_name({}, std::string(name));
    EXPECT_STREQ(lib_name.c_str(), ref_name.c_str());
}

TEST(CoreTests_get_plugin_path, Use_abs_path_as_is) {
    auto lib_name = ov::util::make_plugin_library_name({}, std::string("test_name"));  // libtest_name.so
    auto lib_path = ov::util::get_absolute_file_path(lib_name);
    auto abs_path = from_file_path(get_plugin_path(lib_path));
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), lib_path.c_str());
}

TEST(CoreTests_get_plugin_path, Relative_path_is_from_workdir) {
    auto lib_name =
        ov::util::make_plugin_library_name(std::string("."), std::string("test_name"));  // ./libtest_name.so
    auto abs_path = from_file_path(get_plugin_path(lib_name));
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), get_absolute_file_path(lib_name).c_str());
}

class CoreTests_get_plugin_path_Class : public ::testing::Test {
public:
    void SetUp() override {
        std::ofstream file(lib_path);
        file << "not empty";
        file.flush();
        file.close();
    }

    void TearDown() override {
        std::remove(lib_path.c_str());
    }

    std::string lib_name = ov::util::make_plugin_library_name({}, std::string("test_name"));  // libtest_name.so
    std::string lib_path = ov::util::get_absolute_file_path(lib_name);                        // CWD/libtest_name.so
};

TEST_F(CoreTests_get_plugin_path_Class, Filename_is_from_workdir_if_exists) {
    auto abs_path = from_file_path(get_plugin_path(lib_name));  // libtest_name.so -> CWD/libtest_name.so
    EXPECT_TRUE(is_absolute_file_path(abs_path));
    EXPECT_STREQ(abs_path.c_str(), get_absolute_file_path(lib_name).c_str());
}

TEST(CoreTests_get_plugin_path, Use_filename_as_is_if_not_exist_in_workdir) {
    auto lib_name = "test_name.ext";
    auto abs_path = from_file_path(get_plugin_path(lib_name));  // libtest_name.ext.so -> libtest_name.ext.so
    EXPECT_FALSE(is_absolute_file_path(abs_path));

    auto ref_path = ov::util::make_plugin_library_name({}, std::string(lib_name));
    EXPECT_STREQ(abs_path.c_str(), ref_path.c_str());
}

TEST(CoreTests_check_device_name, is_config_applicable) {
    // Single device
    ASSERT_EQ(ov::is_config_applicable("DEVICE", "DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.", "DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE", "DEVICE."), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x", "DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x.y", "DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x", "DEVICE.x"), true);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x.y", "DEVICE.x"), true);  // sub-device and device
    ASSERT_EQ(ov::is_config_applicable("DEVICE", "DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x", "DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("DEVICE.x.y", "DEVICE.y"), false);
    // HETERO
    ASSERT_EQ(ov::is_config_applicable("HETERO", "HETERO"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO.", "HETERO"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO", "HETERO."), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE", "HETERO:DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE.x", "HETERO:DEVICE.x"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE", "HETERO"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE.x", "HETERO"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE.x,DEVICE.y", "HETERO:DEVICE.x,DEVICE.y"), true);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE.x", "HETERO:DEVICE.x,DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE.x,DEVICE.y", "HETERO:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("HETERO:DEVICE", "HETERO:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("HETERO", "HETERO:DEVICE"), false);
    // MULTI
    ASSERT_EQ(ov::is_config_applicable("MULTI", "MULTI"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI.", "MULTI"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI", "MULTI."), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE", "MULTI:DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE.x", "MULTI:DEVICE.x"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE", "MULTI"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE.x", "MULTI"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE.x,DEVICE.y", "MULTI:DEVICE.x,DEVICE.y"), true);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE.x", "MULTI:DEVICE.x,DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE.x,DEVICE.y", "MULTI:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("MULTI:DEVICE", "MULTI:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("MULTI", "MULTI:DEVICE"), false);
    // AUTO
    ASSERT_EQ(ov::is_config_applicable("AUTO", "AUTO"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO.", "AUTO"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO", "AUTO."), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE", "AUTO:DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE.x", "AUTO:DEVICE.x"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE.x,DEVICE.y", "AUTO:DEVICE.x,DEVICE.y"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE", "AUTO"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE.x", "AUTO"), true);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE.x", "AUTO:DEVICE.x,DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE.x,DEVICE.y", "AUTO:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("AUTO:DEVICE", "AUTO:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("AUTO", "AUTO:DEVICE"), false);
    // BATCH
    ASSERT_EQ(ov::is_config_applicable("BATCH", "BATCH"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH.", "BATCH"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH", "BATCH."), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE", "BATCH:DEVICE"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE.x", "BATCH:DEVICE.x"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE", "BATCH"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE.x", "BATCH"), true);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE.x", "BATCH:DEVICE.x,DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE.x,DEVICE.y", "BATCH:DEVICE.x"), false);
    ASSERT_EQ(ov::is_config_applicable("BATCH:DEVICE.x", "BATCH:DEVICE.y"), false);
    ASSERT_EQ(ov::is_config_applicable("BATCH", "BATCH:DEVICE"), false);
}

TEST(CoreTests_parse_device_config, get_device_config) {
    auto check_parsed_config = [&](const std::string& device,
                                   const ov::AnyMap& config,
                                   const std::string& expected_device,
                                   const ov::AnyMap& expected_config) {
        auto parsed = ov::parseDeviceNameIntoConfig(device, config);
        ASSERT_EQ(parsed._deviceName, expected_device);
        ASSERT_EQ(ov::Any(parsed._config).as<std::string>(), ov::Any(expected_config).as<std::string>());
    };
    // Single device
    check_parsed_config("DEVICE.0", ov::AnyMap{}, "DEVICE", ov::AnyMap{ov::device::id("0")});
    // simple flattening
    check_parsed_config("DEVICE",
                        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR))},
                        "DEVICE",
                        ov::AnyMap{ov::log::level(ov::log::Level::ERR)});
    // sub-property has flattened, property is kept as is, device_id is moved to property
    check_parsed_config(
        "DEVICE.X",
        ov::AnyMap{ov::num_streams(5), ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR))},
        "DEVICE",
        ov::AnyMap{ov::device::id("X"), ov::num_streams(5), ov::log::level(ov::log::Level::ERR)});
    // explicit device sub-property has higher priority than ov::AnyMap
    check_parsed_config(
        "DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "DEVICE",
        ov::AnyMap{ov::log::level(ov::log::Level::ERR)});
    // property always has higher priority than sub-property
    check_parsed_config(
        "DEVICE",
        ov::AnyMap{ov::log::level(ov::log::Level::DEBUG),
                   ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "DEVICE",
        ov::AnyMap{ov::log::level(ov::log::Level::ERR)});
    // DEVICE.X is not applicable for DEVICE
    check_parsed_config(
        "DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE.X", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "DEVICE",
        ov::AnyMap{ov::log::level(ov::log::Level::WARNING)});
    // properties for another device (for example, MULTI) are dropped
    check_parsed_config("DEVICE",
                        ov::AnyMap{ov::device::properties("MULTI", ov::log::level(ov::log::Level::ERR))},
                        "DEVICE",
                        ov::AnyMap{});

    check_parsed_config("DEVICE.0",
                        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                                   ov::device::properties(
                                       ov::AnyMap{{"DEVICE.0", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
                        "DEVICE",
                        ov::AnyMap{ov::device::id(0), ov::log::level(ov::log::Level::WARNING)});
    check_parsed_config("DEVICE.0.1",
                        ov::AnyMap{ov::device::properties("DEVICE.0.1", ov::log::level(ov::log::Level::INFO)),
                                   ov::device::properties(
                                       ov::AnyMap{{"DEVICE.0", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
                        "DEVICE",
                        ov::AnyMap{ov::device::id("0.1"), ov::log::level(ov::log::Level::INFO)});

    // device ID mismatch
    EXPECT_THROW(ov::parseDeviceNameIntoConfig("DEVICE.X", ov::AnyMap{ov::device::id("Y")}), ov::Exception);

    // HETERO
    check_parsed_config("HETERO:DEVICE", ov::AnyMap{}, "HETERO", ov::AnyMap{ov::device::priorities("DEVICE")});
    check_parsed_config(
        "HETERO:DEVICE",
        ov::AnyMap{ov::device::properties("ANOTHER_DEVICE", ov::log::level(ov::log::Level::ERR))},
        "HETERO",
        ov::AnyMap{
            ov::device::priorities("DEVICE"),
            ov::device::properties(ov::AnyMap{{"ANOTHER_DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    check_parsed_config(
        "HETERO:DEVICE",
        ov::AnyMap{ov::device::properties("HETERO", ov::log::level(ov::log::Level::WARNING)),
                   ov::device::properties("ANOTHER_DEVICE", ov::log::level(ov::log::Level::ERR))},
        "HETERO",
        ov::AnyMap{
            ov::device::priorities("DEVICE"),
            ov::log::level(ov::log::Level::WARNING),
            ov::device::properties(ov::AnyMap{{"ANOTHER_DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    check_parsed_config(
        "HETERO:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::num_streams(5)}}})},
        "HETERO",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(
                       ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR), ov::num_streams(5)}}})});
    check_parsed_config(
        "HETERO:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "HETERO",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    // device priorities mismatch
    EXPECT_THROW(ov::parseDeviceNameIntoConfig("HETERO:DEVICE", ov::AnyMap{ov::device::priorities("ANOTHER_DEVICE")}),
                 ov::Exception);

    // MULTI
    check_parsed_config("MULTI:DEVICE", ov::AnyMap{}, "MULTI", ov::AnyMap{ov::device::priorities("DEVICE")});
    check_parsed_config(
        "MULTI:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR))},
        "MULTI",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    check_parsed_config(
        "MULTI:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::num_streams(5)}}})},
        "MULTI",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(
                       ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR), ov::num_streams(5)}}})});
    check_parsed_config(
        "MULTI:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "MULTI",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});

    // AUTO
    check_parsed_config("AUTO:DEVICE", ov::AnyMap{}, "AUTO", ov::AnyMap{ov::device::priorities("DEVICE")});
    check_parsed_config(
        "AUTO:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR))},
        "AUTO",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    check_parsed_config(
        "AUTO:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::num_streams(5)}}})},
        "AUTO",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(
                       ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR), ov::num_streams(5)}}})});
    check_parsed_config(
        "AUTO:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "AUTO",
        ov::AnyMap{ov::device::priorities("DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});

    // BATCH
    check_parsed_config("BATCH:DEVICE", ov::AnyMap{}, "BATCH", ov::AnyMap{{ov::device::priorities.name(), "DEVICE"}});
    check_parsed_config(
        "BATCH:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR))},
        "BATCH",
        ov::AnyMap{std::make_pair<std::string, ov::Any>(ov::device::priorities.name(), "DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});
    check_parsed_config(
        "BATCH:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::num_streams(5)}}})},
        "BATCH",
        ov::AnyMap{std::make_pair<std::string, ov::Any>(ov::device::priorities.name(), "DEVICE"),
                   ov::device::properties(
                       ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR), ov::num_streams(5)}}})});
    check_parsed_config(
        "BATCH:DEVICE",
        ov::AnyMap{ov::device::properties("DEVICE", ov::log::level(ov::log::Level::ERR)),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::WARNING)}}})},
        "BATCH",
        ov::AnyMap{std::make_pair<std::string, ov::Any>(ov::device::priorities.name(), "DEVICE"),
                   ov::device::properties(ov::AnyMap{{"DEVICE", ov::AnyMap{ov::log::level(ov::log::Level::ERR)}}})});

    // MIX
    check_parsed_config(
        "HETERO",
        ov::AnyMap{ov::device::properties("HETERO", ov::device::priorities("MULTI,DEVICE")),
                   ov::device::properties("MULTI", ov::device::priorities("DEVICE"))},
        "HETERO",
        ov::AnyMap{ov::device::priorities("MULTI,DEVICE"),
                   ov::device::properties(ov::AnyMap{{"MULTI", ov::AnyMap{ov::device::priorities("DEVICE")}}})});

    // invalid device name with characters after parenthesis except comma
    EXPECT_THROW(ov::parseDeviceNameIntoConfig("DEVICE(0)ov", ov::AnyMap{}), ov::Exception);
    EXPECT_THROW(ov::parseDeviceNameIntoConfig("MULTI:DEVICE(0)ov,DEVICE(1)", ov::AnyMap{}), ov::Exception);
    EXPECT_THROW(ov::parseDeviceNameIntoConfig("MULTI:DEVICE(0),DEVICE(1),", ov::AnyMap{}), ov::Exception);
}

TEST(CoreTests_parse_device_config, get_batch_device_name) {
    EXPECT_STREQ(ov::DeviceIDParser::get_batch_device("CPU").c_str(), "CPU");
    EXPECT_STREQ(ov::DeviceIDParser::get_batch_device("GPU(4)").c_str(), "GPU");

    OV_EXPECT_THROW(ov::DeviceIDParser::get_batch_device("-CPU"),
                    ov::Exception,
                    ::testing::HasSubstr("Invalid device name '-CPU' for BATCH"));
    OV_EXPECT_THROW(ov::DeviceIDParser::get_batch_device("CPU(0)-"),
                    ov::Exception,
                    ::testing::HasSubstr("Invalid device name 'CPU(0)-' for BATCH"));
    OV_EXPECT_THROW(ov::DeviceIDParser::get_batch_device("GPU(4),CPU"),
                    ov::Exception,
                    ::testing::HasSubstr("BATCH accepts only one device in list but got 'GPU(4),CPU'"));
    OV_EXPECT_THROW(ov::DeviceIDParser::get_batch_device("CPU,GPU"),
                    ov::Exception,
                    ::testing::HasSubstr("BATCH accepts only one device in list but got 'CPU,GPU'"));
}

class ApplyAutoBatchThreading : public testing::Test {
public:
    static void runParallel(std::function<void(void)> func,
                            const unsigned int iterations = 50,
                            const unsigned int threadsNum = 24) {
        std::vector<std::thread> threads(threadsNum);
        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }
        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};

// Tested function: apply_auto_batch
TEST_F(ApplyAutoBatchThreading, ApplyAutoBatch) {
    ov::CoreImpl core;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    ov::Output<ov::Node> intermediate = input->output(0);
    for (size_t i = 0; i < 100; ++i)
        intermediate = std::make_shared<ov::op::v0::Relu>(input)->output(0);
    auto output = std::make_shared<ov::op::v0::Result>(intermediate);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{output}, ov::ParameterVector{input});
    std::string device = "GPU";
    ov::AnyMap config;
    runParallel([&]() {
        core.apply_auto_batching(model, device, config);
    });
}
