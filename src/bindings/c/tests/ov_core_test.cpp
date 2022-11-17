// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"
#include "test_model_repo.hpp"

TEST(ov_version, api_version) {
    ov_version_t version;
    ov_get_openvino_version(&version);
    auto ver = ov::get_openvino_version();

    EXPECT_STREQ(version.buildNumber, ver.buildNumber);
    ov_version_free(&version);
}

TEST(ov_util, ov_get_error_info_check) {
    auto res = ov_get_error_info(ov_status_e::INVALID_C_PARAM);
    auto str = "invalid c input parameters";
    EXPECT_STREQ(res, str);
}

class ov_core : public ::testing::TestWithParam<std::string> {};
INSTANTIATE_TEST_SUITE_P(device_name, ov_core, ::testing::Values("CPU"));

TEST(ov_core, ov_core_create_with_config) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create_with_config(plugins_xml, &core));
    EXPECT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_create_with_no_config) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_no_bin) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, nullptr, &model));
    EXPECT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_from_memory) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    std::vector<uint8_t> weights_content(content_from_file(bin, true));

    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape;
    int64_t dims[2] = {1, (int64_t)weights_content.size()};
    ov_shape_create(2, dims, &shape);
    OV_EXPECT_OK(ov_tensor_create_from_host_ptr(ov_element_type_e::U8, shape, weights_content.data(), &tensor));
    EXPECT_NE(nullptr, tensor);

    std::vector<uint8_t> xml_content(content_from_file(xml, false));
    ov_model_t* model = nullptr;
    OV_EXPECT_OK(
        ov_core_read_model_from_memory(core, reinterpret_cast<const char*>(xml_content.data()), tensor, &model));
    EXPECT_NE(nullptr, model);

    ov_shape_free(&shape);
    ov_tensor_free(tensor);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, nullptr, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_with_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, nullptr, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    const char* key = ov_property_key_num_streams;
    const char* num = "11";
    OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 2, &compiled_model, key, num));
    EXPECT_NE(nullptr, compiled_model);

    char* property_value = nullptr;
    OV_EXPECT_OK(ov_compiled_model_get_property(compiled_model, key, &property_value));
    EXPECT_STREQ(property_value, "11");
    ov_free(property_value);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_with_property_invalid) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, nullptr, &model));
    EXPECT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    const char* key = ov_property_key_num_streams;
    const char* num = "11";
    OV_EXPECT_NOT_OK(ov_core_compile_model(core, model, device_name.c_str(), 1, &compiled_model, key, num));
    OV_EXPECT_NOT_OK(ov_core_compile_model(core, model, device_name.c_str(), 3, &compiled_model, key, num, "Test"));

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_from_file) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property_enum) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_log_level;
    const char* mode = "WARNING";

    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, mode));
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property_invalid_number_property_arguments) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key_1 = ov_property_key_inference_num_threads;
    const char* value_1 = "12";
    const char* key_2 = ov_property_key_num_streams;
    const char* value_2 = "7";

    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_1, value_1, key_2, value_2));
    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_1, &ret));
    EXPECT_STREQ(value_1, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_2, &ret));
    EXPECT_STRNE(value_2, ret);
    ov_free(ret);

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property_enum_invalid) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_hint_performance_mode;
    const char* mode = "LATENCY";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, mode));
    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STREQ(mode, ret);
    ov_free(ret);

    const char* invalid_mode = "LATENCY_TEST";
    OV_EXPECT_NOT_OK(ov_core_set_property(core, device_name.c_str(), key, invalid_mode));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STRNE(invalid_mode, ret);
    ov_free(ret);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_and_get_property_enum) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_affinity;
    const char* affinity = "HYBRID_AWARE";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, affinity));
    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STREQ(affinity, ret);
    ov_free(ret);

    key = ov_property_key_hint_inference_precision;
    const char* precision = "f32";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, precision));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STREQ(precision, ret);
    ov_free(ret);

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_and_get_property_bool) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_enable_profiling;
    const char* enable = "YES";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, enable));

    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STREQ(enable, ret);
    ov_free(ret);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_and_get_property_bool_invalid) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_enable_profiling;
    const char* enable = "TEST";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, enable));

    char* ret = nullptr;
    OV_EXPECT_NOT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    EXPECT_STRNE(enable, ret);
    ov_free(ret);

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    char* property_value;
    OV_EXPECT_OK(
        ov_core_get_property(core, device_name.c_str(), ov_property_key_supported_properties, &property_value));
    ov_free(property_value);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_get_property_str) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_cache_dir;
    const char cache_dir[] = "./cache_dir";

    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, cache_dir));

    char* property_value = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &property_value));
    EXPECT_STREQ(cache_dir, property_value);

    ov_free(property_value);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_get_property_int) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_inference_num_threads;
    const char* num = "8";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, num));

    char* property_value = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key, &property_value));
    EXPECT_STREQ(num, property_value);
    ov_free(property_value);

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property_int_invalid) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key = ov_property_key_inference_num_threads;
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key, "abc"));

    char* ret = nullptr;
    OV_EXPECT_NOT_OK(ov_core_get_property(core, device_name.c_str(), key, &ret));
    ov_free(ret);

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_multiple_common_properties) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    // Test enum
    const char* key_1 = ov_property_key_hint_performance_mode;
    const char* value_1 = "THROUGHPUT";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_1, value_1));

    char* property_value_1 = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_1, &property_value_1));
    EXPECT_STREQ(property_value_1, value_1);
    ov_free(property_value_1);

    // Test string
    const char* key_2 = ov_property_key_cache_dir;
    const char cache_dir[] = "./cache_dir";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_2, cache_dir));

    char* property_value_2 = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_2, &property_value_2));
    EXPECT_STREQ(property_value_2, cache_dir);
    ov_free(property_value_2);

    // Test int32
    const char* key_3 = ov_property_key_hint_num_requests;
    const char* num = "8";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_3, num));

    char* property_value_3 = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_3, &property_value_3));
    EXPECT_STREQ(property_value_3, num);
    ov_free(property_value_3);

    // Test bool
    const char* key_4 = ov_property_key_enable_profiling;
    const char* enable = "YES";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_4, enable));

    char* property_value_4 = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_4, &property_value_4));
    EXPECT_STREQ(property_value_4, "YES");
    ov_free(property_value_4);
    ov_core_free(core);
}

TEST(ov_core, ov_core_get_available_devices) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_available_devices_t devices;
    OV_EXPECT_OK(ov_core_get_available_devices(core, &devices));

    ov_available_devices_free(&devices);
    ov_core_free(core);
}

TEST_P(ov_core, ov_compiled_model_export_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_EXPECT_OK(ov_compiled_model_export_model(compiled_model, export_path.c_str()));

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_import_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;

    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_EXPECT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_EXPECT_OK(ov_compiled_model_export_model(compiled_model, export_path.c_str()));
    ov_compiled_model_free(compiled_model);

    std::vector<uint8_t> buffer(content_from_file(export_path.c_str(), true));
    ov_compiled_model_t* compiled_model_imported = nullptr;
    OV_EXPECT_OK(ov_core_import_model(core,
                                      reinterpret_cast<const char*>(buffer.data()),
                                      buffer.size(),
                                      device_name.c_str(),
                                      &compiled_model_imported));
    EXPECT_NE(nullptr, compiled_model_imported);
    ov_compiled_model_free(compiled_model_imported);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_versions_by_device_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_core_version_list_t version_list;
    OV_EXPECT_OK(ov_core_get_versions_by_device_name(core, device_name.c_str(), &version_list));
    EXPECT_EQ(version_list.size, 1);

    ov_core_versions_free(&version_list);
    ov_core_free(core);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
const std::vector<std::wstring> test_unicode_postfix_vector = {L"unicode_Яㅎあ",
                                                               L"ひらがな日本語",
                                                               L"大家有天分",
                                                               L"עפצקרשתםןףץ",
                                                               L"ث خ ذ ض ظ غ",
                                                               L"그것이정당하다",
                                                               L"АБВГДЕЁЖЗИЙ",
                                                               L"СТУФХЦЧШЩЬЮЯ"};
TEST(ov_core, ov_core_create_with_config_unicode) {
    ov_core_t* core = nullptr;

    for (std::size_t index = 0; index < test_unicode_postfix_vector.size(); index++) {
        std::wstring postfix = L"_" + test_unicode_postfix_vector[index];
        std::wstring plugins_xml_ws = add_unicode_postfix_to_path(plugins_xml, postfix);
        ASSERT_EQ(true, copy_file(plugins_xml, plugins_xml_ws));

        OV_EXPECT_OK(ov_core_create_with_config_unicode(plugins_xml_ws.c_str(), &core));
        EXPECT_NE(nullptr, core);
        remove_file_ws(plugins_xml_ws);
        ov_core_free(core);
    }
}

TEST(ov_core, ov_core_read_model_unicode) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    for (std::size_t index = 0; index < test_unicode_postfix_vector.size(); index++) {
        std::wstring postfix = L"_" + test_unicode_postfix_vector[index];
        std::wstring xml_ws = add_unicode_postfix_to_path(xml, postfix);
        std::wstring bin_ws = add_unicode_postfix_to_path(bin, postfix);

        ASSERT_EQ(true, copy_file(xml, xml_ws));
        ASSERT_EQ(true, copy_file(bin, bin_ws));

        OV_EXPECT_OK(ov_core_read_model_unicode(core, xml_ws.c_str(), bin_ws.c_str(), &model));
        EXPECT_NE(nullptr, model);
        remove_file_ws(xml_ws);
        remove_file_ws(bin_ws);

        ov_model_free(model);
    }

    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_from_file_unicode) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    for (std::size_t index = 0; index < test_unicode_postfix_vector.size(); index++) {
        std::wstring postfix = L"_" + test_unicode_postfix_vector[index];
        std::wstring xml_ws = add_unicode_postfix_to_path(xml, postfix);
        std::wstring bin_ws = add_unicode_postfix_to_path(bin, postfix);
        ASSERT_EQ(true, copy_file(xml, xml_ws));
        ASSERT_EQ(true, copy_file(bin, bin_ws));

        OV_EXPECT_OK(
            ov_core_compile_model_from_file_unicode(core, xml_ws.c_str(), device_name.c_str(), 0, &compiled_model));
        EXPECT_NE(nullptr, compiled_model);
        remove_file_ws(xml_ws);
        remove_file_ws(bin_ws);
        ov_compiled_model_free(compiled_model);
    }

    ov_core_free(core);
}
#endif
