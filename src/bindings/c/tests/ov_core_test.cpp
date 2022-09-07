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
    OV_ASSERT_OK(ov_core_create_with_config(plugins_xml, &core));
    ASSERT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_create_with_no_config) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_no_bin) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_from_memory) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    std::vector<uint8_t> weights_content(content_from_file(bin, true));

    ov_tensor_t* tensor = nullptr;
    ov_shape_t shape;
    int64_t dims[2] = {1, (int64_t)weights_content.size()};
    ov_shape_create(2, dims, &shape);
    OV_ASSERT_OK(ov_tensor_create_from_host_ptr(ov_element_type_e::U8, shape, weights_content.data(), &tensor));
    ASSERT_NE(nullptr, tensor);

    std::vector<uint8_t> xml_content(content_from_file(xml, false));
    ov_model_t* model = nullptr;
    OV_ASSERT_OK(
        ov_core_read_model_from_memory(core, reinterpret_cast<const char*>(xml_content.data()), tensor, &model));
    ASSERT_NE(nullptr, model);

    ov_shape_free(&shape);
    ov_tensor_free(tensor);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, nullptr, &model));
    ASSERT_NE(nullptr, model);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_properties_t* property = nullptr;
    OV_ASSERT_OK(ov_core_compile_model(core, model, device_name.c_str(), property, &compiled_model));
    ASSERT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_compile_model_from_file) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    ov_properties_t* property = nullptr;
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), property, &compiled_model));
    ASSERT_NE(nullptr, compiled_model);

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_create(&properties, 1));

    const char* key = ov_property_key_hint_performance_mode;
    ov_performance_mode_e mode = ov_performance_mode_e::THROUGHPUT;
    ov_any_t value = {(void*)&mode, 1, ov_any_type_e::ENUM};
    properties.size = 1;
    properties.list[0].key = key;
    properties.list[0].value = value;

    OV_ASSERT_OK(ov_core_set_property(core, device_name.c_str(), &properties));
    ov_properties_free(&properties);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_property) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_any_t property_value;
    OV_ASSERT_OK(
        ov_core_get_property(core, device_name.c_str(), ov_property_key_supported_properties, &property_value));
    ov_any_free(&property_value);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_get_property_str) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_create(&properties, 1));

    const char* key = ov_property_key_cache_dir;
    const char cache_dir[] = "./cache_dir";
    ov_any_t value = {(void*)cache_dir, sizeof(cache_dir), ov_any_type_e::CHAR};
    properties.size = 1;
    properties.list[0].key = key;
    properties.list[0].value = value;

    OV_ASSERT_OK(ov_core_set_property(core, device_name.c_str(), &properties));

    ov_any_t property_value;
    OV_ASSERT_OK(ov_core_get_property(core, device_name.c_str(), key, &property_value));
    EXPECT_STREQ(cache_dir, (char*)property_value.ptr);

    ov_properties_free(&properties);
    ov_any_free(&property_value);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_get_property_int) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_create(&properties, 1));

    const char* key = ov_property_key_inference_num_threads;
    int32_t num = 8;
    ov_any_t value = {(void*)&num, 1, ov_any_type_e::INT32};
    properties.size = 1;
    properties.list[0].key = key;
    properties.list[0].value = value;

    OV_ASSERT_OK(ov_core_set_property(core, device_name.c_str(), &properties));

    ov_any_t property_value;
    OV_ASSERT_OK(ov_core_get_property(core, device_name.c_str(), key, &property_value));
    int32_t res = *(int32_t*)property_value.ptr;
    EXPECT_EQ(num, res);
    ov_any_free(&property_value);

    ov_properties_free(&properties);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_set_multiple_properties) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_create(&properties, 3));

    const char* key_1 = ov_property_key_hint_performance_mode;
    ov_performance_mode_e mode = ov_performance_mode_e::THROUGHPUT;
    ov_any_t value_1 = {(void*)&mode, 1, ov_any_type_e::ENUM};
    properties.list[0].key = key_1;
    properties.list[0].value = value_1;

    const char* key_2 = ov_property_key_cache_dir;
    const char cache_dir[] = "./cache_dir";
    ov_any_t value_2 = {(void*)cache_dir, sizeof(cache_dir), ov_any_type_e::CHAR};
    properties.list[1].key = key_2;
    properties.list[1].value = value_2;

    const char* key_3 = ov_property_key_hint_num_requests;
    int32_t num = 8;
    ov_any_t value_3 = {(void*)&num, 1, ov_any_type_e::UINT32};
    properties.list[2].key = key_3;
    properties.list[2].value = value_3;

    OV_ASSERT_OK(ov_core_set_property(core, device_name.c_str(), &properties));

    ov_any_t property_value_1;
    OV_ASSERT_OK(ov_core_get_property(core, device_name.c_str(), key_1, &property_value_1));
    int32_t res_1 = *(ov_performance_mode_e*)property_value_1.ptr;
    EXPECT_EQ(mode, res_1);
    ov_any_free(&property_value_1);

    ov_any_t property_value_2;
    OV_ASSERT_OK(ov_core_get_property(core, device_name.c_str(), key_2, &property_value_2));
    EXPECT_STREQ(cache_dir, (char*)property_value_2.ptr);
    ov_any_free(&property_value_2);

    ov_any_t property_value_3;
    OV_ASSERT_OK(ov_core_get_property(core, device_name.c_str(), key_3, &property_value_3));
    int32_t res_3 = *(int32_t*)property_value_3.ptr;
    EXPECT_EQ(num, res_3);
    ov_any_free(&property_value_3);

    ov_properties_free(&properties);
}

TEST(ov_core, ov_core_get_available_devices) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_available_devices_t devices;
    OV_ASSERT_OK(ov_core_get_available_devices(core, &devices));

    ov_available_devices_free(&devices);
    ov_core_free(core);
}

TEST_P(ov_core, ov_compiled_model_export_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), nullptr, &compiled_model));
    ASSERT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_ASSERT_OK(ov_compiled_model_export_model(compiled_model, export_path.c_str()));

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_import_model) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;

    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_compiled_model_t* compiled_model = nullptr;
    OV_ASSERT_OK(ov_core_compile_model_from_file(core, xml, device_name.c_str(), nullptr, &compiled_model));
    ASSERT_NE(nullptr, compiled_model);

    std::string export_path = TestDataHelpers::generate_model_path("test_model", "exported_model.blob");
    OV_ASSERT_OK(ov_compiled_model_export_model(compiled_model, export_path.c_str()));
    ov_compiled_model_free(compiled_model);

    std::vector<uint8_t> buffer(content_from_file(export_path.c_str(), true));
    ov_compiled_model_t* compiled_model_imported = nullptr;
    OV_ASSERT_OK(ov_core_import_model(core,
                                      reinterpret_cast<const char*>(buffer.data()),
                                      buffer.size(),
                                      device_name.c_str(),
                                      &compiled_model_imported));
    ASSERT_NE(nullptr, compiled_model_imported);
    ov_compiled_model_free(compiled_model_imported);
    ov_core_free(core);
}

TEST_P(ov_core, ov_core_get_versions_by_device_name) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_core_version_list_t version_list;
    OV_ASSERT_OK(ov_core_get_versions_by_device_name(core, device_name.c_str(), &version_list));
    EXPECT_EQ(version_list.size, 1);

    ov_core_versions_free(&version_list);
    ov_core_free(core);
}