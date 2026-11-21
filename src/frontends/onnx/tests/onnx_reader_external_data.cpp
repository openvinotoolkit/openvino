// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/constant.hpp"
#include "utils/tensor_external_data.hpp"

using namespace std;
using namespace ov;
using namespace frontend;
using namespace frontend::onnx::tests;
using testing::AnyOf, testing::HasSubstr;

// API 2.0 tests
class OnnxFeMmapFixture : public ::testing::TestWithParam<bool> {};

TEST_P(OnnxFeMmapFixture, onnx_external_data) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_0_size) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_0_size.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({4.f});
    test_case.add_expected_output<float>({}, {4.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_0_size_offset) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_0_size_offset.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({4.f});
    test_case.add_expected_output<float>({}, {4.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_weightless_cache_attribute) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);

    // "A" is an external initializer, so a real offset and size are expected.
    // The model has external data, so other constants are expected to have no attribute.
    const auto& operations = model->get_ordered_ops();
    size_t external_constants = 0;
    for (const auto& operation : operations) {
        const auto& const_node = ov::as_type_ptr<ov::op::v0::Constant>(operation);
        if (const_node == nullptr)
            continue;
        const auto& rt_info = const_node->get_rt_info();
        const auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (const_node->get_friendly_name() == "A") {
            ASSERT_NE(it, rt_info.end()) << "Constant 'A' is expected to have WeightlessCacheAttribute";
            const auto& weightless_cache = it->second.as<ov::WeightlessCacheAttribute>();
            EXPECT_EQ(weightless_cache.original_dtype, const_node->get_element_type());
            EXPECT_EQ(weightless_cache.bin_offset, 0);
            EXPECT_EQ(weightless_cache.original_size, const_node->get_byte_size());
            ++external_constants;
        } else {
            EXPECT_EQ(it, rt_info.end()) << "Constant '" << const_node->get_friendly_name()
                                         << "' is not external, so it is expected to have no WeightlessCacheAttribute";
        }
    }
    EXPECT_EQ(external_constants, 1);
}

TEST(OnnxWeightlessCacheAttribute, onnx_no_external_data_weightless_cache_attribute) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "add_abc_initializers.onnx");
    Core core;
    const auto model = core.read_model(path);

    // The model has no external data, so all constants are enumerated:
    // an index is used as an offset and a size stays unknown.
    const auto& operations = model->get_ordered_ops();
    size_t constants = 0;
    for (uint32_t idx = 0; idx < operations.size(); ++idx) {
        const auto& const_node = ov::as_type_ptr<ov::op::v0::Constant>(operations[idx]);
        if (const_node == nullptr)
            continue;
        const auto& rt_info = const_node->get_rt_info();
        const auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        ASSERT_NE(it, rt_info.end()) << "Constant '" << const_node->get_friendly_name()
                                     << "' is expected to have WeightlessCacheAttribute";
        const auto& weightless_cache = it->second.as<ov::WeightlessCacheAttribute>();
        EXPECT_EQ(weightless_cache.original_dtype, const_node->get_element_type());
        EXPECT_EQ(weightless_cache.bin_offset, idx);
        EXPECT_EQ(weightless_cache.original_size, 0);
        ++constants;
    }
    EXPECT_GT(constants, 0);
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_from_stream) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    ifstream stream{path, ios::in | ios::binary};
    istream* is = &stream;
    ASSERT_TRUE(stream.is_open());

    auto fem = FrontEndManager();
    auto frontend = fem.load_by_framework("onnx");
    const bool enable_mmap = GetParam();
    const auto in_model = frontend->load(is, path, enable_mmap);
    const auto model = frontend->convert(in_model);

    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();

    stream.close();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_incorrect_size_exception) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(
            string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_incorrect_data_shape.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        const auto model = core.read_model(path);
        FAIL() << "Incorrect size of external data not detected";
    } catch (const Exception& ex) {
        EXPECT_THAT(
            ex.what(),
            HasSubstr(
                "The size of the external data file does not match the byte size of an initializer 'A' in the model"));
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_optional_fields) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_optional_fields.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_size) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_optional_fields_offset_not_aligned.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) +
        "external_data/external_data_offset_not_aligned_with_page_and_less_than_page_size_with_length_provided.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) +
        "external_data/"
        "external_data_offset_not_aligned_with_page_and_greater_than_page_size_with_length_provided.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_offset_not_aligned_with_page_in_two_pages_scope) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/"
                                                            "offset_not_aligned_with_page_in_two_pages_scope.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_different_paths) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_different_paths.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    // first input: {3.f}, second: {1.f, 2.f, 5.f} read from external files
    test_case.add_input<float>({2.f, 7.f, 7.f});

    test_case.add_expected_output<float>({2.f, 4.f, 5.f});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_two_tensors_data_in_the_same_file) {
    const auto path = test::utils::getModelFromTestModelZoo(
        string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data_two_tensors_data_in_the_same_file.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    // first input: {3, 2, 1}, second: {1, 2, 3} read from external file
    test_case.add_input<int32_t>({2, 3, 1});

    test_case.add_expected_output<int32_t>({3, 3, 3});
    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_external_data_exception) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                                "external_data/external_data_file_not_found.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            string("not_existed_file.data, offset: 4096, data_length: 16)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_up_dir_path) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(
            string(TEST_ONNX_MODELS_DIRNAME) + "external_data/inner_scope/external_data_file_in_up_dir.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring, string("which is outside the base directory"), ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_invalid_data_length) {
    try {
        const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                                "external_data/external_data_invalid_data_length.onnx");
        Core core;
        core.set_property(enable_mmap(GetParam()));
        core.read_model(path);
        FAIL() << "Incorrect path to external data not detected";
    } catch (const Exception& ex) {
        EXPECT_PRED_FORMAT2(testing::IsSubstring,
                            string("tensor.data, offset: 0, "
                                   "data_length: 30000000000)"),
                            ex.what());
    } catch (...) {
        FAIL() << "Importing onnx model failed for unexpected reason";
    }
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_sanitize_path) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_sanitize_test.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {3.f, 6.f, 9.f, 12.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_in_constant_node) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_in_constant_node.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({3.f, 5.f, 8.f, 13.f});
    test_case.add_expected_output<float>(Shape{2, 2}, {4.f, 7.f, 11.f, 17.f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int16) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int16.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<int16_t>({-100});
    test_case.add_expected_output<int16_t>(Shape{2, 2}, {-100, 16156, -100, 16284});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint16) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint16.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<uint16_t>({100});
    test_case.add_expected_output<uint16_t>(Shape{2, 2}, {100, 16356, 100, 16484});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int8) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int8.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<int8_t>({-100});
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {-100, -100, 28, -37});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint8) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint8.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<uint8_t>({100});
    test_case.add_expected_output<uint8_t>(Shape{2, 2}, {100, 100, 228, 163});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_int4) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_int4.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<int8_t>(Shape{2, 2}, {static_cast<int8_t>(0x80), 0x3f});

    test_case.run();
}

TEST_P(OnnxFeMmapFixture, onnx_external_data_uint4) {
    const auto path = test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) +
                                                            "external_data/external_data_uint4.onnx");
    Core core;
    core.set_property(enable_mmap(GetParam()));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_expected_output<uint8_t>(Shape{2, 2}, {0x80, 0x3f});

    test_case.run();
}

using ::ONNX_NAMESPACE::ModelProto;

TEST_P(OnnxFeMmapFixture, onnx_external_data_from_ORT_MEM_ADDR) {
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Could not open model file: " << path;
    auto model_proto = std::make_shared<ModelProto>();
    ASSERT_TRUE(model_proto->ParseFromIstream(&ifs)) << "Could not parse ModelProto";
    ifs.close();

    std::vector<float> test_data = {2.0f, 3.0f, 4.0f, 5.0f};
    char* data_ptr = reinterpret_cast<char*>(test_data.data());
    uint64_t data_addr = reinterpret_cast<uint64_t>(data_ptr);
    size_t data_size = test_data.size() * sizeof(float);

    auto* graph = model_proto->mutable_graph();
    for (auto& initializer : *graph->mutable_initializer()) {
        if (initializer.has_data_location() && initializer.data_location() == ::ONNX_NAMESPACE::TensorProto::EXTERNAL) {
            initializer.clear_external_data();
            auto* location_entry = initializer.add_external_data();
            location_entry->set_key("location");
            location_entry->set_value("*/_ORT_MEM_ADDR_/*");
            auto* offset_entry = initializer.add_external_data();
            offset_entry->set_key("offset");
            offset_entry->set_value(std::to_string(data_addr));
            auto* length_entry = initializer.add_external_data();
            length_entry->set_key("length");
            length_entry->set_value(std::to_string(data_size));
            break;
        }
    }
    uint64_t model_proto_ptr = reinterpret_cast<uint64_t>(model_proto.get());
    auto fem = ov::frontend::FrontEndManager();
    auto frontend = fem.load_by_framework("onnx");
    ASSERT_NE(frontend, nullptr);
    std::shared_ptr<ov::frontend::InputModel> input_model;
    ASSERT_NO_THROW(input_model = frontend->load(model_proto_ptr, GetParam()))
        << "Could not load model from modified ModelProto";
    ASSERT_NE(input_model, nullptr);
    model_proto.reset();

    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = frontend->convert(input_model)) << "Could not convert model with ORT_MEM_ADDR";
    ASSERT_NE(model, nullptr);

    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({2, 2}, {4.f, 7.f, 10.f, 13.f});

    test_case.run();
}

INSTANTIATE_TEST_SUITE_P(OnnxFeMMapReadModel, OnnxFeMmapFixture, ::testing::Bool());

TEST(OnnxFeOrtMemAddr, onnx_external_data_from_ORT_MEM_ADDR_stream) {
    // ONNX Runtime passes in-memory initializers as FrontEnd::load({std::istream*, model_path})
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open()) << "Could not open model file: " << path;
    ModelProto model_proto;
    ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Could not parse ModelProto";
    ifs.close();

    std::vector<float> test_data = {2.0f, 3.0f, 4.0f, 5.0f};
    const auto data_addr = reinterpret_cast<uint64_t>(test_data.data());
    const auto data_size = test_data.size() * sizeof(float);
    bool replaced = false;
    for (auto& initializer : *model_proto.mutable_graph()->mutable_initializer()) {
        if (initializer.has_data_location() && initializer.data_location() == ::ONNX_NAMESPACE::TensorProto::EXTERNAL) {
            initializer.clear_external_data();
            auto* location_entry = initializer.add_external_data();
            location_entry->set_key("location");
            location_entry->set_value("*/_ORT_MEM_ADDR_/*");
            auto* offset_entry = initializer.add_external_data();
            offset_entry->set_key("offset");
            offset_entry->set_value(std::to_string(data_addr));
            auto* length_entry = initializer.add_external_data();
            length_entry->set_key("length");
            length_entry->set_value(std::to_string(data_size));
            replaced = true;
            break;
        }
    }
    ASSERT_TRUE(replaced);
    std::istringstream model_stream(model_proto.SerializeAsString());
    std::istream& stream = model_stream;
    ov::AnyVector params{&stream, path};

    auto fem = FrontEndManager();
    auto frontend = fem.load_by_model(params);
    ASSERT_NE(frontend, nullptr);
    std::shared_ptr<ov::frontend::InputModel> input_model;
    ASSERT_NO_THROW(input_model = frontend->load(params)) << "Could not load model from stream with ORT_MEM_ADDR";
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = frontend->convert(input_model)) << "Could not convert model with ORT_MEM_ADDR";

    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>({2, 2}, {4.f, 7.f, 10.f, 13.f});
    test_case.run();
}

// With ORT_MEM_ADDR the "offset" field is a raw memory address, so a model read from a file must not use it
// (zero-size data is still accepted, see onnx_external_data_0_size)
class OnnxFeOrtMemAddrInFileFixture : public ::testing::TestWithParam<std::tuple<std::string, bool>> {};

TEST_P(OnnxFeOrtMemAddrInFileFixture, onnx_external_data_ORT_MEM_ADDR_in_file_is_rejected) {
    const auto& [model_name, use_mmap] = GetParam();
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/" + model_name);
    Core core;
    core.set_property(enable_mmap(use_mmap));
    OV_EXPECT_THROW(
        core.read_model(path),
        Exception,
        HasSubstr("has external data location */_ORT_MEM_ADDR_/* which is not allowed for a model read from a file"));
}

INSTANTIATE_TEST_SUITE_P(OnnxFeOrtMemAddrInFile,
                         OnnxFeOrtMemAddrInFileFixture,
                         ::testing::Combine(::testing::Values("ort_mem_addr_offset_1.onnx",
                                                              "ort_mem_addr_offset_4096.onnx",
                                                              "ort_mem_addr_offset_4096_length_65536.onnx",
                                                              "ort_mem_addr_in_constant_node.onnx",
                                                              "ort_mem_addr_in_subgraph.onnx",
                                                              "ort_mem_addr_in_function.onnx"),
                                            ::testing::Bool()));

TEST(OnnxFeOrtMemAddr, onnx_external_data_ORT_MEM_ADDR_in_function_attribute_proto_and_malformed_length_is_rejected) {
    const auto base_path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/external_data.onnx");
    std::ifstream ifs(base_path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    ModelProto base_proto;
    ASSERT_TRUE(base_proto.ParseFromIstream(&ifs));
    ifs.close();

    const auto tmp_path = base_path + ".tmp_ort_mem_addr_attr.onnx";
    auto write_and_expect_reject = [&](const ModelProto& proto) {
        {
            std::ofstream ofs(tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(ofs.is_open());
            ASSERT_TRUE(proto.SerializeToOstream(&ofs));
        }
        Core core;
        OV_EXPECT_THROW(
            core.read_model(tmp_path),
            Exception,
            HasSubstr(
                "has external data location */_ORT_MEM_ADDR_/* which is not allowed for a model read from a file"));
        std::remove(tmp_path.c_str());
    };

    // 1. FunctionProto.attribute_proto default tensor carrying ORT_MEM_ADDR
    {
        ModelProto proto = base_proto;
        for (auto& initializer : *proto.mutable_graph()->mutable_initializer()) {
            initializer.clear_external_data();
            initializer.set_data_location(::ONNX_NAMESPACE::TensorProto::DEFAULT);
            initializer.set_raw_data(std::string(16, '\0'));
        }
        auto* fn = proto.add_functions();
        fn->set_name("fn_with_default_tensor_attr");
        fn->set_domain("custom");
        auto* attr = fn->add_attribute_proto();
        attr->set_name("default_w");
        attr->set_type(::ONNX_NAMESPACE::AttributeProto::TENSOR);
        auto* t = attr->mutable_t();
        t->set_name("fn_attr_tensor");
        t->set_data_type(::ONNX_NAMESPACE::TensorProto::FLOAT);
        t->add_dims(4);
        t->set_data_location(::ONNX_NAMESPACE::TensorProto::EXTERNAL);
        auto* loc = t->add_external_data();
        loc->set_key("location");
        loc->set_value("*/_ORT_MEM_ADDR_/*");
        auto* off = t->add_external_data();
        off->set_key("offset");
        off->set_value("4096");
        auto* len = t->add_external_data();
        len->set_key("length");
        len->set_value("16");
        write_and_expect_reject(proto);
    }

    // 2. Malformed length strings ("0abc", "invalid", "-0") on ORT_MEM_ADDR
    for (const char* bad_length : {"0abc", "invalid", "-0"}) {
        ModelProto proto = base_proto;
        for (auto& initializer : *proto.mutable_graph()->mutable_initializer()) {
            initializer.clear_external_data();
            initializer.set_data_location(::ONNX_NAMESPACE::TensorProto::EXTERNAL);
            auto* loc = initializer.add_external_data();
            loc->set_key("location");
            loc->set_value("*/_ORT_MEM_ADDR_/*");
            auto* off = initializer.add_external_data();
            off->set_key("offset");
            off->set_value("4096");
            auto* len = initializer.add_external_data();
            len->set_key("length");
            len->set_value(bad_length);
        }
        write_and_expect_reject(proto);
    }
}

class OnnxFeOrtMemAddrControlFixture : public ::testing::TestWithParam<std::tuple<std::string, bool>> {};

TEST_P(OnnxFeOrtMemAddrControlFixture, onnx_external_data_ORT_MEM_ADDR_controls_are_accepted) {
    const auto& [model_name, use_mmap] = GetParam();
    const auto path =
        test::utils::getModelFromTestModelZoo(string(TEST_ONNX_MODELS_DIRNAME) + "external_data/" + model_name);
    Core core;
    core.set_property(enable_mmap(use_mmap));
    const auto model = core.read_model(path);
    auto test_case = test::TestCase(model);
    test_case.add_input<float>({1.f, 2.f, 3.f, 4.f});
    test_case.add_expected_output<float>(Shape{4}, {2.f, 4.f, 6.f, 8.f});
    test_case.run();
}

INSTANTIATE_TEST_SUITE_P(OnnxFeOrtMemAddrControl,
                         OnnxFeOrtMemAddrControlFixture,
                         ::testing::Combine(::testing::Values("ort_mem_addr_control_raw.onnx",
                                                              "ort_mem_addr_control_file.onnx"),
                                            ::testing::Bool()));
