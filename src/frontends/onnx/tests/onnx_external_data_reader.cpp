// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/onnx/external_data.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace ov::frontend::onnx;

namespace {

std::string model_path(const std::string& name) {
    return test::utils::getModelFromTestModelZoo(std::string(TEST_ONNX_MODELS_DIRNAME) + name);
}

std::shared_ptr<op::v0::Constant> get_constant(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto constant = as_type_ptr<op::v0::Constant>(node);
            constant && constant->get_friendly_name() == name) {
            return constant;
        }
    }
    return nullptr;
}

std::shared_ptr<Model> convert(const std::string& name, const ExternalDataReader& reader) {
    frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    const auto input_model = frontend->load(model_path(name), true, AnyMap{{external_data_reader, reader}});
    return frontend->convert(input_model);
}

// The contents of tensors_data/multiple_tensors.data as a test supplies them: data_a at offset 0, data_b at 4096.
std::shared_ptr<std::vector<uint8_t>> make_multiple_tensors_file(const std::vector<int32_t>& a,
                                                                 const std::vector<int32_t>& b) {
    auto file = std::make_shared<std::vector<uint8_t>>(4108, 0);
    std::memcpy(file->data(), a.data(), a.size() * sizeof(int32_t));
    std::memcpy(file->data() + 4096, b.data(), b.size() * sizeof(int32_t));
    return file;
}

constexpr const char* kTwoTensorsModel = "external_data/external_data_two_tensors_data_in_the_same_file.onnx";

}  // namespace

TEST(OnnxExternalDataReader, supplied_file_is_used_and_requested_once) {
    const auto file = make_multiple_tensors_file({1, 2, 3}, {7, 8, 9});
    std::vector<std::string> requested;
    const auto model = convert(kTwoTensorsModel, [&](const std::string& location) {
        requested.push_back(location);
        return ExternalDataBuffer{file->data(), file->size(), file};
    });

    // Both initializers are in the same file, which is requested once by its declared location.
    ASSERT_EQ(requested.size(), 1);
    EXPECT_EQ(requested[0], "tensors_data/multiple_tensors.data");
    EXPECT_EQ(get_constant(model, "data_a")->cast_vector<int32_t>(), (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(get_constant(model, "data_b")->cast_vector<int32_t>(), (std::vector<int32_t>{7, 8, 9}));

    // The weightless offset is still the declared one.
    const auto& rt_info = get_constant(model, "data_b")->get_rt_info();
    const auto wca = rt_info.find(WeightlessCacheAttribute::get_type_info_static());
    ASSERT_NE(wca, rt_info.end());
    EXPECT_EQ(wca->second.as<WeightlessCacheAttribute>().bin_offset, 4096);
}

TEST(OnnxExternalDataReader, supplied_data_is_not_copied) {
    const auto file = make_multiple_tensors_file({1, 2, 3}, {7, 8, 9});
    const auto model = convert(kTwoTensorsModel, [&](const std::string&) {
        return ExternalDataBuffer{file->data(), file->size(), file};
    });
    EXPECT_EQ(get_constant(model, "data_a")->get_data_ptr(), static_cast<const void*>(file->data()));
    EXPECT_GT(file.use_count(), 1) << "Constants must keep the supplied buffer alive";
}

TEST(OnnxExternalDataReader, file_not_supplied_is_read_from_disk) {
    frontend::FrontEndManager manager;
    const auto expected = manager.load_by_framework("onnx")->convert(
        manager.load_by_framework("onnx")->load(model_path(kTwoTensorsModel)));
    size_t calls = 0;
    const auto model = convert(kTwoTensorsModel, [&](const std::string&) {
        ++calls;
        return ExternalDataBuffer{};
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(get_constant(model, "data_a")->cast_vector<int32_t>(),
              get_constant(expected, "data_a")->cast_vector<int32_t>());
}

TEST(OnnxExternalDataReader, short_buffer_is_invalid_external_data) {
    const auto file = std::make_shared<std::vector<uint8_t>>(16, 0);  // data_b at offset 4096 doesn't fit
    EXPECT_ANY_THROW(convert(kTwoTensorsModel, [&](const std::string&) {
        return ExternalDataBuffer{file->data(), file->size(), file};
    }));
}

TEST(OnnxExternalDataReader, reader_error_is_reported_with_location) {
    try {
        convert(kTwoTensorsModel, [](const std::string&) -> ExternalDataBuffer {
            throw std::runtime_error("decryption failed");
        });
        FAIL() << "Expected an exception";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("tensors_data/multiple_tensors.data"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("decryption failed"), std::string::npos) << e.what();
    }
}

TEST(OnnxExternalDataReader, wrong_option_type_is_rejected) {
    frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    EXPECT_ANY_THROW(frontend->load(model_path(kTwoTensorsModel), true, AnyMap{{external_data_reader, 42}}));
}
