// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "blob_serialization.hpp"
#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"

namespace ov {
namespace hetero {
namespace tests {

namespace {

struct TestPayloadHeader {
    char type = 0;
    std::uint64_t size = 0;
};

std::string export_compiled_model(ov::CompiledModel compiled_model) {
    std::stringstream model_stream;
    compiled_model.export_model(model_stream);
    return model_stream.str();
}

TestPayloadHeader read_test_payload_header(std::istream& model_stream) {
    TestPayloadHeader payload_header;
    model_stream.read(&payload_header.type, sizeof(payload_header.type));
    model_stream.read(reinterpret_cast<char*>(&payload_header.size), sizeof(payload_header.size));
    if (!model_stream) {
        throw std::runtime_error("Failed to read HETERO test payload header");
    }
    return payload_header;
}

void append_test_size(std::string& blob, std::uint64_t size) {
    blob.append(reinterpret_cast<const char*>(&size), sizeof(size));
}

std::string make_framed_ir_blob(const std::string& hetero_xml_header, const std::string& payload) {
    auto blob = hetero_xml_header;
    blob.push_back(IR_PAYLOAD);
    append_test_size(blob, payload.size());
    blob += payload;
    return blob;
}

std::string remove_blob_format_version(std::string header) {
    const std::string blob_format_attr =
        std::string{" "} + HETERO_BLOB_FORMAT_VERSION_ATTR + "=\"" + std::to_string(HETERO_BLOB_FORMAT_VERSION) + "\"";
    const auto attr_pos = header.find(blob_format_attr);
    if (attr_pos == std::string::npos) {
        throw std::runtime_error("Failed to find HETERO blob format version attribute");
    }
    header.erase(attr_pos, blob_format_attr.size());
    return header;
}

std::string get_hetero_xml_header(const std::string& blob) {
    const auto header_end = blob.find('\n');
    if (header_end == std::string::npos) {
        throw std::runtime_error("Failed to find HETERO blob XML header");
    }
    return blob.substr(0, header_end + 1);
}

}  // namespace

// IR frontend is needed for import
#ifdef IR_FRONTEND_ENABLED
TEST_F(HeteroTests, import_single_plugins) {
    std::stringstream model_stream;
    auto model = create_model_with_reshape();
    {
        auto compiled_model =
            core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor =
        create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), model->input().get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), model->input().get_element_type());
}

TEST_F(HeteroTests, export_single_plugin_uses_framed_payload) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    auto blob = export_compiled_model(compiled_model);
    std::stringstream model_stream(blob);

    std::string hetero_xml_header;
    std::getline(model_stream, hetero_xml_header);
    EXPECT_NE(hetero_xml_header.find(HETERO_BLOB_FORMAT_VERSION_ATTR), std::string::npos);

    const auto payload_header = read_test_payload_header(model_stream);
    EXPECT_EQ(COMPILED_BLOB_PAYLOAD, payload_header.type);
    EXPECT_GT(payload_header.size, 0);

    std::stringstream import_stream(blob);
    auto imported_model = core.import_model(import_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, imported_model.inputs().size());
    EXPECT_EQ(1, imported_model.outputs().size());
}

TEST_F(HeteroTests, import_large_compiled_payload_uses_bounded_stream_and_throws) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    const auto hetero_xml_header = get_hetero_xml_header(export_compiled_model(compiled_model));

    std::string blob = hetero_xml_header;
    blob.push_back(COMPILED_BLOB_PAYLOAD);
    append_test_size(blob, MAX_IN_MEMORY_COMPILED_PAYLOAD_SIZE + 1);

    std::stringstream model_stream(blob);
    EXPECT_THROW(core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {}), ov::Exception);
}

TEST_F(HeteroTests, import_single_plugin_legacy_unframed_blob_without_format_version) {
    auto model = create_model_with_reshape();
    const auto hetero_compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    auto legacy_blob = remove_blob_format_version(get_hetero_xml_header(export_compiled_model(hetero_compiled_model)));

    const auto submodel_compiled_model = core.compile_model(model, "MOCK0", {});
    legacy_blob += export_compiled_model(submodel_compiled_model);

    std::stringstream model_stream(legacy_blob);
    auto imported_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, imported_model.inputs().size());
    EXPECT_EQ(1, imported_model.outputs().size());
}

TEST_F(HeteroTests, import_truncated_framed_payload_throws) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    auto blob = export_compiled_model(compiled_model);
    ASSERT_FALSE(blob.empty());
    blob.pop_back();

    std::stringstream model_stream(blob);
    EXPECT_THROW(core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {}), ov::Exception);
}

TEST_F(HeteroTests, import_truncated_framed_cpu_payload_throws) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities(ov::test::utils::DEVICE_CPU));
    auto blob = export_compiled_model(compiled_model);
    ASSERT_FALSE(blob.empty());
    blob.pop_back();

    std::stringstream model_stream(blob);
    EXPECT_THROW(core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {}), ov::Exception);
}

TEST_F(HeteroTests, import_framed_ir_payload_with_oversized_xml_size_throws) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    const auto hetero_xml_header = get_hetero_xml_header(export_compiled_model(compiled_model));

    std::string payload;
    append_test_size(payload, std::numeric_limits<std::uint64_t>::max());
    std::stringstream model_stream(make_framed_ir_blob(hetero_xml_header, payload));

    EXPECT_THROW(core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {}), ov::Exception);
}

TEST_F(HeteroTests, import_framed_ir_payload_with_oversized_weights_size_throws) {
    auto model = create_model_with_reshape();
    const auto compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));
    const auto hetero_xml_header = get_hetero_xml_header(export_compiled_model(compiled_model));

    std::string payload;
    append_test_size(payload, 0);
    append_test_size(payload, std::numeric_limits<std::uint64_t>::max());
    std::stringstream model_stream(make_framed_ir_blob(hetero_xml_header, payload));

    EXPECT_THROW(core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {}), ov::Exception);
}

TEST_F(HeteroTests, import_several_plugins) {
    std::stringstream model_stream;
    auto model = create_model_with_subtract();
    {
        auto compiled_model =
            core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
        compiled_model.export_model(model_stream);
    }
    auto compiled_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, compiled_model.inputs().size());
    EXPECT_EQ(1, compiled_model.outputs().size());
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor =
        create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(input_tensor.get_shape(), output_tensor.get_shape());
    EXPECT_EQ(input_tensor.get_element_type(), output_tensor.get_element_type());
    EXPECT_EQ(memcmp(input_tensor.data(), output_tensor.data(), input_tensor.get_byte_size()), 0);
}
#endif
}  // namespace tests
}  // namespace hetero
}  // namespace ov