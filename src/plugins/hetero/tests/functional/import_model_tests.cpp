// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cctype>
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

class NonSeekableOutputBuffer : public std::streambuf {
public:
    const std::string& str() const {
        return _data;
    }

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }
        _data.append(data, static_cast<std::string::size_type>(count));
        return count;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        _data.push_back(traits_type::to_char_type(ch));
        return ch;
    }

    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }

    pos_type seekpos(pos_type, std::ios_base::openmode) override {
        return pos_type(off_type(-1));
    }

private:
    std::string _data;
};

class SeekCountingOutputBuffer : public std::streambuf {
public:
    const std::string& str() const {
        return _data;
    }

    std::size_t seek_count() const {
        return _seek_count;
    }

    void reset_seek_count() {
        _seek_count = 0;
    }

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }

        const auto offset = static_cast<std::string::size_type>(_pos);
        const auto size = static_cast<std::string::size_type>(count);
        if (offset + size > _data.size()) {
            _data.resize(offset + size);
        }
        for (std::string::size_type i = 0; i < size; ++i) {
            _data[offset + i] = data[i];
        }
        _pos += count;
        return count;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }

        const char c = traits_type::to_char_type(ch);
        return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::out) == 0) {
            return pos_type(off_type(-1));
        }

        ++_seek_count;

        off_type base = 0;
        if (dir == std::ios_base::cur) {
            base = _pos;
        } else if (dir == std::ios_base::end) {
            base = static_cast<off_type>(_data.size());
        } else if (dir != std::ios_base::beg) {
            return pos_type(off_type(-1));
        }

        const auto new_pos = base + off;
        if (new_pos < 0) {
            return pos_type(off_type(-1));
        }
        _pos = new_pos;
        return pos_type(_pos);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
    }

private:
    std::string _data;
    off_type _pos = 0;
    std::size_t _seek_count = 0;
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
    const auto key_pos = header.find(HETERO_BLOB_FORMAT_VERSION_ATTR);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("Failed to find HETERO blob format version attribute");
    }

    auto erase_begin = key_pos;
    if (erase_begin > 0 && std::isspace(static_cast<unsigned char>(header[erase_begin - 1]))) {
        --erase_begin;
    }

    const auto eq_pos = header.find('=', key_pos + std::char_traits<char>::length(HETERO_BLOB_FORMAT_VERSION_ATTR));
    if (eq_pos == std::string::npos) {
        throw std::runtime_error("Failed to parse HETERO blob format version attribute");
    }

    const auto value_begin = header.find_first_not_of(" \t", eq_pos + 1);
    if (value_begin == std::string::npos || (header[value_begin] != '"' && header[value_begin] != '\'')) {
        throw std::runtime_error("Failed to parse HETERO blob format version attribute value");
    }

    const auto quote = header[value_begin];
    const auto value_end = header.find(quote, value_begin + 1);
    if (value_end == std::string::npos) {
        throw std::runtime_error("Failed to parse HETERO blob format version attribute value");
    }

    header.erase(erase_begin, value_end - erase_begin + 1);
    return header;
}

std::string get_hetero_xml_header(const std::string& blob) {
    const auto header_end = blob.find('\n');
    if (header_end == std::string::npos) {
        throw std::runtime_error("Failed to find HETERO blob XML header");
    }
    return blob.substr(0, header_end + 1);
}

std::string make_legacy_unframed_blob_from_framed_blob(const std::string& framed_blob, char expected_payload_type) {
    auto legacy_blob = remove_blob_format_version(get_hetero_xml_header(framed_blob));

    std::stringstream framed_stream(framed_blob);
    std::string hetero_xml_header;
    std::getline(framed_stream, hetero_xml_header);

    const auto payload_header = read_test_payload_header(framed_stream);
    if (payload_header.type != expected_payload_type) {
        throw std::runtime_error("Unexpected HETERO test payload type");
    }
    if (payload_header.size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("HETERO test payload is too large");
    }

    std::string legacy_payload(static_cast<std::string::size_type>(payload_header.size), '\0');
    if (!legacy_payload.empty()) {
        framed_stream.read(&legacy_payload[0], static_cast<std::streamsize>(payload_header.size));
        if (!framed_stream) {
            throw std::runtime_error("Failed to read HETERO test payload");
        }
    }

    legacy_blob += legacy_payload;
    return legacy_blob;
}

void expect_compiled_model_runs(HeteroTests& fixture, ov::CompiledModel compiled_model) {
    ASSERT_EQ(1, compiled_model.inputs().size());
    ASSERT_EQ(1, compiled_model.outputs().size());

    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor =
        fixture.create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    ASSERT_NO_THROW(infer_request.infer());

    auto output_tensor = infer_request.get_output_tensor();
    EXPECT_EQ(compiled_model.output().get_shape(), output_tensor.get_shape());
    EXPECT_EQ(compiled_model.output().get_element_type(), output_tensor.get_element_type());
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
    EXPECT_TRUE(payload_header.type == COMPILED_BLOB_PAYLOAD || payload_header.type == IR_PAYLOAD);
    EXPECT_GT(payload_header.size, 0);

    std::stringstream import_stream(blob);
    auto imported_model = core.import_model(import_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, imported_model.inputs().size());
    EXPECT_EQ(1, imported_model.outputs().size());
}

TEST_F(HeteroTests, export_single_plugin_without_export_import_uses_framed_ir_payload) {
    auto model = create_model_with_reshape();
    EXPECT_NO_THROW(core.get_available_devices());
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCKIR"));
    auto blob = export_compiled_model(compiled_model);
    std::stringstream model_stream(blob);

    std::string hetero_xml_header;
    std::getline(model_stream, hetero_xml_header);
    EXPECT_NE(hetero_xml_header.find(HETERO_BLOB_FORMAT_VERSION_ATTR), std::string::npos);

    const auto payload_header = read_test_payload_header(model_stream);
    EXPECT_EQ(IR_PAYLOAD, payload_header.type);
    EXPECT_GT(payload_header.size, 0);

    std::stringstream import_stream(blob);
    auto imported_model = core.import_model(import_stream, ov::test::utils::DEVICE_HETERO, {});
    expect_compiled_model_runs(*this, imported_model);
}

TEST_F(HeteroTests, export_single_plugin_to_non_seekable_stream) {
    auto model = create_model_with_reshape();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0"));

    NonSeekableOutputBuffer output_buffer;
    std::ostream model_stream(&output_buffer);
    compiled_model.export_model(model_stream);
    EXPECT_TRUE(model_stream);

    std::stringstream import_stream(output_buffer.str());
    auto imported_model = core.import_model(import_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, imported_model.inputs().size());
    EXPECT_EQ(1, imported_model.outputs().size());
}

TEST_F(HeteroTests, export_ir_payload_to_non_seekable_stream) {
    auto model = create_model_with_reshape();
    EXPECT_NO_THROW(core.get_available_devices());
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCKIR"));

    NonSeekableOutputBuffer output_buffer;
    std::ostream model_stream(&output_buffer);
    compiled_model.export_model(model_stream);
    EXPECT_TRUE(model_stream);

    std::stringstream model_blob(output_buffer.str());
    std::string hetero_xml_header;
    std::getline(model_blob, hetero_xml_header);
    EXPECT_NE(hetero_xml_header.find(HETERO_BLOB_FORMAT_VERSION_ATTR), std::string::npos);
    const auto payload_header = read_test_payload_header(model_blob);
    EXPECT_EQ(IR_PAYLOAD, payload_header.type);
    EXPECT_GT(payload_header.size, 0);

    std::stringstream import_stream(output_buffer.str());
    auto imported_model = core.import_model(import_stream, ov::test::utils::DEVICE_HETERO, {});
    expect_compiled_model_runs(*this, imported_model);
}

TEST_F(HeteroTests, framed_payload_output_buffer_rejects_forward_seek_beyond_written_data) {
    std::stringstream model_stream;
    const auto payload_frame = start_framed_payload(model_stream, COMPILED_BLOB_PAYLOAD);

    FramedPayloadOutputBuffer payload_buffer(model_stream);
    std::ostream payload_stream(&payload_buffer);

    payload_stream.write("abc", 3);
    ASSERT_TRUE(payload_stream);

    payload_stream.seekp(10, std::ios::beg);
    EXPECT_TRUE(payload_stream.fail());

    payload_stream.clear();
    payload_stream.seekp(0, std::ios::end);
    ASSERT_TRUE(payload_stream);
    payload_stream.write("d", 1);
    ASSERT_TRUE(payload_stream);

    finish_framed_payload(model_stream, payload_frame, payload_buffer.written_size());

    std::stringstream verify_stream(model_stream.str());
    const auto payload_header = read_test_payload_header(verify_stream);
    ASSERT_EQ(COMPILED_BLOB_PAYLOAD, payload_header.type);
    ASSERT_EQ(4, payload_header.size);

    std::string payload(static_cast<std::string::size_type>(payload_header.size), '\0');
    verify_stream.read(payload.data(), static_cast<std::streamsize>(payload_header.size));
    ASSERT_TRUE(verify_stream);
    EXPECT_EQ("abcd", payload);
}

TEST_F(HeteroTests, framed_payload_output_buffer_avoids_seeks_for_sequential_writes) {
    SeekCountingOutputBuffer output_buffer;
    std::ostream model_stream(&output_buffer);

    FramedPayloadOutputBuffer payload_buffer(model_stream);
    std::ostream payload_stream(&payload_buffer);

    output_buffer.reset_seek_count();
    payload_stream.write("abc", 3);
    payload_stream.write("def", 3);
    ASSERT_TRUE(payload_stream);

    EXPECT_EQ(0U, output_buffer.seek_count());
    EXPECT_EQ(6, payload_buffer.written_size());
    EXPECT_EQ("abcdef", output_buffer.str());
}

TEST_F(HeteroTests, bounded_string_output_buffer_rejects_payload_larger_than_limit) {
    BoundedStringOutputBuffer payload_buffer(3);
    std::ostream payload_stream(&payload_buffer);

    payload_stream.write("abc", 3);
    ASSERT_TRUE(payload_stream);

    payload_stream.write("d", 1);
    EXPECT_FALSE(payload_stream);
    EXPECT_EQ("abc", payload_buffer.str());
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
    const auto framed_blob = export_compiled_model(hetero_compiled_model);
    auto legacy_blob = remove_blob_format_version(get_hetero_xml_header(framed_blob));

    std::stringstream framed_stream(framed_blob);
    std::string hetero_xml_header;
    std::getline(framed_stream, hetero_xml_header);

    const auto payload_header = read_test_payload_header(framed_stream);
    ASSERT_GT(payload_header.size, 0);
    ASSERT_LE(payload_header.size, static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()));

    std::string legacy_payload(static_cast<std::string::size_type>(payload_header.size), '\0');
    framed_stream.read(legacy_payload.data(), static_cast<std::streamsize>(payload_header.size));
    ASSERT_TRUE(framed_stream);
    legacy_blob += legacy_payload;

    std::stringstream model_stream(legacy_blob);
    auto imported_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    EXPECT_EQ(1, imported_model.inputs().size());
    EXPECT_EQ(1, imported_model.outputs().size());
}

TEST_F(HeteroTests, import_single_plugin_legacy_unframed_ir_blob_without_format_version) {
    auto model = create_model_with_reshape();
    EXPECT_NO_THROW(core.get_available_devices());
    const auto hetero_compiled_model =
        core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCKIR"));
    const auto framed_blob = export_compiled_model(hetero_compiled_model);
    const auto legacy_blob = make_legacy_unframed_blob_from_framed_blob(framed_blob, IR_PAYLOAD);

    std::stringstream model_stream(legacy_blob);
    auto imported_model = core.import_model(model_stream, ov::test::utils::DEVICE_HETERO, {});
    expect_compiled_model_runs(*this, imported_model);
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