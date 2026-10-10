// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"
#include "utils/graph_serializer/deserializer.hpp"

namespace ov::intel_cpu {
namespace {

using DataHeader = ov::pass::StreamSerialize::DataHeader;

std::string make_blob(const DataHeader& header, size_t size) {
    std::string blob(size, 'x');
    std::memcpy(blob.data(), &header, sizeof(header));
    return blob;
}

void expect_invalid_header(ModelDeserializer& deserializer, std::shared_ptr<ov::Model>& model) {
    try {
        deserializer >> model;
        FAIL() << "Expected an invalid compiled-model header";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("[CPU] Could not deserialize by device xml header."),
                  std::string::npos);
    }
}

TEST(GraphSerializerDeserializer, RejectsSectionsOutsideBlob) {
    constexpr size_t blob_size = 4096;
    DataHeader header{};
    header.custom_data_offset = sizeof(header);
    header.custom_data_size = 1048576;
    header.consts_offset = header.custom_data_offset + header.custom_data_size;
    header.model_offset = 100;
    header.consts_size = header.model_offset - header.consts_offset;

    auto reversed = header;
    reversed.consts_offset = sizeof(header) - 1;
    reversed.custom_data_size = reversed.consts_offset - reversed.custom_data_offset;
    reversed.consts_size = reversed.model_offset - reversed.consts_offset;

    auto inconsistent = header;
    inconsistent.consts_offset = sizeof(header) + 1;
    inconsistent.custom_data_size = 2;
    inconsistent.consts_size = inconsistent.model_offset - inconsistent.consts_offset;

    auto no_model = header;
    no_model.consts_offset = sizeof(header);
    no_model.custom_data_size = 0;
    no_model.model_offset = blob_size;
    no_model.consts_size = no_model.model_offset - no_model.consts_offset;

    const std::array<DataHeader, 4> invalid{header, reversed, inconsistent, no_model};
    for (const auto& candidate : invalid) {
        const auto blob = make_blob(candidate, blob_size);
        auto buffer = std::make_shared<ov::AlignedBuffer>(blob.size());
        std::memcpy(buffer->get_ptr(), blob.data(), blob.size());
        std::shared_ptr<ov::Model> model;
        auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();

        ModelDeserializer from_buffer(buffer, core, CacheDecrypt{}, false);
        expect_invalid_header(from_buffer, model);

        std::istringstream stream(blob);
        ModelDeserializer from_stream(stream, core, CacheDecrypt{}, false);
        expect_invalid_header(from_stream, model);
    }
}

TEST(GraphSerializerDeserializer, RejectsPreviouslyUnpinnedSectionTerms) {
    constexpr size_t blob_size = 4096;
    DataHeader wrong_start{};
    wrong_start.custom_data_offset = sizeof(DataHeader) + 1;
    wrong_start.consts_offset = wrong_start.custom_data_offset;
    wrong_start.model_offset = wrong_start.consts_offset;

    DataHeader reversed_sections{};
    reversed_sections.custom_data_offset = sizeof(DataHeader);
    reversed_sections.consts_offset = sizeof(DataHeader) + 2;
    reversed_sections.custom_data_size = reversed_sections.consts_offset - reversed_sections.custom_data_offset;
    reversed_sections.model_offset = sizeof(DataHeader) + 1;
    reversed_sections.consts_size = reversed_sections.model_offset - reversed_sections.consts_offset;

    DataHeader wrong_const_size{};
    wrong_const_size.custom_data_offset = sizeof(DataHeader);
    wrong_const_size.consts_offset = sizeof(DataHeader);
    wrong_const_size.model_offset = sizeof(DataHeader) + 1;
    wrong_const_size.consts_size = 2;

    const std::array<std::pair<const char*, DataHeader>, 3> cases{{
        {"custom offset is not the end of the header", wrong_start},
        {"constants begin after the model", reversed_sections},
        {"constants size disagrees with its span", wrong_const_size},
    }};
    for (const auto& [name, candidate] : cases) {
        SCOPED_TRACE(name);
        const auto blob = make_blob(candidate, blob_size);
        auto buffer = std::make_shared<ov::AlignedBuffer>(blob.size());
        std::memcpy(buffer->get_ptr(), blob.data(), blob.size());
        auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
        std::shared_ptr<ov::Model> model;

        ModelDeserializer from_buffer(buffer, core, CacheDecrypt{}, false);
        expect_invalid_header(from_buffer, model);

        std::istringstream stream(blob);
        ModelDeserializer from_stream(stream, core, CacheDecrypt{}, false);
        expect_invalid_header(from_stream, model);
    }
}

TEST(GraphSerializerDeserializer, AcceptsNonemptyCustomDataAndConstants) {
    constexpr char custom_data[] = "<cnndata/>";
    constexpr char constants[] = "abcd";
    constexpr char model_xml[] = "<net/>";
    DataHeader header{};
    header.custom_data_offset = sizeof(header);
    header.custom_data_size = sizeof(custom_data) - 1;
    header.consts_offset = header.custom_data_offset + header.custom_data_size;
    header.consts_size = sizeof(constants) - 1;
    header.model_offset = header.consts_offset + header.consts_size;
    const auto size = header.model_offset + sizeof(model_xml) - 1;
    auto blob = make_blob(header, size);
    blob.replace(header.custom_data_offset, header.custom_data_size, custom_data);
    blob.replace(header.consts_offset, header.consts_size, constants);
    blob.replace(header.model_offset, sizeof(model_xml) - 1, model_xml);

    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    EXPECT_CALL(*core, read_model(testing::_, testing::_))
        .Times(2)
        .WillRepeatedly(
            [&](const std::shared_ptr<ov::AlignedBuffer>& xml, const std::shared_ptr<ov::AlignedBuffer>& weights) {
                EXPECT_EQ(std::string(xml->get_ptr<char>(), xml->size()), model_xml);
                EXPECT_EQ(std::string(weights->get_ptr<char>(), weights->size()), constants);
                return std::shared_ptr<ov::Model>{};
            });

    auto buffer = std::make_shared<ov::AlignedBuffer>(blob.size());
    std::memcpy(buffer->get_ptr(), blob.data(), blob.size());
    std::shared_ptr<ov::Model> model;
    ModelDeserializer from_buffer(buffer, core, CacheDecrypt{}, false);
    EXPECT_NO_THROW(from_buffer >> model);

    std::istringstream stream(blob);
    ModelDeserializer from_stream(stream, core, CacheDecrypt{}, false);
    EXPECT_NO_THROW(from_stream >> model);
}

TEST(GraphSerializerDeserializer, RejectsIncompleteHeader) {
    auto buffer = std::make_shared<ov::AlignedBuffer>(sizeof(DataHeader) - 1);
    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    std::shared_ptr<ov::Model> model;

    ModelDeserializer from_buffer(buffer, core, CacheDecrypt{}, false);
    expect_invalid_header(from_buffer, model);

    std::istringstream stream(std::string(sizeof(DataHeader) - 1, '\0'));
    ModelDeserializer from_stream(stream, core, CacheDecrypt{}, false);
    expect_invalid_header(from_stream, model);
}

class ReportedPositionsBuf final : public std::stringbuf {
public:
    ReportedPositionsBuf(std::string contents, std::streamoff begin, std::streamoff end)
        : std::stringbuf(std::move(contents), std::ios::in),
          m_begin(begin),
          m_end(end) {}

protected:
    std::streampos seekoff(std::streamoff offset,
                           std::ios_base::seekdir direction,
                           std::ios_base::openmode which) override {
        if ((which & std::ios::in) && offset == 0) {
            if (direction == std::ios::end) {
                m_at_end = true;
                return std::streampos(m_end);
            }
            if (direction == std::ios::cur) {
                return std::streampos(m_at_end ? m_end : m_begin);
            }
        }
        return std::stringbuf::seekoff(offset, direction, which);
    }

    std::streampos seekpos(std::streampos position, std::ios_base::openmode which) override {
        m_at_end = false;
        return std::stringbuf::seekpos(position, which);
    }

private:
    std::streamoff m_begin;
    std::streamoff m_end;
    bool m_at_end = false;
};

TEST(GraphSerializerDeserializer, RejectsFailedInitialStreamPosition) {
    std::istringstream stream(std::string(sizeof(DataHeader) + 1, 'x'));
    stream.setstate(std::ios::badbit);
    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    std::shared_ptr<ov::Model> model;
    ModelDeserializer deserializer(stream, core, CacheDecrypt{}, false);
    expect_invalid_header(deserializer, model);
}

TEST(GraphSerializerDeserializer, RejectsEndBeforeHeaderPosition) {
    ReportedPositionsBuf positions("x", 100, 99);
    std::istream stream(&positions);
    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    std::shared_ptr<ov::Model> model;
    ModelDeserializer deserializer(stream, core, CacheDecrypt{}, false);
    expect_invalid_header(deserializer, model);
}

TEST(GraphSerializerDeserializer, RejectsTruncatedStreamHeader) {
    ReportedPositionsBuf positions("x", 0, sizeof(DataHeader) + 1);
    std::istream stream(&positions);
    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    std::shared_ptr<ov::Model> model;
    ModelDeserializer deserializer(stream, core, CacheDecrypt{}, false);
    expect_invalid_header(deserializer, model);
}

TEST(GraphSerializerDeserializer, ReadsSectionsRelativeToStreamPosition) {
    DataHeader header{};
    header.custom_data_offset = sizeof(header);
    header.consts_offset = sizeof(header);
    header.model_offset = sizeof(header);
    const auto blob = make_blob(header, sizeof(header) + 1);

    auto core = std::make_shared<testing::StrictMock<ov::MockICore>>();
    EXPECT_CALL(*core, read_model(testing::_, testing::_))
        .WillOnce([](const std::shared_ptr<ov::AlignedBuffer>& xml, const std::shared_ptr<ov::AlignedBuffer>& weights) {
            EXPECT_EQ(xml->size(), 1);
            EXPECT_EQ(*xml->get_ptr<char>(), 'x');
            EXPECT_EQ(weights->size(), 0);
            return std::shared_ptr<ov::Model>{};
        });

    std::istringstream stream(std::string(8, 'p') + blob);
    stream.seekg(8);
    std::shared_ptr<ov::Model> model;
    ModelDeserializer from_stream(stream, core, CacheDecrypt{}, false);
    EXPECT_NO_THROW(from_stream >> model);
}

}  // namespace
}  // namespace ov::intel_cpu
