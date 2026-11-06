// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

std::string collect(const hsm::ISectionEncoder& encoder) {
    std::string encoded;
    encoder.encode([&encoded](ov::util::MemoryView chunk) {
        encoded.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
    });
    return encoded;
}

class XmlSource final : public hsm::IModelStructSource {
public:
    void write_structure(const hsm::SectionSink& sink) override {
        hsm::encode_model_struct(xml, sink);
    }

    std::string xml;
};

}  // namespace

TEST(HsmModelStructTest, round_trips_xml) {
    const std::string expected = "<?xml version=\"1.0\"?><net name=\"test\">\n<layers/>\n</net>";
    std::string encoded;
    hsm::encode_model_struct(expected, [&encoded](ov::util::MemoryView chunk) {
        encoded.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
    });
    EXPECT_EQ(encoded, expected);

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = encoded.size();
    const hsm::Section section{entry, {reinterpret_cast<const std::byte*>(encoded.data()), encoded.size()}};
    EXPECT_EQ(hsm::decode_model_struct(section), expected);
}

TEST(HsmModelStructTest, round_trips_empty_xml) {
    std::string encoded;
    hsm::encode_model_struct({}, [&encoded](ov::util::MemoryView chunk) {
        encoded.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
    });
    EXPECT_TRUE(encoded.empty());

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    const hsm::Section section{entry, ov::util::MemoryView{}};
    EXPECT_EQ(hsm::decode_model_struct(section), std::string{});
}

TEST(HsmModelStructTest, encoder_reads_xml_at_encode_time) {
    auto source = std::make_shared<XmlSource>();
    hsm::ModelStructEncoder encoder{source};
    source->xml = "<net name=\"deferred\"/>";
    source.reset();

    EXPECT_EQ(collect(encoder), "<net name=\"deferred\"/>");
}

TEST(HsmModelStructTest, decodes_stream_section) {
    const std::string expected = "<net/>";
    std::istringstream stream{expected};
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = expected.size();
    const hsm::Section section{entry, stream, std::streampos{0}};

    EXPECT_EQ(hsm::decode_model_struct(section), expected);
}

}  // namespace ov::test
