// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "orc.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ov::npuw::orc;

// Small sequential TypeIds that fit in u16.
constexpr TypeId TYPE_NCMD = 0x0001u;
constexpr TypeId TYPE_META = 0x0002u;
constexpr TypeId TYPE_PART = 0x0003u;
constexpr TypeId TYPE_WGHT = 0x0004u;
constexpr TypeId TYPE_TEST = 0x0005u;
constexpr TypeId TYPE_UNKNOWN = 0xFFFFu;

const SchemaUUID TEST_UUID =
    {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};

struct BlobSummary {
    std::uint32_t subgraph_count = 0u;
    std::vector<std::string> devices;
};

void serialize(Stream& stream, BlobSummary& value) {
    stream & value.subgraph_count & value.devices;
}

struct MetaV1 {
    static constexpr Version kVersion = 1u;

    std::uint32_t subgraph_count = 0u;
    std::vector<std::string> devices;

    void serialize(Stream& stream) {
        stream & subgraph_count & devices;
    }
};

struct MetaV2 : MetaV1 {
    ORC_DECLARE_VERSION(MetaV2, MetaV1)

    bool weightless = false;

    void serialize(Stream& stream) {
        Prev::serialize(stream);
        stream & weightless;
    }
};

struct MetaV3 : MetaV2 {
    ORC_DECLARE_VERSION(MetaV3, MetaV2)

    std::string layout;

    void serialize(Stream& stream) {
        Prev::serialize(stream);
        stream & layout;
    }
};

void expect_blob_summary(const BlobSummary& expected, const BlobSummary& actual) {
    EXPECT_EQ(expected.subgraph_count, actual.subgraph_count);
    EXPECT_EQ(expected.devices, actual.devices);
}

struct CounterMeta {
    static constexpr Version kVersion = 0u;
    std::uint32_t value = 0u;

    // Tiny fixed payload used to prove that scoped sections can stream inline metadata
    // before nested ORC children without going through the higher-level Section tree API.
    void serialize(Stream& stream) {
        stream & value;
    }
};

}  // namespace

TEST(OrcTest, FileRoundTripsNestedSections) {
    const BlobSummary meta{2u, {"NPU"}};
    const BlobSummary part{7u, {"prefill", "decode"}};
    const BlobSummary wght{1u, {"lazy"}};

    const auto root =
        Section::container(TYPE_NCMD,
                           1u,
                           {make_payload_section(TYPE_META, 1u, meta),
                            Section::container(TYPE_PART, 1u, {make_payload_section(TYPE_TEST, 1u, part)}),
                            make_payload_section(TYPE_WGHT, 1u, wght)});

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    write_file(buffer, root, TEST_UUID);

    const auto decoded = read_file(buffer);
    ASSERT_TRUE(decoded.is_container());
    ASSERT_EQ(decoded.type, TYPE_NCMD);
    ASSERT_EQ(decoded.children.size(), 3u);

    EXPECT_EQ(decoded.children[0].type, TYPE_META);
    expect_blob_summary(meta, decode<BlobSummary>(decoded.children[0].payload));

    EXPECT_TRUE(decoded.children[1].is_container());
    ASSERT_EQ(decoded.children[1].children.size(), 1u);
    expect_blob_summary(part, decode<BlobSummary>(decoded.children[1].children[0].payload));

    EXPECT_EQ(decoded.children[2].type, TYPE_WGHT);
    expect_blob_summary(wght, decode<BlobSummary>(decoded.children[2].payload));
}

TEST(OrcTest, RejectsTruncatedFile) {
    const auto root =
        Section::container(TYPE_NCMD, 1u, {make_payload_section(TYPE_META, 1u, BlobSummary{3u, {"NPU"}})});

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    write_file(buffer, root, TEST_UUID);

    auto bytes = buffer.str();
    ASSERT_FALSE(bytes.empty());
    bytes.pop_back();

    std::stringstream truncated(bytes, std::ios::in | std::ios::out | std::ios::binary);
    EXPECT_THROW(read_file(truncated), ov::Exception);
}

TEST(OrcTest, ScopedSectionsRoundTripMetadataBeforeChildren) {
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    write_file_header(buffer, TEST_UUID);
    with_section(buffer, TYPE_NCMD, CounterMeta::kVersion, 0u, [&] {
        auto root_stream = Stream::writer(buffer);
        CounterMeta root_meta{2u};
        root_stream & root_meta;

        with_section(buffer, TYPE_TEST, CounterMeta::kVersion, 0u, [&] {
            auto child_stream = Stream::writer(buffer);
            CounterMeta child_meta{7u};
            child_stream & child_meta;
        });
    });

    EXPECT_EQ(read_file_header(buffer).schema_uuid, TEST_UUID);

    ScopedReadSection root(buffer);
    EXPECT_EQ(root.header().type, TYPE_NCMD);
    auto root_stream = Stream::reader(buffer);
    CounterMeta root_meta;
    root_stream & root_meta;
    EXPECT_EQ(root_meta.value, 2u);

    ScopedReadSection child(buffer);
    EXPECT_EQ(child.header().type, TYPE_TEST);
    auto child_stream = Stream::reader(buffer);
    CounterMeta child_meta;
    child_stream & child_meta;
    EXPECT_EQ(child_meta.value, 7u);
    child.expect_end();
    root.expect_end();
}

TEST(OrcTest, IsOrcReturnsTrueForValidBlob) {
    const auto root =
        Section::container(TYPE_NCMD, 1u, {make_payload_section(TYPE_META, 1u, BlobSummary{1u, {"NPU"}})});

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    write_file(buffer, root, TEST_UUID);

    // is_orc must return a header and leave the stream at its original position
    const auto header = is_orc(buffer);
    ASSERT_TRUE(header.has_value());
    EXPECT_EQ(header->version, 0u);
    EXPECT_EQ(header->schema_uuid, TEST_UUID);
    // read_file must still work after the probe
    EXPECT_NO_THROW(read_file(buffer));
}

TEST(OrcTest, IsOrcReturnsFalseForGarbage) {
    std::stringstream buffer("not an orc blob", std::ios::in | std::ios::out | std::ios::binary);
    EXPECT_FALSE(is_orc(buffer).has_value());
}

TEST(OrcTest, TryReadBytesAdvancesOnSuccessAndRollsBackOnFailure) {
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    buffer.write("abcdef", 6);

    std::array<char, 3> prefix{};
    EXPECT_TRUE(try_read_bytes(buffer, prefix.data(), prefix.size()));
    EXPECT_EQ(std::string(prefix.data(), prefix.size()), "abc");
    EXPECT_EQ(buffer.tellg(), std::streampos{3});

    std::array<char, 8> oversized{};
    const auto pos_before_failure = buffer.tellg();
    EXPECT_FALSE(try_read_bytes(buffer, oversized.data(), oversized.size()));
    EXPECT_EQ(buffer.tellg(), pos_before_failure);
}

TEST(OrcTest, SchemaSkipsUnknownOptionalChildren) {
    const auto root = Section::container(TYPE_NCMD,
                                         1u,
                                         {make_payload_section(TYPE_META, 1u, BlobSummary{2u, {"NPU"}}),
                                          Section::raw(TYPE_UNKNOWN,
                                                       1u,
                                                       std::vector<std::byte>{std::byte{0xAB}, std::byte{0xCD}},
                                                       static_cast<SectionFlags>(SectionFlag::OPTIONAL)),
                                          make_payload_section(TYPE_WGHT, 1u, BlobSummary{1u, {"lazy"}})});

    Schema schema;
    schema.register_loader<BlobSummary>(TYPE_META, [](const Section& section, const Schema&) {
        return decode<BlobSummary>(section.payload);
    });
    schema.register_loader<BlobSummary>(TYPE_WGHT, [](const Section& section, const Schema&) {
        return decode<BlobSummary>(section.payload);
    });

    const auto children = schema.load_children(root);
    ASSERT_EQ(children.size(), 2u);
    EXPECT_EQ(children[0].type, TYPE_META);
    EXPECT_EQ(children[1].type, TYPE_WGHT);

    expect_blob_summary(BlobSummary{2u, {"NPU"}}, std::any_cast<BlobSummary>(children[0].value));
    expect_blob_summary(BlobSummary{1u, {"lazy"}}, std::any_cast<BlobSummary>(children[1].value));
}

TEST(OrcTest, SchemaRejectsUnknownRequiredChildren) {
    const auto root = Section::container(TYPE_NCMD,
                                         1u,
                                         {make_payload_section(TYPE_META, 1u, BlobSummary{2u, {"NPU"}}),
                                          Section::raw(TYPE_UNKNOWN, 1u, std::vector<std::byte>{std::byte{0xAA}})});

    Schema schema;
    schema.register_loader<BlobSummary>(TYPE_META, [](const Section& section, const Schema&) {
        return decode<BlobSummary>(section.payload);
    });

    EXPECT_THROW(schema.load_children(root), ov::Exception);
}

TEST(OrcTest, SchemaRejectsDuplicateRegistration) {
    Schema schema;
    schema.register_loader<BlobSummary>(TYPE_META, [](const Section& section, const Schema&) {
        return decode<BlobSummary>(section.payload);
    });

    EXPECT_THROW(
        {
            schema.register_loader<BlobSummary>(TYPE_META, [](const Section& section, const Schema&) {
                return decode<BlobSummary>(section.payload);
            });
        },
        ov::Exception);
}

TEST(OrcTest, SchemaRejectsTypedMismatch) {
    const auto section = make_payload_section(TYPE_META, 1u, BlobSummary{2u, {"NPU"}});

    Schema schema;
    schema.register_loader<BlobSummary>(TYPE_META, [](const Section& payload, const Schema&) {
        return decode<BlobSummary>(payload.payload);
    });

    EXPECT_THROW(schema.load<std::string>(section), ov::Exception);
}

TEST(OrcTest, LoadVersionedPayloadMigratesAcrossVersions) {
    Schema schema;
    schema.register_loader<MetaV3>(TYPE_META, [](const Section& section, const Schema&) {
        return load_versioned_payload<MetaV3, MetaV1, MetaV2, MetaV3>(section);
    });

    const auto meta_v1 =
        schema.load<MetaV3>(make_payload_section(TYPE_META, MetaV1::kVersion, MetaV1{3u, {"NPU", "CPU"}}));
    EXPECT_EQ(meta_v1.subgraph_count, 3u);
    EXPECT_EQ(meta_v1.devices, (std::vector<std::string>{"NPU", "CPU"}));
    EXPECT_FALSE(meta_v1.weightless);
    EXPECT_TRUE(meta_v1.layout.empty());

    MetaV2 v2;
    v2.subgraph_count = 4u;
    v2.devices = {"NPU"};
    v2.weightless = true;
    const auto meta_v2 = schema.load<MetaV3>(make_payload_section(TYPE_META, MetaV2::kVersion, v2));
    EXPECT_EQ(meta_v2.subgraph_count, 4u);
    EXPECT_EQ(meta_v2.devices, (std::vector<std::string>{"NPU"}));
    EXPECT_TRUE(meta_v2.weightless);
    EXPECT_TRUE(meta_v2.layout.empty());

    MetaV3 v3;
    v3.subgraph_count = 5u;
    v3.devices = {"NPU", "GPU"};
    v3.weightless = true;
    v3.layout = "split";
    const auto meta_v3 = schema.load<MetaV3>(make_payload_section(TYPE_META, MetaV3::kVersion, v3));
    EXPECT_EQ(meta_v3.subgraph_count, 5u);
    EXPECT_EQ(meta_v3.devices, (std::vector<std::string>{"NPU", "GPU"}));
    EXPECT_TRUE(meta_v3.weightless);
    EXPECT_EQ(meta_v3.layout, "split");
}

TEST(OrcTest, LoadVersionedPayloadRejectsUnsupportedVersion) {
    Schema schema;
    schema.register_loader<MetaV3>(TYPE_META, [](const Section& section, const Schema&) {
        return load_versioned_payload<MetaV3, MetaV1, MetaV2, MetaV3>(section);
    });

    const auto unsupported = make_payload_section(TYPE_META, 99u, MetaV1{1u, {"NPU"}});
    EXPECT_THROW(schema.load<MetaV3>(unsupported), ov::Exception);
}
