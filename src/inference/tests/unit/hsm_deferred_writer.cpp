// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_deferred_writer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "openvino/core/except.hpp"

// DeferredWriter-specific tests
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

constexpr hsm::DeviceId fake_device_id = 7;
constexpr size_t k_buffer_capacity = 4096;

ov::util::MemoryView view_of(const std::string& s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

// Encodes an owned copy of a std::string - safe even if the source is destroyed before finalize().
class OwnedStringEncoder final : public hsm::ISectionEncoder {
public:
    explicit OwnedStringEncoder(std::string text) : m_text(std::move(text)) {}
    void encode(const hsm::SectionSink& sink) const override {
        sink({reinterpret_cast<const std::byte*>(m_text.data()), m_text.size()});
    }

private:
    std::string m_text;
};

// Encodes a live std::string by reference - the referenced string must outlive finalize().
class StringViewEncoder final : public hsm::ISectionEncoder {
public:
    explicit StringViewEncoder(const std::string& text) : m_text(text) {}
    void encode(const hsm::SectionSink& sink) const override {
        sink({reinterpret_cast<const std::byte*>(m_text.data()), m_text.size()});
    }

private:
    const std::string& m_text;
};

// Writes "chunk0-chunk1-chunk2-" across three separate sink() calls.
class ChunkedEncoder final : public hsm::ISectionEncoder {
public:
    void encode(const hsm::SectionSink& sink) const override {
        for (int i = 0; i < 3; ++i) {
            const std::string chunk = "chunk" + std::to_string(i) + "-";
            sink({reinterpret_cast<const std::byte*>(chunk.data()), chunk.size()});
        }
    }
};

// Does nothing - never calls sink() at all.
class NoOpEncoder final : public hsm::ISectionEncoder {
public:
    void encode(const hsm::SectionSink&) const override {}
};

// Sinks `count` zero bytes in one call.
class ZeroBytesEncoder final : public hsm::ISectionEncoder {
public:
    explicit ZeroBytesEncoder(size_t count) : m_bytes(count) {}
    void encode(const hsm::SectionSink& sink) const override {
        sink({m_bytes.data(), m_bytes.size()});
    }

private:
    std::vector<std::byte> m_bytes;
};

// Sinks a declared `size` larger than the single real byte behind `data` - the overflow/capacity check
// must reject this before any copy happens.
class OverflowEncoder final : public hsm::ISectionEncoder {
public:
    OverflowEncoder(const std::byte* data, size_t size) : m_data(data), m_size(size) {}
    void encode(const hsm::SectionSink& sink) const override {
        sink({m_data, m_size});
    }

private:
    const std::byte* m_data;
    size_t m_size;
};

// Always throws - simulates an encoder that blows up.
class ThrowingEncoder final : public hsm::ISectionEncoder {
public:
    void encode(const hsm::SectionSink&) const override {
        OPENVINO_THROW("encoder blew up");
    }
};

// Counts invocations; throws while `should_throw` is true, otherwise sinks a fixed 4-byte payload.
class RetryEncoder final : public hsm::ISectionEncoder {
public:
    RetryEncoder(int& call_count, bool& should_throw) : m_call_count(call_count), m_should_throw(should_throw) {}
    void encode(const hsm::SectionSink& sink) const override {
        ++m_call_count;
        if (m_should_throw) {
            OPENVINO_THROW("encoder blew up");
        }
        sink({reinterpret_cast<const std::byte*>("data"), 4});
    }

private:
    int& m_call_count;
    bool& m_should_throw;
};

// Sinks a single empty chunk.
class EmptyChunkEncoder final : public hsm::ISectionEncoder {
public:
    void encode(const hsm::SectionSink& sink) const override {
        sink(ov::util::MemoryView{});
    }
};

hsm::DeferredWriter open_writer(std::ostream& stream) {
    auto writer = hsm::DeferredWriter::open(stream);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

hsm::DeferredWriter open_buffer_writer(std::byte* dst, size_t size) {
    auto writer = hsm::DeferredWriter::open(dst, size);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

// Fails (writing only half the requested bytes) on a chosen xsputn() call, simulating a stream write
// that partially succeeds before std::ostream sets badbit - built on std::stringbuf so seeking still
// works, letting finalize()'s retry be observed.
class FlakyOnceStreamBuf : public std::stringbuf {
public:
    int fail_on_call = -1;  // -1 = never fail
    int call_count = 0;

protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (++call_count == fail_on_call) {
            const auto partial = n / 2;
            std::stringbuf::xsputn(s, partial);
            return partial;
        }
        return std::stringbuf::xsputn(s, n);
    }
};

class AppendOnlyStreamBuf : public std::stringbuf {
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        seekoff(0, std::ios::end, std::ios::out);
        return std::stringbuf::xsputn(s, n);
    }
};

struct ParsedContainer {
    hsm::Header header{};
    std::vector<hsm::ManifestEntry> entries;
    std::vector<std::byte> bytes;
};

std::optional<ParsedContainer> parse_container(std::vector<std::byte> raw) {
    if (raw.size() < sizeof(hsm::Header)) {
        return std::nullopt;
    }
    const auto& header = hsm::Header::view(reinterpret_cast<const uint8_t*>(raw.data()));
    if (!hsm::is_valid_header_fields(header)) {
        return std::nullopt;
    }
    const auto first_entry = reinterpret_cast<const hsm::ManifestEntry*>(raw.data() + header.manifest_offset);
    std::vector<hsm::ManifestEntry> entries(first_entry,
                                            first_entry + header.manifest_size / sizeof(hsm::ManifestEntry));
    return ParsedContainer{header, std::move(entries), std::move(raw)};
}

std::optional<ParsedContainer> parse_container(const std::string& stream_bytes) {
    const auto data = reinterpret_cast<const std::byte*>(stream_bytes.data());
    return parse_container(std::vector<std::byte>(data, data + stream_bytes.size()));
}

std::optional<ParsedContainer> parse_container(const std::byte* data, size_t size) {
    return parse_container(std::vector<std::byte>(data, data + size));
}

const hsm::ManifestEntry* find_entry(const ParsedContainer& container, hsm::DeviceId device, uint32_t tag_id) {
    for (const auto& entry : container.entries) {
        if (entry.device == device && entry.tag.id() == tag_id) {
            return &entry;
        }
    }
    return nullptr;
}

std::vector<const hsm::ManifestEntry*> find_entries(const ParsedContainer& container,
                                                    hsm::DeviceId device,
                                                    uint32_t tag_id) {
    std::vector<const hsm::ManifestEntry*> found;
    for (const auto& entry : container.entries) {
        if (entry.device == device && entry.tag.id() == tag_id) {
            found.push_back(&entry);
        }
    }
    return found;
}

std::string payload_string(const ParsedContainer& container, const hsm::ManifestEntry& entry) {
    if (entry.tag.is_inline()) {
        return {reinterpret_cast<const char*>(entry.inline_bytes.data()), entry.inline_bytes.size()};
    }
    return {reinterpret_cast<const char*>(container.bytes.data() + entry.offset), entry.size};
}

}  // namespace

static_assert(std::is_move_constructible_v<hsm::DeferredWriter>);
static_assert(std::is_move_assignable_v<hsm::DeferredWriter>);
static_assert(!std::is_copy_constructible_v<hsm::DeferredWriter>);
static_assert(!std::is_copy_assignable_v<hsm::DeferredWriter>);

// --- open() factories ---

TEST(HsmDeferredWriterTest, open_stream_rejects_an_already_failed_stream) {
    std::stringstream stream;
    stream.setstate(std::ios::badbit);
    EXPECT_FALSE(hsm::DeferredWriter::open(stream).has_value());
}

TEST(HsmDeferredWriterTest, open_stream_rejects_a_non_seekable_stream) {
    struct NonSeekableStreamBuf : std::streambuf {};  // base class's seekoff/seekpos always fail
    NonSeekableStreamBuf buf;
    std::ostream stream(&buf);
    EXPECT_FALSE(hsm::DeferredWriter::open(stream).has_value());
}

TEST(HsmDeferredWriterTest, open_buffer_rejects_a_buffer_too_small_for_the_header) {
    std::vector<std::byte> too_small(sizeof(hsm::Header) - 1);
    EXPECT_FALSE(hsm::DeferredWriter::open(too_small.data(), too_small.size()).has_value());
}

TEST(HsmDeferredWriterTest, open_buffer_rejects_a_null_buffer) {
    EXPECT_FALSE(hsm::DeferredWriter::open(nullptr, 0).has_value());
}

// --- Writing into the two destination kinds ---

TEST(HsmDeferredWriterTest, writes_into_a_preallocated_buffer) {
    const std::string model = "model-bytes";

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "model-bytes");
}

TEST(HsmDeferredWriterTest, pads_an_alignment_gap_larger_than_the_internal_zero_fill_chunk) {
    // write_zeros() fills padding gaps via a small internal buffer reused in a loop - this forces a gap
    // bigger than that buffer, exercising more than one iteration of it.
    const std::string a = "a";
    const std::string b = "b";
    const auto b_tag = hsm::SectionTag::make_device_tag(5, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/1024});
    writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/1024});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto a_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    const auto b_entry = find_entry(*container, fake_device_id, b_tag.id());
    ASSERT_NE(a_entry, nullptr);
    ASSERT_NE(b_entry, nullptr);
    EXPECT_EQ(b_entry->offset, a_entry->offset + 1024);
    EXPECT_EQ(payload_string(*container, *a_entry), "a");
    EXPECT_EQ(payload_string(*container, *b_entry), "b");
}

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_a_chunk_without_overflowing_the_capacity_check) {
    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    const std::byte dummy{};
    static constexpr size_t huge = std::numeric_limits<size_t>::max() - 31;
    writer.add_section(hsm::any_device_id, hsm::model_tag, huge, std::make_shared<OverflowEncoder>(&dummy, huge));
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

TEST(HsmDeferredWriterTest, sized_section_encoder_fits_a_buffer_sized_to_the_exact_byte) {
    static constexpr size_t section_size = 4;
    constexpr size_t exact_capacity = sizeof(hsm::Header) + section_size + sizeof(hsm::ManifestEntry);
    std::vector<std::byte> buffer(exact_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       section_size,
                       std::make_shared<ZeroBytesEncoder>(section_size));
    EXPECT_EQ(writer.finalize(), std::error_code{});
}

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_a_buffer_one_byte_short_of_fitting) {
    static constexpr size_t section_size = 4;
    constexpr size_t exact_capacity = sizeof(hsm::Header) + section_size + sizeof(hsm::ManifestEntry);
    std::vector<std::byte> buffer(exact_capacity - 1);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       section_size,
                       std::make_shared<ZeroBytesEncoder>(section_size));
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

TEST(HsmDeferredWriterTest, finalize_reports_write_failed_for_an_undersized_buffer) {
    std::vector<std::byte> too_small(sizeof(hsm::Header));  // no room for even one section
    auto writer = open_buffer_writer(too_small.data(), too_small.size());
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

TEST(HsmDeferredWriterTest, finalize_s_returned_error_code_has_a_human_readable_message) {
    std::vector<std::byte> too_small(sizeof(hsm::Header));
    auto writer = open_buffer_writer(too_small.data(), too_small.size());
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));

    const auto ec = writer.finalize();
    EXPECT_EQ(ec, hsm::WriteErrc::write_failed);
    EXPECT_FALSE(ec.message().empty());
}

TEST(HsmDeferredWriterTest, finalize_reports_write_failed_when_the_stream_is_already_bad) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    stream.setstate(std::ios::badbit);  // fails every subsequent write
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

// --- add_section() validates its inputs ---

#ifndef NDEBUG
TEST(HsmDeferredWriterTest, sized_section_encoder_debug_asserts_on_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, 4, std::make_shared<NoOpEncoder>()),
                 ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_debug_asserts_on_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, std::make_shared<NoOpEncoder>()),
                 ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, sized_section_encoder_debug_asserts_on_a_null_encoder) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_tag, 4, nullptr), ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_debug_asserts_on_a_null_encoder) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_tag, nullptr), ov::AssertFailure);
}
#else
TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_FALSE(writer.add_section(hsm::any_device_id, hsm::model_id_tag, 4, std::make_shared<NoOpEncoder>()));
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_rejects_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_FALSE(writer.add_section(hsm::any_device_id, hsm::model_id_tag, std::make_shared<NoOpEncoder>()));
}

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_a_null_encoder) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_FALSE(writer.add_section(hsm::any_device_id, hsm::model_tag, 4, nullptr));
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_rejects_a_null_encoder) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_FALSE(writer.add_section(hsm::any_device_id, hsm::model_tag, nullptr));
}
#endif

// --- finalize() semantics ---

TEST(HsmDeferredWriterTest, finalize_is_idempotent) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    const auto first = writer.finalize();
    const auto written_once = stream.str();
    const auto second = writer.finalize();

    EXPECT_EQ(first, second);
    EXPECT_EQ(stream.str(), written_once);  // no bytes written again
}

TEST(HsmDeferredWriterTest, finalize_propagates_an_exception_thrown_by_a_section_encoder) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, 4, std::make_shared<ThrowingEncoder>());
    EXPECT_THROW(writer.finalize(), ov::Exception);
}

TEST(HsmDeferredWriterTest, finalize_retries_the_encoder_after_a_thrown_attempt_and_can_then_succeed) {
    int call_count = 0;
    bool should_throw = true;
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, 4, std::make_shared<RetryEncoder>(call_count, should_throw));
    EXPECT_THROW(writer.finalize(), ov::Exception);
    EXPECT_EQ(call_count, 1);

    should_throw = false;  // the transient issue is gone by the next attempt
    EXPECT_EQ(writer.finalize(), std::error_code{});
    EXPECT_EQ(call_count, 2);
}

TEST(HsmDeferredWriterTest, finalize_retries_correctly_after_a_stream_write_throws_partway) {
    FlakyOnceStreamBuf buf;
    buf.fail_on_call = 2;  // the header write succeeds; the payload write fails partway through
    std::ostream stream(&buf);
    stream.exceptions(std::ios::badbit);

    auto writer = open_writer(stream);
    const std::string model = "model-bytes";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    EXPECT_THROW(writer.finalize(), std::ios_base::failure);

    stream.clear();
    buf.fail_on_call = -1;  // the transient issue is gone by the next attempt
    EXPECT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buf.str());
    ASSERT_TRUE(container.has_value());
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "model-bytes");
}

TEST(HsmDeferredWriterTest, finalize_retries_correctly_after_the_header_write_itself_throws_partway) {
    FlakyOnceStreamBuf buf;
    buf.fail_on_call = 1;  // the header write itself fails partway through, before size is ever updated
    std::ostream stream(&buf);
    stream.exceptions(std::ios::badbit);

    auto writer = open_writer(stream);
    const std::string model = "model-bytes";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    EXPECT_THROW(writer.finalize(), std::ios_base::failure);

    stream.clear();
    buf.fail_on_call = -1;  // the transient issue is gone by the next attempt
    EXPECT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buf.str());
    ASSERT_TRUE(container.has_value());
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "model-bytes");
}

TEST(HsmDeferredWriterTest, finalize_caches_a_normal_non_throwing_failure_instead_of_retrying) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    stream.setstate(std::ios::badbit);
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);

    stream.clear();
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);  // cached, not re-derived
}

TEST(HsmDeferredWriterTest, finalize_rejects_a_sized_layout_whose_slot_computation_would_overflow) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string payload = "x";
    constexpr size_t huge_align = size_t{1} << 63;  // aligned start and aligned size both become 1<<63
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(payload), {huge_align, huge_align});
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
    EXPECT_TRUE(stream.str().empty());  // rejected before writing anything, not after a wrapped-around size
}

TEST(HsmDeferredWriterTest, finalize_detects_a_misplaced_patch_on_an_append_only_stream) {
    AppendOnlyStreamBuf buf;
    std::ostream stream(&buf);
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       std::make_shared<OwnedStringEncoder>("discovered-at-write-time"));
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

TEST(HsmDeferredWriterTest, unsized_section_forces_the_header_to_be_patched_after_the_fact) {
    // With only sized content, the header is computed in one pass up front. An unsized section forces the
    // placeholder-then-patch fallback - verified here by confirming the final header is still correct.
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       std::make_shared<OwnedStringEncoder>("discovered-at-write-time"));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(hsm::is_valid_header_fields(container->header));
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "discovered-at-write-time");
}

TEST(HsmDeferredWriterTest, patches_the_unsized_section_header_relative_to_a_nonzero_stream_start) {
    std::stringstream stream;
    const std::string prefix = "PREFIX-BYTES";
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));

    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       std::make_shared<OwnedStringEncoder>("discovered-at-write-time"));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto whole = stream.str();
    ASSERT_EQ(whole.compare(0, prefix.size(), prefix), 0);  // prefix left untouched by the patch
    const auto container = parse_container(whole.substr(prefix.size()));
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(hsm::is_valid_header_fields(container->header));
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "discovered-at-write-time");
}

TEST(HsmDeferredWriterTest, empty_inline_payload_is_written_without_undefined_behavior) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_id_tag, ov::util::MemoryView{});
    EXPECT_EQ(writer.finalize(), std::error_code{});
}

TEST(HsmDeferredWriterTest, empty_pointer_mode_payload_is_written_without_undefined_behavior) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, ov::util::MemoryView{});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->size, 0u);
}

TEST(HsmDeferredWriterTest, empty_encoder_chunk_is_written_without_undefined_behavior) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, 0, std::make_shared<EmptyChunkEncoder>());
    EXPECT_EQ(writer.finalize(), std::error_code{});
}

// --- add_section() view overload ---

TEST(HsmDeferredWriterTest, empty_container_is_valid_and_has_no_sections) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(container->entries.empty());
    EXPECT_EQ(find_entry(*container, hsm::any_device_id, hsm::model), nullptr);
}

TEST(HsmDeferredWriterTest, round_trips_inline_and_pointer_sections) {
    const std::string id = "id";
    const std::string model = "compiled-model-bytes";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id));  // inline-mode Core tag
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));  // pointer-mode Core tag
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    // Inline entries always carry the full 24-byte slot; the writer zero-fills past the payload.
    const auto read_id = find_entry(*container, hsm::any_device_id, hsm::model_id);
    ASSERT_NE(read_id, nullptr);
    EXPECT_TRUE(read_id->tag.is_inline());
    EXPECT_EQ(payload_string(*container, *read_id).substr(0, 2), "id");

    const auto read_model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(read_model, nullptr);
    EXPECT_TRUE(read_model->tag.is_pointer());
    EXPECT_EQ(payload_string(*container, *read_model), "compiled-model-bytes");
}

TEST(HsmDeferredWriterTest, tag_reserved_bytes_round_trip_and_default_to_zero) {
    const std::string payload = "device-payload";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false);
    const std::array<uint8_t, 4> reserved{0x01, 0x02, 0x03, 0x04};

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, hsm::SectionTagReserved{shard_tag, reserved}, view_of(payload));
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(payload));  // bare SectionTag still compiles
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    const auto shard = find_entry(*container, fake_device_id, shard_tag.id());
    ASSERT_NE(shard, nullptr);
    EXPECT_EQ(shard->tag_reserved, reserved);

    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag_reserved, (std::array<uint8_t, 4>{}));
}

TEST(HsmDeferredWriterTest, device_specific_section_is_scoped_to_its_device) {
    const std::string payload = "device-payload";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false);

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, shard_tag, view_of(payload));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    EXPECT_EQ(find_entry(*container, hsm::any_device_id, shard_tag.id()), nullptr);  // scoped to its device
    const auto shard = find_entry(*container, fake_device_id, shard_tag.id());
    ASSERT_NE(shard, nullptr);
    EXPECT_EQ(payload_string(*container, *shard), "device-payload");
}

TEST(HsmDeferredWriterTest, aligns_pointer_section_offset_and_pads_slot_to_aligned_size) {
    const std::string weights = "weights";  // 7 bytes; slot padded to a multiple of 64
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64});
    // offset_align 1: its offset reveals whether the previous slot's size was padded to 64.
    writer.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());

    const auto weights_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(weights_entry, nullptr);
    EXPECT_EQ(weights_entry->offset % 64, 0u);
    EXPECT_EQ(payload_string(*container, *weights_entry), "weights");

    const auto tail_entry = find_entry(*container, fake_device_id, tail_tag.id());
    ASSERT_NE(tail_entry, nullptr);
    // Starts after the previous slot padded up to 64 bytes, not right after the 7 payload bytes.
    EXPECT_EQ(tail_entry->offset, weights_entry->offset + 64);
}

TEST(HsmDeferredWriterTest, zero_alignment_behaves_the_same_as_one) {
    const std::string a = "aa";
    const std::string b = "bbb";
    const auto b_tag = hsm::SectionTag::make_device_tag(4, false);

    std::stringstream zero_stream;
    auto zero_writer = open_writer(zero_stream);
    zero_writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/0});
    zero_writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/0});
    ASSERT_EQ(zero_writer.finalize(), std::error_code{});

    std::stringstream one_stream;
    auto one_writer = open_writer(one_stream);
    one_writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/1});
    one_writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/1});
    ASSERT_EQ(one_writer.finalize(), std::error_code{});

    EXPECT_EQ(zero_stream.str(), one_stream.str());
}

TEST(HsmDeferredWriterTest, size_align_pads_the_slot_independently_of_offset_align) {
    const std::string weights = "weights";  // 7 bytes
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    // Offset aligned to 64, but the slot itself only needs to be padded to a multiple of 16.
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64, /*size_align=*/16});
    writer.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());

    const auto weights_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(weights_entry, nullptr);
    EXPECT_EQ(weights_entry->offset % 64, 0u);

    const auto tail_entry = find_entry(*container, fake_device_id, tail_tag.id());
    ASSERT_NE(tail_entry, nullptr);
    // Slot padded to 16 (not 64, which the default single-value behavior would have produced).
    EXPECT_EQ(tail_entry->offset, weights_entry->offset + 16);
}

TEST(HsmDeferredWriterTest, preserves_multiple_sections_sharing_one_tag_in_order) {
    const std::string s0 = "shard-0";
    const std::string s1 = "shard-1";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, shard_tag, view_of(s0));
    writer.add_section(fake_device_id, shard_tag, view_of(s1));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto shards = find_entries(*container, fake_device_id, shard_tag.id());
    ASSERT_EQ(shards.size(), 2u);
    EXPECT_EQ(payload_string(*container, *shards[0]), "shard-0");
    EXPECT_EQ(payload_string(*container, *shards[1]), "shard-1");
}

TEST(HsmDeferredWriterTest, rejects_inline_payload_exceeding_entry_capacity) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string too_big(25, 'x');  // inline capacity is 24 bytes
    EXPECT_FALSE(writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(too_big)));
}

TEST(HsmDeferredWriterTest, rejects_non_power_of_two_alignment) {
    const std::string x = "x";
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(x), {/*offset_align=*/3}),
                 ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, inline_section_ignores_a_non_power_of_two_alignment) {
    const std::string id = "x";
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_NO_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id), {/*offset_align=*/3}));
    EXPECT_EQ(writer.finalize(), std::error_code{});
}

TEST(HsmDeferredWriterTest, zero_size_align_inherits_alignment) {
    const std::string weights = "weights";  // 7 bytes
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::stringstream inherited_stream;
    auto inherited = open_writer(inherited_stream);
    inherited.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64});  // size_align=0
    inherited.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_EQ(inherited.finalize(), std::error_code{});

    std::stringstream explicit_stream;
    auto explicit_same = open_writer(explicit_stream);
    explicit_same.add_section(hsm::any_device_id,
                              hsm::model_tag,
                              view_of(weights),
                              {/*offset_align=*/64, /*size_align=*/64});
    explicit_same.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_EQ(explicit_same.finalize(), std::error_code{});

    EXPECT_EQ(inherited_stream.str(), explicit_stream.str());
}

// --- add_section() ISectionEncoder overloads (sized and unsized) ---

TEST(HsmDeferredWriterTest, section_encoder_can_capture_a_copy_of_a_transient_payload) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    {
        const std::string transient = "transient-model";
        writer.add_section(hsm::any_device_id,
                           hsm::model_tag,
                           transient.size(),
                           std::make_shared<OwnedStringEncoder>(transient));
    }  // transient destroyed before finalize() - the encoder already captured its own copy
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "transient-model");
}

TEST(HsmDeferredWriterTest, fill_in_place_writes_generated_payload_into_the_destination) {
    const std::string content = "filled-bytes";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       content.size(),
                       std::make_shared<StringViewEncoder>(content));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "filled-bytes");
}

TEST(HsmDeferredWriterTest, sized_section_encoder_can_split_its_output_into_several_chunks) {
    const std::string expected = "chunk0-chunk1-chunk2-";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, expected.size(), std::make_shared<ChunkedEncoder>());
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), expected);
}

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_writing_past_its_declared_size_via_add_section) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       /*size=*/2,
                       std::make_shared<OwnedStringEncoder>("too-long"));
    EXPECT_THROW(writer.finalize(), ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_writing_less_than_its_declared_size_via_add_section) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       /*size=*/10,
                       std::make_shared<OwnedStringEncoder>("short"));
    EXPECT_THROW(writer.finalize(), ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_discovers_size_after_writing) {
    // Simulates content whose length isn't known until it's actually produced (e.g. serializing a
    // variable-length object) - no size is passed to add_section() at all.
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, std::make_shared<ChunkedEncoder>());
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "chunk0-chunk1-chunk2-");
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_works_with_a_preallocated_buffer_too) {
    const std::string content = "discovered-at-write-time";

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, std::make_shared<StringViewEncoder>(content));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "discovered-at-write-time");
}

// --- add_sections() / ISectionWriterHandler content integration (dispatch logic itself is covered by
// mocks in hsm_writer.cpp - these confirm a real writer produces correct bytes end to end) ---

TEST(HsmDeferredWriterTest, add_sections_lets_a_handler_contribute_its_own_sections) {
    class PluginWriter : public hsm::ISectionWriterHandler {
    public:
        void handle_section(hsm::IWriter& writer) const override {
            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/3, /*is_inline=*/false),
                               view_of(m_payload));
        }

    private:
        std::string m_payload = "plugin-section";
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    PluginWriter plugin;
    writer.add_sections({&plugin});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(3, false);
    const auto section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    EXPECT_EQ(payload_string(*container, *section), "plugin-section");
}

TEST(HsmDeferredWriterTest, section_encoder_avoids_materializing_a_view_for_a_transient_source_at_all) {
    // Builds the view into `value` only inside encode(), not ahead of time at add_section() time.
    class ValueEncoder final : public hsm::ISectionEncoder {
    public:
        explicit ValueEncoder(int value) : m_value(value) {}
        void encode(const hsm::SectionSink& sink) const override {
            sink({reinterpret_cast<const std::byte*>(&m_value), sizeof(m_value)});
        }

    private:
        int m_value;
    };

    class PluginWriter : public hsm::ISectionWriterHandler {
    public:
        void handle_section(hsm::IWriter& writer) const override {
            const int value = 42;
            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/6, /*is_inline=*/false),
                               sizeof(value),
                               std::make_shared<ValueEncoder>(value));
        }
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    PluginWriter plugin;
    writer.add_sections({&plugin});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(6, false);
    const auto section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    int decoded = 0;
    std::memcpy(&decoded, container->bytes.data() + section->offset, sizeof(decoded));
    EXPECT_EQ(decoded, 42);
}

TEST(HsmDeferredWriterTest, handler_based_add_section_reuses_one_handler_for_several_sections_sharing_a_tag) {
    class ShardHandler : public hsm::ISectionWriterHandler {
    public:
        explicit ShardHandler(std::vector<std::string> shards) : m_shards(std::move(shards)) {}

        void handle_section(hsm::IWriter& writer) const override {
            for (size_t i = 0; i < m_shards.size(); ++i) {
                writer.add_section(fake_device_id,
                                   hsm::SectionTag::make_device_tag(/*local_id=*/8, /*is_inline=*/false),
                                   m_shards[i].size(),
                                   std::make_shared<StringViewEncoder>(m_shards[i]));
            }
        }

    private:
        std::vector<std::string> m_shards;
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    ShardHandler handler({"shard-0", "shard-1"});
    writer.add_sections({&handler});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(8, false);
    const auto shards = find_entries(*container, fake_device_id, tag.id());
    ASSERT_EQ(shards.size(), 2u);
    EXPECT_EQ(payload_string(*container, *shards[0]), "shard-0");
    EXPECT_EQ(payload_string(*container, *shards[1]), "shard-1");
}

TEST(HsmDeferredWriterTest, handler_forwards_to_a_named_member_function_instead_of_inlining_logic_in_the_lambda) {
    struct CompiledOptions {
        uint32_t version;
        float scale;
    };

    class OptionsHandler : public hsm::ISectionWriterHandler {
    public:
        explicit OptionsHandler(CompiledOptions options) : m_options(options) {}

        void handle_section(hsm::IWriter& writer) const override {
            // ForwardingEncoder is only a one-line forwarder; write_options() below is an ordinary
            // member function - as long or recursive as needed, with full access to this handler's state.
            class ForwardingEncoder final : public hsm::ISectionEncoder {
            public:
                explicit ForwardingEncoder(const OptionsHandler& owner) : m_owner(owner) {}
                void encode(const hsm::SectionSink& sink) const override {
                    m_owner.write_options(sink);
                }

            private:
                const OptionsHandler& m_owner;
            };

            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/10, /*is_inline=*/false),
                               sizeof(m_options),
                               std::make_shared<ForwardingEncoder>(*this));
        }

    private:
        void write_options(const hsm::SectionSink& sink) const {
            sink({reinterpret_cast<const std::byte*>(&m_options), sizeof(m_options)});
        }

        CompiledOptions m_options;
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    OptionsHandler handler(CompiledOptions{7, 0.5f});
    writer.add_sections({&handler});
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(10, false);
    const auto section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    CompiledOptions decoded{};
    std::memcpy(&decoded, container->bytes.data() + section->offset, sizeof(decoded));
    EXPECT_EQ(decoded.version, 7u);
    EXPECT_FLOAT_EQ(decoded.scale, 0.5f);
}

TEST(HsmDeferredWriterTest, section_encoder_composes_from_several_sub_encoders) {
    // A section built from independent sub-encoders composed via a tiny local helper - no new API
    // needed beyond ISectionEncoder itself: each part is its own encoder, composition just calls each
    // in turn (e.g. for a section assembled from several sub-objects).
    class ComposedEncoder final : public hsm::ISectionEncoder {
    public:
        explicit ComposedEncoder(std::vector<hsm::SectionEncoderPtr> parts) : m_parts(std::move(parts)) {}
        void encode(const hsm::SectionSink& sink) const override {
            for (const auto& part : m_parts) {
                part->encode(sink);
            }
        }

    private:
        std::vector<hsm::SectionEncoderPtr> m_parts;
    };

    const std::string header = "head-";
    const std::string body = "body-";
    const std::string footer = "foot";
    auto encoder = std::make_shared<ComposedEncoder>(std::vector<hsm::SectionEncoderPtr>{
        std::make_shared<StringViewEncoder>(header),
        std::make_shared<StringViewEncoder>(body),
        std::make_shared<StringViewEncoder>(footer),
    });

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id,
                       hsm::model_tag,
                       header.size() + body.size() + footer.size(),
                       std::move(encoder));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "head-body-foot");
}

// --- Wire-format compatibility: the writer's output must satisfy hsm_format.hpp's own contract ---

TEST(HsmDeferredWriterTest, written_header_satisfies_the_format_contract) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    hsm::Header header{};
    const auto bytes = stream.str();
    header = hsm::Header::view(reinterpret_cast<const uint8_t*>(bytes.data()));
    EXPECT_TRUE(hsm::is_recognized_header(header));
    EXPECT_TRUE(hsm::is_valid_header_fields(header));
    EXPECT_EQ(header.magic, hsm::BlobMagic::single);
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);
}

TEST(HsmDeferredWriterTest, every_written_manifest_entry_has_valid_section_bounds) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string id = "id";
    const std::string model = "model-bytes";
    writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id));
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_EQ(writer.finalize(), std::error_code{});

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    for (const auto& entry : container->entries) {
        EXPECT_TRUE(hsm::is_valid_section_bounds(entry, container->header));
    }
}

}  // namespace ov::test
