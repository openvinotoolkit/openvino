// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_deferred_writer.hpp"

#include <gtest/gtest.h>

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

// Raw container parsing, no Reader dependency - same pattern as hsm_writer.cpp, kept file-local on purpose.
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
    const auto* a_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    const auto* b_entry = find_entry(*container, fake_device_id, b_tag.id());
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

}  // namespace ov::test
