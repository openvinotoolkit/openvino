// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_multi_blob_writer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ios>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hsm_test_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/memory.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

using hsm::BlobKind;
using hsm::MultiBlobOpenMode;
using hsm::MultiBlobWriter;
using hsm::WriteErrc;

// The empty shared-context blob written in create mode, spelled out to pin its wire layout.
constexpr std::array<uint8_t, 32> empty_shared_context{
    'O',  'V',  'W', 'S', 'H',           // magic
    0x01, 0x00,                          // version_major = 1
    0x00,                                // version_minor = 0
    0x20, 0,    0,   0,   0,   0, 0, 0,  // container_size = 32
    0x20, 0,    0,   0,   0,   0, 0, 0,  // manifest_offset = 32
    0,    0,    0,   0,   0,   0, 0, 0,  // manifest_size = 0
};

constexpr std::byte filler{0xAB};

template <class T>
std::vector<std::byte> bytes_of(const T& value) {
    const auto view = ov::util::as_memory_view(value);
    return {view.data(), view.data() + view.size()};
}

std::vector<std::byte> concat(std::initializer_list<std::vector<std::byte>> parts) {
    std::vector<std::byte> out;
    for (const auto& part : parts) {
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

ov::util::MemoryView view_of(const std::vector<std::byte>& bytes) {
    return {bytes.data(), bytes.size()};
}

hsm::Header make_header(const hsm::BlobMagic& magic, size_t container_size, size_t manifest_offset) {
    return {magic,
            hsm::FormatVersion::major,
            hsm::FormatVersion::minor,
            container_size,
            manifest_offset,
            container_size - manifest_offset};
}

// A valid container: header, then @p payload, then an empty manifest.
std::vector<std::byte> make_blob(const hsm::BlobMagic& magic, std::string_view payload) {
    const auto size = sizeof(hsm::Header) + payload.size();
    return concat({bytes_of(make_header(magic, size, size)), bytes_of(payload)});
}

std::vector<std::byte> make_model(std::string_view payload = "model") {
    return make_blob(hsm::BlobMagic::single, payload);
}

std::vector<std::byte> make_shared_context(std::string_view payload = "ctx") {
    return make_blob(hsm::BlobMagic::multi, payload);
}

const auto wsh = bytes_of(empty_shared_context);

std::vector<std::byte> stream_bytes(const std::stringstream& stream) {
    return bytes_of(stream.str());
}

std::vector<std::byte> filled_buffer(size_t size) {
    return std::vector<std::byte>(size, filler);
}

MultiBlobWriter open_writer(std::ostream& stream, MultiBlobOpenMode mode = MultiBlobOpenMode::create) {
    auto writer = MultiBlobWriter::open(stream, mode);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

MultiBlobWriter open_writer(std::span<std::byte> dst,
                            MultiBlobOpenMode mode = MultiBlobOpenMode::create,
                            size_t origin = 0) {
    auto writer = MultiBlobWriter::open(dst, mode, origin);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

// Writes fixed bytes in one call and records what the sink reported.
class BytesEncoder final : public hsm::IBlobEncoder {
public:
    explicit BytesEncoder(std::vector<std::byte> bytes) : m_bytes(std::move(bytes)) {}

    void encode(hsm::IBlobSink& sink) const override {
        seen_origin = sink.origin();
        sink.write(view_of(m_bytes));
        seen_good = sink.good();
    }

    mutable std::optional<size_t> seen_origin;
    mutable bool seen_good = false;

private:
    std::vector<std::byte> m_bytes;
};

// Writes a zero header first and fills it in last, like DeferredWriter does for an unsized section.
class HeaderLastEncoder final : public hsm::IBlobEncoder {
public:
    explicit HeaderLastEncoder(std::vector<std::byte> bytes) : m_bytes(std::move(bytes)) {}

    void encode(hsm::IBlobSink& sink) const override {
        const std::vector<std::byte> zeros(sizeof(hsm::Header));
        sink.write(view_of(zeros));
        sink.write({m_bytes.data() + sizeof(hsm::Header), m_bytes.size() - sizeof(hsm::Header)});
        sink.patch(0, {m_bytes.data(), sizeof(hsm::Header)});
    }

private:
    std::vector<std::byte> m_bytes;
};

// Writes some bytes, then throws.
class ThrowingBlobEncoder final : public hsm::IBlobEncoder {
public:
    static constexpr size_t written = 10;

    void encode(hsm::IBlobSink& sink) const override {
        const std::vector<std::byte> bytes(written, std::byte{0xEE});
        sink.write(view_of(bytes));
        OPENVINO_THROW("blob encoder blew up");
    }
};

// Patches bytes it hasn't written yet.
class OutOfBoundsPatchEncoder final : public hsm::IBlobEncoder {
public:
    static constexpr size_t written = 4;

    void encode(hsm::IBlobSink& sink) const override {
        const std::vector<std::byte> bytes(written, std::byte{0xEE});
        sink.write(view_of(bytes));
        sink.patch(2, view_of(bytes));
    }
};

}  // namespace

static_assert(std::is_move_constructible_v<MultiBlobWriter>);
static_assert(std::is_move_assignable_v<MultiBlobWriter>);
static_assert(!std::is_copy_constructible_v<MultiBlobWriter>);
static_assert(!std::is_copy_assignable_v<MultiBlobWriter>);

// --- open() ---

TEST(HsmMultiBlobWriterTest, open_stream_rejects_a_failed_stream) {
    std::stringstream stream;
    stream.setstate(std::ios::badbit);
    EXPECT_FALSE(MultiBlobWriter::open(stream).has_value());
}

TEST(HsmMultiBlobWriterTest, open_stream_rejects_a_non_seekable_stream) {
    NonSeekableStreamBuf buf;
    std::ostream stream(&buf);
    EXPECT_FALSE(MultiBlobWriter::open(stream).has_value());
}

TEST(HsmMultiBlobWriterTest, open_buffer_rejects_a_buffer_without_data) {
    EXPECT_FALSE(MultiBlobWriter::open(std::span<std::byte>{}).has_value());
}

TEST(HsmMultiBlobWriterTest, open_buffer_rejects_an_origin_that_overflows) {
    auto buffer = filled_buffer(8);
    const auto origin = std::numeric_limits<size_t>::max() - buffer.size() + 1;
    EXPECT_FALSE(MultiBlobWriter::open(buffer, MultiBlobOpenMode::create, origin).has_value());
}

TEST(HsmMultiBlobWriterTest, open_buffer_accepts_a_full_buffer_but_cannot_write_to_it) {
    auto buffer = filled_buffer(64);
    auto writer = open_writer(std::span{buffer}.subspan(buffer.size()));
    const auto model = make_model();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);
    EXPECT_EQ(buffer, filled_buffer(64));
}

// --- append() of a ready blob ---

TEST(HsmMultiBlobWriterTest, create_mode_writes_an_empty_shared_context_before_the_first_model) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({wsh, model}));
}

TEST(HsmMultiBlobWriterTest, create_mode_adds_no_shared_context_when_one_comes_first) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto context = make_shared_context();
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::shared_context, view_of(context)), std::error_code{});
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({context, model}));
}

TEST(HsmMultiBlobWriterTest, create_mode_writes_one_shared_context_for_two_models) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto first = make_model("first");
    const auto second = make_model("second");
    ASSERT_EQ(writer.append(BlobKind::model, view_of(first)), std::error_code{});
    ASSERT_EQ(writer.append(BlobKind::model, view_of(second)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({wsh, first, second}));
}

TEST(HsmMultiBlobWriterTest, append_mode_writes_a_model_without_a_shared_context) {
    std::stringstream stream;
    auto writer = open_writer(stream, MultiBlobOpenMode::append);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), model);
}

TEST(HsmMultiBlobWriterTest, buffer_writes_only_into_the_given_span) {
    constexpr size_t used = 5;
    auto buffer = filled_buffer(128);
    auto writer = open_writer(std::span{buffer}.subspan(used));
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});

    auto expected = concat({filled_buffer(used), wsh, model});
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);
}

TEST(HsmMultiBlobWriterTest, stream_keeps_bytes_before_its_start_position) {
    std::stringstream stream;
    stream << "PREFIX";
    auto writer = open_writer(stream);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({bytes_of(std::string_view{"PREFIX"}), wsh, model}));
}

// --- append() of a ready blob: invalid blobs ---

TEST(HsmMultiBlobWriterTest, rejects_a_blob_whose_magic_does_not_match_its_kind) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    const auto context = make_shared_context();
    EXPECT_EQ(writer.append(BlobKind::shared_context, view_of(model)), WriteErrc::invalid_blob);
    EXPECT_EQ(writer.append(BlobKind::model, view_of(context)), WriteErrc::invalid_blob);
    EXPECT_TRUE(stream.str().empty());
}

TEST(HsmMultiBlobWriterTest, rejects_a_blob_whose_size_differs_from_its_container_size) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    auto too_long = model;
    too_long.push_back(std::byte{0});
    const std::span<const std::byte> short_views[] = {std::span{model}.first(model.size() - 1),
                                                      std::span{model}.first(10)};

    EXPECT_EQ(writer.append(BlobKind::model, view_of(too_long)), WriteErrc::invalid_blob);
    for (const auto& view : short_views) {
        EXPECT_EQ(writer.append(BlobKind::model, ov::util::MemoryView{view.data(), view.size()}),
                  WriteErrc::invalid_blob);
    }
    EXPECT_TRUE(stream.str().empty());
}

TEST(HsmMultiBlobWriterTest, rejects_a_blob_with_invalid_header_fields) {
    std::stringstream stream;
    auto writer = open_writer(stream);

    auto wrong_version = make_header(hsm::BlobMagic::single, sizeof(hsm::Header), sizeof(hsm::Header));
    ++wrong_version.version_major;
    auto manifest_past_end = make_header(hsm::BlobMagic::single, sizeof(hsm::Header), sizeof(hsm::Header));
    manifest_past_end.manifest_offset = sizeof(hsm::Header) + 1;

    EXPECT_EQ(writer.append(BlobKind::model, view_of(bytes_of(wrong_version))), WriteErrc::invalid_blob);
    EXPECT_EQ(writer.append(BlobKind::model, view_of(bytes_of(manifest_past_end))), WriteErrc::invalid_blob);
    EXPECT_TRUE(stream.str().empty());
}

TEST(HsmMultiBlobWriterTest, rejects_a_blob_with_a_section_outside_the_container) {
    std::stringstream stream;
    auto writer = open_writer(stream);

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.offset = sizeof(hsm::Header);
    entry.size = 100;
    constexpr auto size = sizeof(hsm::Header) + sizeof(hsm::ManifestEntry);
    const auto blob =
        concat({bytes_of(make_header(hsm::BlobMagic::single, size, sizeof(hsm::Header))), bytes_of(entry)});

    EXPECT_EQ(writer.append(BlobKind::model, view_of(blob)), WriteErrc::invalid_blob);
    EXPECT_TRUE(stream.str().empty());
}

TEST(HsmMultiBlobWriterTest, writer_stays_usable_after_an_invalid_blob) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto context = make_shared_context();
    const auto model = make_model();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(context)), WriteErrc::invalid_blob);
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({wsh, model}));
}

// --- append() through an IBlobEncoder ---

TEST(HsmMultiBlobWriterTest, writes_an_encoder_blob) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, BytesEncoder{model}), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({wsh, model}));
}

TEST(HsmMultiBlobWriterTest, encoder_may_write_the_header_last_through_patch) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, HeaderLastEncoder{model}), std::error_code{});
    EXPECT_EQ(stream_bytes(stream), concat({wsh, model}));
}

TEST(HsmMultiBlobWriterTest, encoder_sees_the_blob_origin_in_a_stream) {
    std::stringstream stream;
    stream << "PREFIX";
    auto writer = open_writer(stream);
    const BytesEncoder first{make_model("first")};
    const BytesEncoder second{make_model("second")};
    ASSERT_EQ(writer.append(BlobKind::model, first), std::error_code{});
    ASSERT_EQ(writer.append(BlobKind::model, second), std::error_code{});

    const size_t first_origin = 6 + wsh.size();
    EXPECT_EQ(first.seen_origin, first_origin);
    EXPECT_EQ(second.seen_origin, first_origin + make_model("first").size());
}

TEST(HsmMultiBlobWriterTest, encoder_sees_the_given_origin_in_a_buffer) {
    constexpr size_t origin = 100;
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer, MultiBlobOpenMode::append, origin);
    const auto context = make_shared_context();
    const BytesEncoder first{context};
    const BytesEncoder second{context};
    ASSERT_EQ(writer.append(BlobKind::shared_context, first), std::error_code{});
    ASSERT_EQ(writer.append(BlobKind::shared_context, second), std::error_code{});

    EXPECT_EQ(first.seen_origin, origin);
    EXPECT_EQ(second.seen_origin, origin + context.size());
    EXPECT_TRUE(std::equal(context.begin(), context.end(), buffer.begin()));  // origin doesn't move writing
}

TEST(HsmMultiBlobWriterTest, invalid_encoder_blob_is_zero_filled_in_a_buffer) {
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer, MultiBlobOpenMode::append);
    const auto context = make_shared_context();
    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{context}), WriteErrc::invalid_blob);

    auto expected = std::vector<std::byte>(context.size());
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);

    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    std::ranges::copy(model, expected.begin());
    EXPECT_EQ(buffer, expected);
}

TEST(HsmMultiBlobWriterTest, next_blob_starts_where_an_invalid_encoder_blob_started_in_a_stream) {
    std::stringstream stream;
    auto writer = open_writer(stream, MultiBlobOpenMode::append);
    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{make_shared_context("longer-context")}),
              WriteErrc::invalid_blob);

    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    const auto bytes = stream_bytes(stream);
    ASSERT_GE(bytes.size(), model.size());
    EXPECT_TRUE(std::equal(model.begin(), model.end(), bytes.begin()));
}

TEST(HsmMultiBlobWriterTest, encoder_exception_is_rethrown_after_zero_filling_a_buffer) {
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer, MultiBlobOpenMode::append, /*origin=*/100);
    EXPECT_THROW(std::ignore = writer.append(BlobKind::model, ThrowingBlobEncoder{}), ov::Exception);

    auto expected = std::vector<std::byte>(ThrowingBlobEncoder::written);
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);

    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    EXPECT_TRUE(std::equal(model.begin(), model.end(), buffer.begin()));
}

TEST(HsmMultiBlobWriterTest, encoder_exception_is_rethrown_and_the_stream_stays_usable) {
    std::stringstream stream;
    auto writer = open_writer(stream, MultiBlobOpenMode::append);
    EXPECT_THROW(std::ignore = writer.append(BlobKind::model, ThrowingBlobEncoder{}), ov::Exception);

    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    const auto bytes = stream_bytes(stream);
    ASSERT_GE(bytes.size(), model.size());
    EXPECT_TRUE(std::equal(model.begin(), model.end(), bytes.begin()));
}

TEST(HsmMultiBlobWriterTest, rejects_an_encoder_blob_whose_size_differs_from_its_container_size) {
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer, MultiBlobOpenMode::append);
    auto too_long = make_model();
    too_long.push_back(std::byte{0});
    const auto model = make_model();
    const std::vector<std::byte> too_short(model.begin(), model.end() - 1);

    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{too_long}), WriteErrc::invalid_blob);
    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{too_short}), WriteErrc::invalid_blob);
    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{{}}), WriteErrc::invalid_blob);

    auto expected = std::vector<std::byte>(too_long.size());
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);
}

TEST(HsmMultiBlobWriterTest, encoder_patch_past_written_bytes_throws_and_is_rolled_back) {
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer, MultiBlobOpenMode::append);
    EXPECT_THROW(std::ignore = writer.append(BlobKind::model, OutOfBoundsPatchEncoder{}), ov::AssertFailure);

    auto expected = std::vector<std::byte>(OutOfBoundsPatchEncoder::written);
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);
}

TEST(HsmMultiBlobWriterTest, shared_context_stays_when_the_first_model_encoder_fails) {
    auto buffer = filled_buffer(128);
    auto writer = open_writer(buffer);
    const auto context = make_shared_context();
    EXPECT_EQ(writer.append(BlobKind::model, BytesEncoder{context}), WriteErrc::invalid_blob);

    auto expected = concat({wsh, std::vector<std::byte>(context.size())});
    expected.resize(buffer.size(), filler);
    EXPECT_EQ(buffer, expected);

    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});
    std::ranges::copy(model, expected.begin() + wsh.size());
    EXPECT_EQ(buffer, expected);
}

// --- append(): write failures ---

TEST(HsmMultiBlobWriterTest, buffer_too_small_for_a_blob_fails_for_good) {
    const auto model = make_model();
    auto buffer = filled_buffer(wsh.size() + model.size() - 1);
    auto writer = open_writer(buffer);
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);

    const auto tiny = make_shared_context("");
    EXPECT_EQ(writer.append(BlobKind::shared_context, view_of(tiny)), WriteErrc::write_failed);
}

TEST(HsmMultiBlobWriterTest, buffer_too_small_for_an_encoder_blob_fails_for_good) {
    auto buffer = filled_buffer(wsh.size() + 10);
    auto writer = open_writer(buffer);
    const BytesEncoder encoder{make_model()};
    EXPECT_EQ(writer.append(BlobKind::model, encoder), WriteErrc::write_failed);
    EXPECT_FALSE(encoder.seen_good);
    EXPECT_EQ(buffer, concat({wsh, filled_buffer(10)}));

    const auto tiny = make_shared_context("");
    EXPECT_EQ(writer.append(BlobKind::shared_context, view_of(tiny)), WriteErrc::write_failed);
}

TEST(HsmMultiBlobWriterTest, stream_write_failing_partway_fails_for_good) {
    FlakyOnceStreamBuf buf;
    buf.fail_on_call = 2;  // the shared-context write succeeds; the model write fails halfway
    std::ostream stream(&buf);
    auto writer = open_writer(stream);
    const auto model = make_model();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);

    stream.clear();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);
}

TEST(HsmMultiBlobWriterTest, stream_going_bad_between_appends_fails_for_good) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const auto model = make_model();
    ASSERT_EQ(writer.append(BlobKind::model, view_of(model)), std::error_code{});

    stream.setstate(std::ios::badbit);
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);

    stream.clear();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);
}

TEST(HsmMultiBlobWriterTest, stream_failing_during_rollback_still_rethrows_the_encoder_exception) {
    FlakyOnceStreamBuf buf;
    buf.fail_on_call = 2;  // the encoder's write succeeds; the zero-fill write fails halfway
    std::ostream stream(&buf);
    stream.exceptions(std::ios::badbit);
    auto writer = open_writer(stream, MultiBlobOpenMode::append);
    EXPECT_THROW(std::ignore = writer.append(BlobKind::model, ThrowingBlobEncoder{}), ov::Exception);

    stream.clear();
    const auto model = make_model();
    EXPECT_EQ(writer.append(BlobKind::model, view_of(model)), WriteErrc::write_failed);
}

TEST(HsmMultiBlobWriterTest, invalid_blob_error_has_a_readable_message) {
    const auto message = make_error_code(WriteErrc::invalid_blob).message();
    EXPECT_FALSE(message.empty());
    EXPECT_NE(message, "unknown error");
}

}  // namespace ov::test
