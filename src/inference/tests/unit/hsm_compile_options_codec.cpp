// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_compile_options_codec.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
using namespace std::string_literals;
namespace {

using SectionDecodeFn = std::optional<ov::AnyMap> (*)(const hsm::Section&) noexcept;
using EncodedBytes = decltype(hsm::encode_compile_options(std::declval<const ov::AnyMap&>()));

static_assert(std::is_convertible_v<decltype(static_cast<SectionDecodeFn>(&hsm::decode_compile_options)),
                                    hsm::SectionDecoder<ov::AnyMap>>,
              "decode_compile_options(const Section&) must be usable as a SectionDecoder<ov::AnyMap>.");
static_assert(std::is_base_of_v<hsm::ISectionEncoder, hsm::CompileOptionsEncoder>,
              "CompileOptionsEncoder must be an ISectionEncoder.");
static_assert(std::is_convertible_v<std::shared_ptr<hsm::CompileOptionsEncoder>, hsm::SectionEncoderPtr>,
              "CompileOptionsEncoder must be passable as a SectionEncoderPtr.");
static_assert(std::is_constructible_v<hsm::CompileOptionsEncoder, const ov::AnyMap&>,
              "CompileOptionsEncoder must borrow an existing map.");
static_assert(!std::is_constructible_v<hsm::CompileOptionsEncoder, ov::AnyMap&&>,
              "CompileOptionsEncoder must not borrow a temporary map.");
static_assert(!std::is_constructible_v<hsm::CompileOptionsEncoder, const ov::AnyMap&&>,
              "CompileOptionsEncoder must not borrow a temporary const map.");
static_assert(std::is_same_v<decltype(std::declval<hsm::IWriter&>().add_section(
                                 hsm::DeviceId{},
                                 hsm::compile_options_tag,
                                 ov::util::MemoryView{std::declval<const EncodedBytes&>().data(),
                                                      std::declval<const EncodedBytes&>().size()})),
                             bool>,
              "encode_compile_options() result must fit IWriter::add_section(MemoryView).");
static_assert(std::is_same_v<decltype(std::declval<hsm::IWriter&>().add_section(
                                 hsm::DeviceId{},
                                 hsm::compile_options_tag,
                                 std::declval<std::shared_ptr<hsm::CompileOptionsEncoder>>())),
                             bool>,
              "CompileOptionsEncoder must fit the unsized IWriter::add_section(SectionEncoderPtr).");

constexpr hsm::DeviceId fake_device_id = hsm::private_device_id_range_start + 42;

/// @p input is what the encoder gets; @p decoded what the decoder must return (if unset, same as @p input).
struct ReferenceCase {
    std::string name;
    ov::AnyMap input;
    std::optional<ov::AnyMap> decoded;
    std::string bytes;

    const ov::AnyMap& expected_decoded() const {
        return decoded ? *decoded : input;
    }
};

void PrintTo(const ReferenceCase& reference, std::ostream* os) {
    *os << reference.name;
}

// Reference payloads are spelled out byte by byte (little-endian, like HSM) - never produced by the codec.
const ReferenceCase empty_map_case{"empty_map", {}, std::nullopt, "\x00\x00\x00\x00"s};

const ReferenceCase single_string_case{"single_string",
                                       {{"k", "v"}},
                                       std::nullopt,
                                       "\x01\x00\x00\x00"
                                       "\x01\x00\x00\x00"
                                       "k"
                                       "\x00"
                                       "\x01\x00\x00\x00"
                                       "v"s};

const ReferenceCase empty_string_case{"empty_string",
                                      {{"E", ""}},
                                      std::nullopt,
                                      "\x01\x00\x00\x00"
                                      "\x01\x00\x00\x00"
                                      "E"
                                      "\x00"
                                      "\x00\x00\x00\x00"s};

const ReferenceCase empty_any_case{"empty_any",
                                   {{"U", ov::Any{}}},
                                   std::nullopt,
                                   "\x01\x00\x00\x00"
                                   "\x01\x00\x00\x00"
                                   "U"
                                   "\x02"
                                   "\x00\x00\x00\x00"s};

const ReferenceCase typed_case{"typed",
                               {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                ov::hint::inference_precision(ov::element::f16),
                                ov::enable_profiling(true),
                                ov::hint::num_requests(4)},
                               ov::AnyMap{{"INFERENCE_PRECISION_HINT", "f16"},
                                          {"PERFORMANCE_HINT", "THROUGHPUT"},
                                          {"PERFORMANCE_HINT_NUM_REQUESTS", "4"},
                                          {"PERF_COUNT", "YES"}},
                               "\x04\x00\x00\x00"
                               "\x18\x00\x00\x00"
                               "INFERENCE_PRECISION_HINT"
                               "\x00"
                               "\x03\x00\x00\x00"
                               "f16"
                               "\x10\x00\x00\x00"
                               "PERFORMANCE_HINT"
                               "\x00"
                               "\x0a\x00\x00\x00"
                               "THROUGHPUT"
                               "\x1d\x00\x00\x00"
                               "PERFORMANCE_HINT_NUM_REQUESTS"
                               "\x00"
                               "\x01\x00\x00\x00"
                               "4"
                               "\x0a\x00\x00\x00"
                               "PERF_COUNT"
                               "\x00"
                               "\x03\x00\x00\x00"
                               "YES"s};

const ReferenceCase special_chars_case{"special_chars",
                                       {{"a\nb=c", "{x:y,z}"}, {"\xC5\xBC", "\xE2\x9C\x93"}},
                                       std::nullopt,
                                       "\x02\x00\x00\x00"
                                       "\x05\x00\x00\x00"
                                       "a\nb=c"
                                       "\x00"
                                       "\x07\x00\x00\x00"
                                       "{x:y,z}"
                                       "\x02\x00\x00\x00"
                                       "\xC5\xBC"
                                       "\x00"
                                       "\x03\x00\x00\x00"
                                       "\xE2\x9C\x93"s};

const ReferenceCase nested_empty_map_case{"nested_empty_map",
                                          {{"M", ov::AnyMap{}}},
                                          std::nullopt,
                                          "\x01\x00\x00\x00"
                                          "\x01\x00\x00\x00"
                                          "M"
                                          "\x01"
                                          "\x04\x00\x00\x00"
                                          "\x00\x00\x00\x00"s};

const ReferenceCase nested_two_levels_case{"nested_two_levels",
                                           {{"D", ov::AnyMap{{"C", ov::AnyMap{{"k", "v"}}}}}, {"T", "1"}},
                                           std::nullopt,
                                           "\x02\x00\x00\x00"
                                           "\x01\x00\x00\x00"
                                           "D"
                                           "\x01"
                                           "\x1d\x00\x00\x00"
                                           "\x01\x00\x00\x00"
                                           "\x01\x00\x00\x00"
                                           "C"
                                           "\x01"
                                           "\x0f\x00\x00\x00"
                                           "\x01\x00\x00\x00"
                                           "\x01\x00\x00\x00"
                                           "k"
                                           "\x00"
                                           "\x01\x00\x00\x00"
                                           "v"
                                           "\x01\x00\x00\x00"
                                           "T"
                                           "\x00"
                                           "\x01\x00\x00\x00"
                                           "1"s};

const ReferenceCase sorted_keys_case{"sorted_keys",
                                     {{"b", "2"}, {"a", "1"}},
                                     std::nullopt,
                                     "\x02\x00\x00\x00"
                                     "\x01\x00\x00\x00"
                                     "a"
                                     "\x00"
                                     "\x01\x00\x00\x00"
                                     "1"
                                     "\x01\x00\x00\x00"
                                     "b"
                                     "\x00"
                                     "\x01\x00\x00\x00"
                                     "2"s};

std::vector<ReferenceCase> reference_cases() {
    return {empty_map_case,
            single_string_case,
            empty_string_case,
            empty_any_case,
            typed_case,
            special_chars_case,
            nested_empty_map_case,
            nested_two_levels_case,
            sorted_keys_case};
}

// Builders only for payloads too large to spell out; the reference payloads above stay literal.
std::string u32(uint32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string entry(std::string_view key, uint8_t kind, std::string_view value) {
    return u32(static_cast<uint32_t>(key.size())) + std::string(key) + static_cast<char>(kind) +
           u32(static_cast<uint32_t>(value.size())) + std::string(value);
}

std::string map_of(uint32_t count, const std::string& entries = {}) {
    return u32(count) + entries;
}

std::string nested_payload(size_t depth) {
    auto bytes = map_of(0);
    for (size_t level = 1; level < depth; ++level) {
        bytes = map_of(1, entry("level", 1, bytes));
    }
    return bytes;
}

ov::AnyMap nested_options(size_t depth) {
    ov::AnyMap options{{"leaf", "value"}};
    for (size_t level = 1; level < depth; ++level) {
        options = ov::AnyMap{{"level", options}};
    }
    return options;
}

const std::string at_value_limit(hsm::max_compile_option_value_size, 'x');
const std::string over_value_limit(hsm::max_compile_option_value_size + 1, 'x');

std::string to_string(ov::util::MemoryView bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::string encode(const ov::AnyMap& options) {
    const auto bytes = hsm::encode_compile_options(options);
    return to_string({bytes.data(), bytes.size()});
}

ov::util::MemoryView view_of(std::string_view bytes) {
    return {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()};
}

std::optional<ov::AnyMap> decode(std::string_view bytes) {
    return hsm::decode_compile_options(view_of(bytes));
}

std::string encode_with_encoder(const ov::AnyMap& options) {
    const hsm::CompileOptionsEncoder encoder{options};
    std::string written;
    encoder.encode([&written](ov::util::MemoryView chunk) {
        written += to_string(chunk);
    });
    return written;
}

hsm::ManifestEntry make_entry(size_t size) {
    hsm::ManifestEntry entry{};
    entry.device = fake_device_id;
    entry.tag = hsm::compile_options_tag;
    entry.size = size;
    return entry;
}

// Any::operator== is false for two empty Anys, so compare structurally; decoded leaves must be strings.
::testing::AssertionResult same_options(const ov::AnyMap& actual,
                                        const ov::AnyMap& expected,
                                        const std::string& path = {}) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << path << "/: " << actual.size() << " entries, expected " << expected.size();
    }
    for (const auto& [key, value] : expected) {
        const auto where = path + "/" + key;
        const auto found = actual.find(key);
        if (found == actual.end()) {
            return ::testing::AssertionFailure() << where << ": missing";
        }
        const auto& got = found->second;
        if (value.empty()) {
            if (!got.empty()) {
                return ::testing::AssertionFailure() << where << ": expected an empty Any";
            }
        } else if (value.is<ov::AnyMap>()) {
            if (!got.is<ov::AnyMap>()) {
                return ::testing::AssertionFailure() << where << ": expected a nested map";
            }
            if (auto nested = same_options(got.as<ov::AnyMap>(), value.as<ov::AnyMap>(), where); !nested) {
                return nested;
            }
        } else if (!got.is<std::string>() || got.as<std::string>() != value.as<std::string>()) {
            return ::testing::AssertionFailure()
                   << where << ": got " << (got.is<std::string>() ? got.as<std::string>() : "<not a string>")
                   << ", expected " << value.as<std::string>();
        }
    }
    return ::testing::AssertionSuccess();
}

template <class Case>
std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    return info.param.name;
}

}  // namespace

// --- Encoding: options -> reference bytes ---

using HsmCompileOptionsEncodeReferenceTest = ::testing::TestWithParam<ReferenceCase>;

TEST_P(HsmCompileOptionsEncodeReferenceTest, encode_produces_reference_bytes) {
    EXPECT_EQ(encode(GetParam().input), GetParam().bytes);
}

TEST_P(HsmCompileOptionsEncodeReferenceTest, encoder_writes_reference_bytes_to_sink) {
    EXPECT_EQ(encode_with_encoder(GetParam().input), GetParam().bytes);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsEncodeReferenceTest,
                         ::testing::ValuesIn(reference_cases()),
                         case_name<ReferenceCase>);

namespace {
struct InvalidOptionsCase {
    std::string name;
    ov::AnyMap options;
};

void PrintTo(const InvalidOptionsCase& invalid, std::ostream* os) {
    *os << invalid.name;
}

std::vector<InvalidOptionsCase> invalid_options_cases() {
    ov::AnyMap oversized_payload;
    for (size_t i = 0; i <= hsm::max_compile_options_size / at_value_limit.size(); ++i) {
        oversized_payload.emplace("KEY_" + std::to_string(i), at_value_limit);
    }
    return {
        {"empty_key", {{"", "v"}}},
        {"empty_nested_key", {{"M", ov::AnyMap{{"", "v"}}}}},
        {"oversized_key", {{over_value_limit, "v"}}},
        {"oversized_value", {{"k", over_value_limit}}},
        {"too_deep", nested_options(hsm::max_compile_options_depth + 1)},
        {"oversized_payload", oversized_payload},
    };
}
}  // namespace

using HsmCompileOptionsEncodeInvalidTest = ::testing::TestWithParam<InvalidOptionsCase>;

TEST_P(HsmCompileOptionsEncodeInvalidTest, encode_throws) {
    EXPECT_THROW(hsm::encode_compile_options(GetParam().options), ov::Exception);
}

TEST_P(HsmCompileOptionsEncodeInvalidTest, encoder_throws) {
    EXPECT_THROW(encode_with_encoder(GetParam().options), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsEncodeInvalidTest,
                         ::testing::ValuesIn(invalid_options_cases()),
                         case_name<InvalidOptionsCase>);

TEST(HsmCompileOptionsEncodeTest, limits_are_inclusive) {
    EXPECT_NO_THROW(hsm::encode_compile_options({{at_value_limit, at_value_limit}}));
    EXPECT_NO_THROW(hsm::encode_compile_options(nested_options(hsm::max_compile_options_depth)));
}

TEST(HsmCompileOptionsEncodeTest, encoder_encodes_borrowed_map_only_in_encode) {
    ov::AnyMap options{{"", "invalid until fixed"}};
    const hsm::CompileOptionsEncoder encoder{options};

    EXPECT_THROW(encoder.encode([](ov::util::MemoryView) {}), ov::Exception);

    options = single_string_case.input;
    std::string written;
    encoder.encode([&written](ov::util::MemoryView chunk) {
        written += to_string(chunk);
    });
    EXPECT_EQ(written, single_string_case.bytes);
}

// --- Decoding: reference bytes -> options ---

using HsmCompileOptionsDecodeReferenceTest = ::testing::TestWithParam<ReferenceCase>;

TEST_P(HsmCompileOptionsDecodeReferenceTest, decode_produces_reference_options) {
    const auto decoded = decode(GetParam().bytes);

    ASSERT_TRUE(decoded);
    EXPECT_TRUE(same_options(*decoded, GetParam().expected_decoded()));
}

TEST_P(HsmCompileOptionsDecodeReferenceTest, every_truncated_prefix_is_rejected) {
    const std::string_view bytes = GetParam().bytes;

    for (size_t size = 0; size < bytes.size(); ++size) {
        EXPECT_FALSE(decode(bytes.substr(0, size))) << "prefix size: " << size;
    }
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsDecodeReferenceTest,
                         ::testing::ValuesIn(reference_cases()),
                         case_name<ReferenceCase>);

namespace {
struct MalformedCase {
    std::string name;
    std::string bytes;
};

void PrintTo(const MalformedCase& malformed, std::ostream* os) {
    *os << malformed.name;
}

std::vector<MalformedCase> malformed_cases() {
    std::string oversized_entries;
    uint32_t oversized_count = 0;
    for (; oversized_entries.size() <= hsm::max_compile_options_size; ++oversized_count) {
        oversized_entries += entry("KEY_" + std::to_string(oversized_count), 0, at_value_limit);
    }
    return {
        {"trailing_byte", single_string_case.bytes + '\0'},
        {"count_beyond_payload", "\xff\xff\xff\xff"s},
        {"key_len_beyond_payload",
         "\x01\x00\x00\x00"
         "\x05\x00\x00\x00"
         "k"s},
        {"key_len_huge",
         "\x01\x00\x00\x00"
         "\xff\xff\xff\xff"
         "k"s},
        {"value_len_beyond_payload",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "k"
         "\x00"
         "\x05\x00\x00\x00"
         "v"s},
        {"value_len_huge",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "k"
         "\x00"
         "\xff\xff\xff\xff"
         "v"s},
        {"empty_key",
         "\x01\x00\x00\x00"
         "\x00\x00\x00\x00"
         "\x00"
         "\x01\x00\x00\x00"
         "v"s},
        {"duplicate_key",
         "\x02\x00\x00\x00"
         "\x01\x00\x00\x00"
         "k"
         "\x00"
         "\x01\x00\x00\x00"
         "1"
         "\x01\x00\x00\x00"
         "k"
         "\x00"
         "\x01\x00\x00\x00"
         "2"s},
        {"empty_kind_with_value",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "k"
         "\x02"
         "\x01\x00\x00\x00"
         "x"s},
        {"nested_map_longer_than_value_len",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "M"
         "\x01"
         "\x05\x00\x00\x00"
         "\x00\x00\x00\x00"
         "\x00"s},
        {"nested_map_shorter_than_value_len",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "M"
         "\x01"
         "\x03\x00\x00\x00"
         "\x00\x00\x00"s},
        {"nested_map_count_beyond_value_len",
         "\x01\x00\x00\x00"
         "\x01\x00\x00\x00"
         "M"
         "\x01"
         "\x04\x00\x00\x00"
         "\x01\x00\x00\x00"s},
        {"too_deep", nested_payload(hsm::max_compile_options_depth + 1)},
        {"oversized_key", map_of(1, entry(over_value_limit, 0, "v"))},
        {"oversized_value", map_of(1, entry("k", 0, over_value_limit))},
        {"oversized_payload", map_of(oversized_count, oversized_entries)},
    };
}
}  // namespace

using HsmCompileOptionsDecodeMalformedTest = ::testing::TestWithParam<MalformedCase>;

TEST_P(HsmCompileOptionsDecodeMalformedTest, decode_rejects) {
    EXPECT_FALSE(decode(GetParam().bytes));
}

INSTANTIATE_TEST_SUITE_P(HsmCompileOptions,
                         HsmCompileOptionsDecodeMalformedTest,
                         ::testing::ValuesIn(malformed_cases()),
                         case_name<MalformedCase>);

TEST(HsmCompileOptionsDecodeTest, limits_are_inclusive) {
    EXPECT_TRUE(decode(map_of(1, entry(at_value_limit, 0, at_value_limit))));
    EXPECT_TRUE(decode(nested_payload(hsm::max_compile_options_depth)));
}

TEST(HsmCompileOptionsDecodeTest, decoded_strings_convert_to_property_types) {
    const auto decoded = decode(typed_case.bytes);

    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded->at(ov::hint::performance_mode.name()).as<ov::hint::PerformanceMode>(),
              ov::hint::PerformanceMode::THROUGHPUT);
    EXPECT_EQ(decoded->at(ov::hint::inference_precision.name()).as<ov::element::Type>(), ov::element::f16);
    EXPECT_TRUE(decoded->at(ov::enable_profiling.name()).as<bool>());
    EXPECT_EQ(decoded->at(ov::hint::num_requests.name()).as<uint32_t>(), 4u);
}

TEST(HsmCompileOptionsDecodeTest, unknown_kind_is_skipped) {
    const auto decoded = decode("\x02\x00\x00\x00"
                                "\x01\x00\x00\x00"
                                "F"
                                "\x7f"
                                "\x03\x00\x00\x00"
                                "abc"
                                "\x01\x00\x00\x00"
                                "k"
                                "\x00"
                                "\x01\x00\x00\x00"
                                "v"s);

    ASSERT_TRUE(decoded);
    EXPECT_TRUE(same_options(*decoded, single_string_case.input));
}

// --- Decoding from a Section ---

TEST(HsmCompileOptionsSectionTest, decodes_buffer_backed_section) {
    const auto& reference = nested_two_levels_case;

    const auto decoded =
        hsm::decode_compile_options(hsm::Section{make_entry(reference.bytes.size()), view_of(reference.bytes)});

    ASSERT_TRUE(decoded);
    EXPECT_TRUE(same_options(*decoded, reference.expected_decoded()));
}

TEST(HsmCompileOptionsSectionTest, decodes_stream_backed_section) {
    const auto& reference = nested_two_levels_case;
    std::stringstream stream{"prefix" + reference.bytes};

    const auto decoded = hsm::decode_compile_options(hsm::Section{make_entry(reference.bytes.size()), stream, 6});

    ASSERT_TRUE(decoded);
    EXPECT_TRUE(same_options(*decoded, reference.expected_decoded()));
}

TEST(HsmCompileOptionsSectionTest, oversized_section_is_rejected_before_read) {
    std::stringstream stream{empty_map_case.bytes};

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(hsm::max_compile_options_size + 1), stream, 0}));
}

TEST(HsmCompileOptionsSectionTest, unreadable_section_is_rejected) {
    const auto& reference = single_string_case;
    std::stringstream stream{reference.bytes.substr(0, reference.bytes.size() - 1)};

    EXPECT_FALSE(hsm::decode_compile_options(hsm::Section{make_entry(reference.bytes.size()), stream, 0}));
}

}  // namespace ov::test
