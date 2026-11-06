// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_writer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

// IWriter-interface-level tests only - no concrete IWriter implementation (e.g. DeferredWriter) is used
// here; a mock stands in wherever one is needed. Tests exercising a real implementation's behavior
// belong in that implementation's own test file (e.g. hsm_deferred_writer.cpp for DeferredWriter).
#include "openvino/runtime/hsm_weights.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;

using testing::_, testing::A, testing::Ref, testing::Return;
namespace {

// Mocks IWriter's pure virtuals so add_sections() - the interface's one concrete member - can be tested
// without any concrete IWriter implementation.
class MockIWriter : public hsm::IWriter {
public:
    MOCK_METHOD(bool,
                add_section,
                (hsm::DeviceId, hsm::SectionTagReserved, ov::util::MemoryView, hsm::SectionAlignment),
                (override));
    MOCK_METHOD(bool,
                add_section,
                (hsm::DeviceId, hsm::SectionTagReserved, size_t, hsm::SectionEncoderPtr, hsm::SectionAlignment),
                (override));
    MOCK_METHOD(bool,
                add_section,
                (hsm::DeviceId, hsm::SectionTagReserved, hsm::SectionEncoderPtr, hsm::SectionAlignment),
                (override));
    MOCK_METHOD(std::error_code, finalize, (), (override));
};

class MockSectionWriterHandler : public hsm::ISectionWriterHandler {
public:
    MOCK_METHOD(void, handle_section, (hsm::IWriter&), (const, override));
};

}  // namespace

// --- add_sections() - IWriter's one concrete (non-pure-virtual) member ---

TEST(HsmWriterTest, weights_encoder_can_be_registered_with_alignment) {
    const std::vector<std::byte> weights{
        std::byte{0x00},
        std::byte{0x01},
        std::byte{0x7F},
        std::byte{0x80},
        std::byte{0xFE},
        std::byte{0xFF},
    };

    MockIWriter writer;
    EXPECT_CALL(writer, add_section(hsm::any_device_id, _, weights.size(), A<hsm::SectionEncoderPtr>(), _))
        .WillOnce([&](hsm::DeviceId,
                      hsm::SectionTagReserved tag,
                      size_t size,
                      hsm::SectionEncoderPtr encoder,
                      hsm::SectionAlignment align) {
            EXPECT_EQ(tag.tag.id(), hsm::weights);
            EXPECT_TRUE(tag.is_pointer());
            EXPECT_EQ(align.offset_align, 64u);
            EXPECT_EQ(align.size_align, 16u);
            EXPECT_EQ(size, weights.size());
            encoder->encode([&](ov::util::MemoryView payload) {
                EXPECT_EQ(payload.data(), weights.data());
                EXPECT_EQ(payload.size(), weights.size());
            });
            return true;
        });

    ASSERT_TRUE(writer.add_section(hsm::any_device_id,
                                   hsm::weights_tag,
                                   weights.size(),
                                   hsm::encode_weights({weights.data(), weights.size()}),
                                   {/*offset_align=*/64, /*size_align=*/16}));
}

TEST(HsmWriterTest, add_sections_dispatches_to_each_non_null_handler_once_in_order) {
    MockIWriter writer;
    MockSectionWriterHandler first;
    MockSectionWriterHandler second;

    {
        ::testing::InSequence in_order;
        EXPECT_CALL(first, handle_section(Ref(writer))).Times(1);
        EXPECT_CALL(second, handle_section(Ref(writer))).Times(1);
    }

    writer.add_sections({&first, &second});
}

TEST(HsmWriterTest, add_sections_skips_a_null_handler) {
    MockIWriter writer;
    MockSectionWriterHandler handler;
    EXPECT_CALL(handler, handle_section(Ref(writer))).Times(1);

    EXPECT_NO_THROW(writer.add_sections({nullptr, &handler, nullptr}));
}

// --- IWriter::add_section()'s 3 overloads, each reachable through a dispatched handler ---

TEST(HsmWriterTest, add_sections_lets_a_dispatched_handler_use_any_add_section_overload) {
    MockIWriter writer;
    MockSectionWriterHandler handler;
    const std::string payload = "data";
    const ov::util::MemoryView view{reinterpret_cast<const std::byte*>(payload.data()), payload.size()};

    EXPECT_CALL(handler, handle_section(Ref(writer))).WillOnce([&](auto& w) {
        w.add_section(hsm::any_device_id, hsm::model_tag, view);                // view overload
        w.add_section(hsm::any_device_id, hsm::model_tag, size_t{4}, nullptr);  // sized-encoder overload
        w.add_section(hsm::any_device_id, hsm::model_tag, nullptr);             // unsized-encoder overload
    });
    EXPECT_CALL(writer, add_section(_, _, A<ov::util::MemoryView>(), _)).WillOnce(Return(true));
    EXPECT_CALL(writer, add_section(_, _, size_t{4}, _, _)).WillOnce(Return(true));
    EXPECT_CALL(writer, add_section(_, _, A<hsm::SectionEncoderPtr>(), _)).WillOnce(Return(true));

    writer.add_sections({&handler});
}

// --- WriteErrc / make_error_code ---

TEST(HsmWriterTest, write_errc_message_is_human_readable) {
    const auto ec = hsm::make_error_code(hsm::WriteErrc::write_failed);
    EXPECT_FALSE(ec.message().empty());
    EXPECT_EQ(ec, hsm::WriteErrc::write_failed);
}

TEST(HsmWriterTest, write_errc_category_reports_unknown_for_an_unrecognized_value) {
    const auto ec = hsm::make_error_code(static_cast<hsm::WriteErrc>(0));
    EXPECT_EQ(ec.message(), "unknown error");
    EXPECT_STREQ(ec.category().name(), "ov::runtime::hsm::Write");
}

}  // namespace ov::test
