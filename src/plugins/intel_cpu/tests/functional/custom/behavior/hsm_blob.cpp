// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "internal_properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "utils/graph_serializer/hsm_common.hpp"

namespace ov::test {

namespace hsm = ov::intel_cpu::hsm;
using behavior::HsmEntry;
using behavior::hsm_repack;
using behavior::is_hsm;

static bool is_section(const HsmEntry& entry, const hsm::SectionId& id) {
    return entry.device == id.device && entry.tag.tag.id() == id.tag.id();
}

// TODO: when HSM becomes the default, compile() no longer needs export_hsm_format(true); the legacy interop test then
// needs export_hsm_format(false) and goes away with the legacy format.
class CpuHsmBlobTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ov::test::utils::make_conv_pool_relu({1, 3, 16, 16}, ov::element::f32);
        input = ov::test::utils::create_and_fill_tensor(ov::element::f32, model->input().get_shape());
        reference = infer(core.compile_model(model, ov::test::utils::DEVICE_CPU), input);
    }

    ov::CompiledModel compile(const std::shared_ptr<ov::Model>& m, bool hsm_format) {
        return core.compile_model(m, ov::test::utils::DEVICE_CPU, ov::intel_cpu::export_hsm_format(hsm_format));
    }

    std::string export_blob(bool hsm_format) {
        std::stringstream stream;
        compile(model, hsm_format).export_model(stream);
        return stream.str();
    }

    static ov::Tensor infer(ov::CompiledModel compiled, const ov::Tensor& in) {
        auto request = compiled.create_infer_request();
        request.set_input_tensor(in);
        request.infer();
        const auto out = request.get_output_tensor();
        ov::Tensor result(out.get_element_type(), out.get_shape());
        out.copy_to(result);
        return result;
    }

    void expect_import_matches_reference(const std::string& blob) {
        std::stringstream stream(blob);
        auto from_stream = core.import_model(stream, ov::test::utils::DEVICE_CPU);
        ov::test::utils::compare(reference, infer(from_stream, input));

        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        auto from_tensor = core.import_model(tensor, ov::test::utils::DEVICE_CPU);
        ov::test::utils::compare(reference, infer(from_tensor, input));
    }

    void expect_import_rejected(const std::string& blob) {
        std::stringstream stream(blob);
        EXPECT_THROW((void)core.import_model(stream, ov::test::utils::DEVICE_CPU), ov::Exception);

        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        EXPECT_THROW((void)core.import_model(tensor, ov::test::utils::DEVICE_CPU), ov::Exception);
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::Tensor input;
    ov::Tensor reference;
};

TEST_F(CpuHsmBlobTest, exported_blob_is_hsm_with_cpu_sections) {
    const auto blob = export_blob(true);
    ASSERT_TRUE(is_hsm(blob));

    const auto reader = ov::runtime::hsm::Reader::open(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(reader->count(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id()), 1u);
    EXPECT_EQ(reader->count(hsm::serialized_model.device, hsm::serialized_model.tag.id()), 1u);

    const auto requirements = reader->section(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id());
    ASSERT_TRUE(requirements.has_value());
    const auto bytes = requirements->to_bytes();
    ASSERT_TRUE(bytes.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size()),
              compile(model, true).get_property(ov::runtime_requirements));
}

TEST_F(CpuHsmBlobTest, hsm_blob_is_imported_from_stream_and_tensor) {
    expect_import_matches_reference(export_blob(true));
}

TEST_F(CpuHsmBlobTest, legacy_blob_is_imported_by_the_hsm_capable_reader) {
    const auto blob = export_blob(false);
    EXPECT_FALSE(is_hsm(blob));
    expect_import_matches_reference(blob);
}

TEST_F(CpuHsmBlobTest, model_without_constants_round_trips) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 8});
    auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
    auto no_constants = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu),
                                                    ov::ParameterVector{parameter},
                                                    "no_constants");
    std::stringstream stream;
    compile(no_constants, true).export_model(stream);

    auto imported = core.import_model(stream, ov::test::utils::DEVICE_CPU);

    const auto in = ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 8});
    ov::test::utils::compare(infer(compile(no_constants, true), in), infer(imported, in));
}

TEST_F(CpuHsmBlobTest, unknown_sections_are_skipped) {
    const auto repacked = hsm_repack(export_blob(true), [](std::vector<HsmEntry>& entries) {
        const std::string payload = "future section";
        entries.push_back({hsm::device_id,
                           ov::runtime::hsm::SectionTag::make_device_tag(0x123, false),
                           {reinterpret_cast<const std::byte*>(payload.data()),
                            reinterpret_cast<const std::byte*>(payload.data()) + payload.size()}});
    });
    expect_import_matches_reference(repacked);
}

TEST_F(CpuHsmBlobTest, import_rejects_blob_without_a_mandatory_section) {
    for (const auto& mandatory : {hsm::runtime_requirements, hsm::serialized_model}) {
        const auto repacked = hsm_repack(export_blob(true), [&](std::vector<HsmEntry>& entries) {
            entries.erase(std::remove_if(entries.begin(),
                                         entries.end(),
                                         [&](const HsmEntry& entry) {
                                             return is_section(entry, mandatory);
                                         }),
                          entries.end());
        });
        SCOPED_TRACE("missing section tag id " + std::to_string(mandatory.tag.id()));
        expect_import_rejected(repacked);
    }
}

TEST_F(CpuHsmBlobTest, import_rejects_duplicated_section) {
    const auto repacked = hsm_repack(export_blob(true), [](std::vector<HsmEntry>& entries) {
        const auto model_entry = std::find_if(entries.begin(), entries.end(), [](const HsmEntry& entry) {
            return is_section(entry, hsm::serialized_model);
        });
        ASSERT_NE(model_entry, entries.end());
        entries.push_back(*model_entry);
    });
    expect_import_rejected(repacked);
}

TEST_F(CpuHsmBlobTest, import_rejects_mismatched_runtime_requirements) {
    const auto repacked = hsm_repack(export_blob(true), [](std::vector<HsmEntry>& entries) {
        for (auto& entry : entries) {
            if (is_section(entry, hsm::runtime_requirements)) {
                entry.payload.front() ^= std::byte{0x01};
            }
        }
    });
    expect_import_rejected(repacked);
}

}  // namespace ov::test
