// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "internal_properties.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/memory.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "utils/graph_serializer/hsm_common.hpp"

namespace ov::test {

namespace hsm = intel_cpu::hsm;
using behavior::HsmEntry;
using behavior::hsm_repack;
using behavior::is_hsm;

static bool has_section_id(runtime::hsm::DeviceId device,
                           runtime::hsm::SectionTag tag,
                           const hsm::SectionId& id) {
    return hsm::section_key(device, tag) == hsm::section_key(id);
}

static bool is_page_aligned(size_t offset) {
    return util::align_size_up(offset, util::min_page_alignment) == offset;
}

// TODO: when HSM becomes the default, compile() no longer needs export_hsm_format(true).
class CpuHsmBlobTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = utils::make_conv_pool_relu({1, 3, 16, 16}, element::f32);
    }

    CompiledModel compile(const std::shared_ptr<Model>& m, AnyMap config = {}) {
        config.emplace(intel_cpu::export_hsm_format(true));
        return core.compile_model(m, utils::DEVICE_CPU, config);
    }

    std::string export_blob(const std::shared_ptr<Model>& m, const AnyMap& config = {}) {
        std::stringstream stream;
        compile(m, config).export_model(stream);
        return stream.str();
    }

    static Tensor infer(CompiledModel compiled, const Tensor& in) {
        auto request = compiled.create_infer_request();
        request.set_input_tensor(in);
        request.infer();
        const auto out = request.get_output_tensor();
        Tensor result(out.get_element_type(), out.get_shape());
        out.copy_to(result);
        return result;
    }

    static size_t weights_section_size(const std::string& blob) {
        const auto reader =
            runtime::hsm::Reader::open(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
        OPENVINO_ASSERT(reader.has_value(), "Not an HSM blob");
        const auto section = reader->section(hsm::weights.device, hsm::weights.tag.id());
        return section ? section->size() : 0;
    }

    std::string export_blob_without(const hsm::SectionId& id) {
        return hsm_repack(export_blob(model), [&](std::vector<HsmEntry>& entries) {
            entries.erase(std::remove_if(entries.begin(),
                                         entries.end(),
                                         [&](const HsmEntry& entry) {
                                             return has_section_id(entry.device, entry.tag.tag, id);
                                         }),
                          entries.end());
        });
    }

    std::string export_blob_with_duplicate(const hsm::SectionId& id) {
        return hsm_repack(export_blob(model), [&](std::vector<HsmEntry>& entries) {
            const auto found = std::find_if(entries.begin(), entries.end(), [&](const HsmEntry& entry) {
                return has_section_id(entry.device, entry.tag.tag, id);
            });
            ASSERT_NE(found, entries.end());
            entries.push_back(*found);
        });
    }

    // The writer does not produce a model id yet, so tests add the section themselves.
    std::string export_blob_with_model_ids(size_t count, runtime::hsm::SectionTag tag = hsm::model_id.tag) {
        return hsm_repack(export_blob(model), [&](std::vector<HsmEntry>& entries) {
            const uint64_t id = 0x1234567890ABCDEFull;
            const auto view = util::as_memory_view(id);
            for (size_t i = 0; i < count; ++i) {
                entries.push_back({hsm::model_id.device, {tag, {}}, std::vector<std::byte>(view.begin(), view.end())});
            }
        });
    }

    void expect_import_rejected(const std::string& blob) {
        std::stringstream stream(blob);
        EXPECT_THROW((void)core.import_model(stream, utils::DEVICE_CPU), Exception);

        Tensor tensor(element::u8, Shape{blob.size()});
        std::memcpy(tensor.data(), blob.data(), blob.size());
        EXPECT_THROW((void)core.import_model(tensor, utils::DEVICE_CPU), Exception);
    }

    Core core;
    std::shared_ptr<Model> model;
};

TEST_F(CpuHsmBlobTest, exported_blob_is_hsm_with_cpu_sections) {
    const auto blob = export_blob(model);
    ASSERT_TRUE(is_hsm(blob));

    const auto reader = runtime::hsm::Reader::open(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(reader->count(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id()), 1u);
    EXPECT_EQ(reader->count(hsm::model_id.device, hsm::model_id.tag.id()), 0u);
    EXPECT_EQ(reader->count(hsm::model_struct.device, hsm::model_struct.tag.id()), 1u);
    EXPECT_EQ(reader->count(hsm::weights.device, hsm::weights.tag.id()), 1u);

    const auto requirements = reader->section(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id());
    ASSERT_TRUE(requirements.has_value());
    const auto bytes = requirements->to_bytes();
    ASSERT_TRUE(bytes.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size()),
              compile(model).get_property(runtime_requirements));
}

TEST_F(CpuHsmBlobTest, weights_and_model_are_separate_page_aligned_sections) {
    const auto blob = export_blob(model);
    const auto reader = runtime::hsm::Reader::open(reinterpret_cast<const std::byte*>(blob.data()), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto find_manifest_entry = [&](const hsm::SectionId& id) {
        const auto found = std::find_if(reader->entries().begin(), reader->entries().end(), [&](const auto& entry) {
            return has_section_id(entry.device, entry.tag, id);
        });
        EXPECT_NE(found, reader->entries().end());
        return *found;
    };
    const auto weights = find_manifest_entry(hsm::weights);
    const auto model_struct = find_manifest_entry(hsm::model_struct);

    EXPECT_GT(weights.size, 0u);
    EXPECT_GT(model_struct.size, 0u);
    EXPECT_TRUE(is_page_aligned(weights.offset));
    EXPECT_TRUE(is_page_aligned(model_struct.offset));
    EXPECT_LE(weights.offset + weights.size, model_struct.offset);
    // The model slot is padded to whole pages, so the manifest after it starts on a page boundary.
    EXPECT_TRUE(is_page_aligned(runtime::hsm::Header::view(blob.data()).manifest_offset));
}

TEST_F(CpuHsmBlobTest, model_without_constants_has_an_empty_weights_section_and_round_trips) {
    auto parameter = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto relu = std::make_shared<op::v0::Relu>(parameter);
    auto no_constants = std::make_shared<Model>(std::make_shared<op::v0::Result>(relu),
                                                    ParameterVector{parameter},
                                                    "no_constants");
    const auto blob = export_blob(no_constants);
    EXPECT_EQ(weights_section_size(blob), 0u);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, utils::DEVICE_CPU);

    const auto in = utils::create_and_fill_tensor(element::f32, Shape{1, 8});
    utils::compare(infer(compile(no_constants), in), infer(imported, in));
}

TEST_F(CpuHsmBlobTest, weightless_constants_are_not_stored_in_the_weights_section) {
    // The first constant is weightless and restored from a file on import; the second has no attribute, so it is
    // still exported.
    const auto weightless_data = utils::create_and_fill_tensor(element::f32, Shape{256, 512});
    const auto plain_data = utils::create_and_fill_tensor(element::f32, Shape{512, 128});
    auto weightless_constant = std::make_shared<op::v0::Constant>(weightless_data);
    weightless_constant->get_rt_info()[WeightlessCacheAttribute::get_type_info_static()] =
        WeightlessCacheAttribute(weightless_data.get_byte_size(), 0, element::f32);
    auto parameter = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 256});
    auto first = std::make_shared<op::v0::MatMul>(parameter, weightless_constant);
    auto second = std::make_shared<op::v0::MatMul>(first, std::make_shared<op::v0::Constant>(plain_data));
    auto weightless_model = std::make_shared<Model>(std::make_shared<op::v0::Result>(second),
                                                        ParameterVector{parameter},
                                                        "weightless");
    const auto in = utils::create_and_fill_tensor(element::f32, Shape{1, 256});
    const auto expected = infer(compile(weightless_model), in);

    const auto weights_file = utils::generateTestFilePrefix() + "_weights.bin";
    std::ofstream(weights_file, std::ios::binary)
        .write(static_cast<const char*>(weightless_data.data()),
               static_cast<std::streamsize>(weightless_data.get_byte_size()));

    const AnyMap weightless_config{cache_mode(CacheMode::OPTIMIZE_SIZE)};
    const auto blob = export_blob(weightless_model, weightless_config);
    EXPECT_GE(weights_section_size(blob), plain_data.get_byte_size());
    EXPECT_LT(weights_section_size(blob), weightless_data.get_byte_size());

    std::stringstream stream(blob);
    auto import_config = weightless_config;
    import_config.emplace(weights_path(weights_file));
    utils::compare(expected, infer(core.import_model(stream, utils::DEVICE_CPU, import_config), in));
    utils::removeFile(weights_file);
}

TEST_F(CpuHsmBlobTest, import_rejects_blob_without_runtime_requirements) {
    expect_import_rejected(export_blob_without(hsm::runtime_requirements));
}

TEST_F(CpuHsmBlobTest, import_rejects_blob_without_model_struct) {
    expect_import_rejected(export_blob_without(hsm::model_struct));
}

TEST_F(CpuHsmBlobTest, import_rejects_blob_without_the_weights_of_a_model_with_constants) {
    expect_import_rejected(export_blob_without(hsm::weights));
}

TEST_F(CpuHsmBlobTest, import_accepts_blob_with_model_id) {
    std::stringstream stream(export_blob_with_model_ids(1));
    EXPECT_NO_THROW(std::ignore = core.import_model(stream, utils::DEVICE_CPU));
}

TEST_F(CpuHsmBlobTest, import_rejects_duplicated_model_id) {
    expect_import_rejected(export_blob_with_model_ids(2));
}

TEST_F(CpuHsmBlobTest, import_ignores_sections_with_unexpected_inline_flag) {
    // Duplicates would be rejected if they matched the model id section.
    const auto pointer_mode_id = runtime::hsm::SectionTag::make(hsm::model_id.tag.id(), /*is_inline=*/false);
    std::stringstream stream(export_blob_with_model_ids(2, pointer_mode_id));
    EXPECT_NO_THROW(std::ignore = core.import_model(stream, utils::DEVICE_CPU));
}

TEST_F(CpuHsmBlobTest, import_rejects_blob_with_model_struct_of_unexpected_inline_flag) {
    const auto blob = hsm_repack(export_blob(model), [](std::vector<HsmEntry>& entries) {
        for (auto& entry : entries) {
            if (has_section_id(entry.device, entry.tag.tag, hsm::model_struct)) {
                entry.tag.tag = runtime::hsm::SectionTag::make(hsm::model_struct.tag.id(), /*is_inline=*/true);
                entry.payload.resize(24);
            }
        }
    });
    expect_import_rejected(blob);
}

TEST_F(CpuHsmBlobTest, import_rejects_duplicated_model_struct) {
    expect_import_rejected(export_blob_with_duplicate(hsm::model_struct));
}

TEST_F(CpuHsmBlobTest, import_rejects_duplicated_weights) {
    expect_import_rejected(export_blob_with_duplicate(hsm::weights));
}

TEST_F(CpuHsmBlobTest, import_rejects_mismatched_runtime_requirements) {
    const auto repacked = hsm_repack(export_blob(model), [](std::vector<HsmEntry>& entries) {
        for (auto& entry : entries) {
            if (has_section_id(entry.device, entry.tag.tag, hsm::runtime_requirements)) {
                entry.payload.front() ^= std::byte{0x01};
            }
        }
    });
    expect_import_rejected(repacked);
}

}  // namespace ov::test
