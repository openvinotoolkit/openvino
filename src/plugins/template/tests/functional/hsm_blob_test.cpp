// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>

#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/file_util.hpp"
#include "template/properties.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;

class TemplateHsmBlobTest : public ::testing::Test {
protected:
    static inline auto* const device = utils::DEVICE_TEMPLATE;
    static constexpr hsm::DeviceId template_device_id = 128;
    static constexpr auto blob_info_tag = hsm::SectionTag::make_device_tag(3, /*is_inline=*/true);
    static constexpr size_t blob_info_weights_embedded = 0;
    static constexpr size_t blob_info_weightless = 1;

    struct BlobInfoWire {
        uint32_t layout_version = 1;
        uint32_t flags = 0;
    };

    struct SerializedModel {
        std::string xml;
        std::shared_ptr<ov::AlignedBuffer> weights;
    };

    struct BlobSpec {
        std::string model_xml;
        bool include_model_struct = true;
        std::shared_ptr<ov::AlignedBuffer> weights;
        bool weightless = false;
        hsm::DeviceId weights_device = template_device_id;
        bool include_model_id = true;
        hsm::DeviceId model_id_device = hsm::any_device_id;
        bool reverse_order = false;
    };

    static std::shared_ptr<ov::Model> make_model() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
        input->set_friendly_name("input");
        auto add = std::make_shared<ov::op::v1::Add>(
            input,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {1.f, 2.f, 3.f, 4.f}));
        auto mul = std::make_shared<ov::op::v1::Multiply>(
            add,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {0.5f, -1.f, 2.f, 3.f}));
        auto result = std::make_shared<ov::op::v0::Result>(mul);
        result->set_friendly_name("output");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "hsm_template_model");
    }

    static std::string export_blob(ov::CompiledModel compiled) {
        std::stringstream stream;
        compiled.export_model(stream);
        return stream.str();
    }

    static ov::util::MemoryView view_of(const std::string& data) {
        return {reinterpret_cast<const std::byte*>(data.data()), data.size()};
    }
    static ov::util::MemoryView view_of(const ov::AlignedBuffer& data) {
        return {reinterpret_cast<const std::byte*>(data.get_ptr()), data.size()};
    }

    static SerializedModel serialize_model(const std::shared_ptr<ov::Model>& model) {
        std::stringstream xml_stream, weights_stream;
        ov::pass::Serialize(xml_stream, weights_stream).run_on_model(model);
        const auto weights_str = weights_stream.str();
        auto weights = std::make_shared<ov::AlignedBuffer>(weights_str.size());
        std::memcpy(weights->get_ptr(), weights_str.data(), weights_str.size());
        return {xml_stream.str(), weights};
    }

    static std::string build_blob(const BlobSpec& spec) {
        std::stringstream stream;
        auto writer = hsm::DeferredWriter::open(stream);
        OPENVINO_ASSERT(writer, "Cannot open HSM writer");

        const uint64_t model_id_value = 0xC0FFEE;
        std::bitset<32> blob_info_flags;
        blob_info_flags[blob_info_weights_embedded] = spec.weights != nullptr;
        blob_info_flags[blob_info_weightless] = spec.weightless;
        const BlobInfoWire blob_info{1, static_cast<uint32_t>(blob_info_flags.to_ulong())};

        std::vector<std::function<void()>> sections;
        if (spec.include_model_struct) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(template_device_id, hsm::model_tag, view_of(spec.model_xml)));
            });
        }
        if (spec.weights) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(spec.weights_device, hsm::weights_tag, view_of(*spec.weights)));
            });
        }
        if (spec.include_model_id) {
            sections.push_back([&] {
                OPENVINO_ASSERT(writer->add_section(spec.model_id_device,
                                                    hsm::model_id_tag,
                                                    ov::util::as_memory_view(model_id_value)));
            });
        }
        sections.push_back([&] {
            OPENVINO_ASSERT(
                writer->add_section(template_device_id, blob_info_tag, ov::util::as_memory_view(blob_info)));
        });

        if (spec.reverse_order) {
            std::reverse(sections.begin(), sections.end());
        }
        for (const auto& add_section : sections) {
            add_section();
        }
        OPENVINO_ASSERT(!writer->finalize(), "HSM build failed");
        return stream.str();
    }

    static ov::Tensor infer(ov::CompiledModel& compiled) {
        const auto input =
            ov::test::utils::create_tensor<float>(ov::element::f32, ov::Shape{1, 4}, {1.f, -2.f, 3.f, -4.f});
        auto request = compiled.create_infer_request();
        request.set_input_tensor(input);
        request.infer();
        const auto output = request.get_output_tensor();
        ov::Tensor result(output.get_element_type(), output.get_shape());
        output.copy_to(result);
        return result;
    }

    static std::string read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }

    // Lazily generate one blob per suite; generateTestFilePrefix() requires an active test.
    static void generated_blob() {
        if (generated_blob_path.empty()) {
            const auto serialized = serialize_model(make_model());
            BlobSpec spec;
            spec.model_xml = serialized.xml;
            spec.weights = serialized.weights;
            const auto blob = build_blob(spec);

            generated_blob_path = std::filesystem::path(ov::test::utils::getExecutableDirectory()) /
                                  (ov::test::utils::generateTestFilePrefix() + "_hsm_generated.blob");
            std::ofstream(generated_blob_path, std::ios::binary).write(blob.data(), blob.size());
        }
    }

    static void TearDownTestSuite() {
        std::filesystem::remove(generated_blob_path);
    }

    // Exports in HSM format; @p config selects e.g. the weightless mode.
    std::string export_hsm(ov::AnyMap config = {}) {
        config.emplace(ov::template_plugin::export_hsm_format(true));
        auto compiled = core.compile_model(model, device, config);
        return export_blob(compiled);
    }

    static auto find_entry(std::vector<behavior::HsmEntry>& entries, hsm::DeviceId device, hsm::SectionTag tag) {
        return std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
            return entry.device == device && entry.tag.tag.id() == tag.id();
        });
    }

    // Saves the weights embedded in @p blob to a file that can be used as ov::weights_path.
    std::string save_weights(const std::string& blob) {
        std::stringstream stream(blob);
        const auto reader = hsm::Reader::open(stream);
        OPENVINO_ASSERT(reader, "Not an HSM blob");
        const auto weights = reader->section(template_device_id, hsm::weights);
        OPENVINO_ASSERT(weights, "No weights section");
        const auto bytes = weights->to_bytes();
        OPENVINO_ASSERT(bytes, "Cannot read the weights section");

        weights_file = ov::test::utils::generateTestFilePrefix() + "_hsm_weights.bin";
        std::ofstream(weights_file, std::ios::binary)
            .write(reinterpret_cast<const char*>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
        return weights_file.string();
    }

    void check_cache_format(bool hsm_format) {
        const std::filesystem::path cache_path = "template_hsm_cache_" + utils::generateTestFilePrefix();
        // Plugin-level: Core reads the format from the plugin, not from the compile config, to decide on the header.
        core.set_property(device, ov::template_plugin::export_hsm_format(hsm_format));

        core.compile_model(model, device, ov::cache_path(cache_path));
        auto cached = core.compile_model(model, device, ov::cache_path(cache_path));

        EXPECT_TRUE(cached.get_property(ov::loaded_from_cache));
        utils::compare(reference, infer(cached));
        const auto blobs = utils::listFilesWithExt(ov::util::path_to_string(cache_path), "blob");
        ASSERT_EQ(blobs.size(), 1u);
        EXPECT_EQ(behavior::is_hsm(read_file(blobs.front())), hsm_format);

        utils::removeFilesWithExt(ov::util::path_to_string(cache_path), "blob");
        utils::removeDir(ov::util::path_to_string(cache_path));
    }

    void SetUp() override {
        core = utils::create_core(device);
        model = make_model();
        auto reference_compiled = core.compile_model(model, device);
        reference = infer(reference_compiled);
    }

    void TearDown() override {
        if (ov::util::file_exists(weights_file)) {
            std::filesystem::remove(weights_file);
        }
    }

    static inline std::filesystem::path generated_blob_path;

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::Tensor reference;
    std::filesystem::path weights_file;
};

TEST_F(TemplateHsmBlobTest, weights_path_hint_restores_model_when_weights_section_is_omitted) {
    const auto serialized = serialize_model(model);
    const auto weights_path = ov::test::utils::generateTestFilePrefix() + "_hsm_weights.bin";
    {
        std::ofstream file(weights_path, std::ios::binary);
        file.write((serialized.weights->get_ptr<char>()), static_cast<std::streamsize>(serialized.weights->size()));
    }

    BlobSpec spec;
    spec.model_xml = serialized.xml;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device, {ov::weights_path(weights_path)});
    utils::compare(reference, infer(imported));

    std::filesystem::remove(weights_path);
}

TEST_F(TemplateHsmBlobTest, missing_mandatory_weights_section_is_rejected) {
    BlobSpec spec;
    spec.model_xml = serialize_model(model).xml;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory weights");
}

TEST_F(TemplateHsmBlobTest, missing_mandatory_model_struct_section_is_rejected) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.include_model_struct = false;
    spec.weights = serialized.weights;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory model struct");
}

TEST_F(TemplateHsmBlobTest, model_struct_stored_inline_is_not_recognized) {
    const auto blob = behavior::hsm_repack(export_hsm(), [](std::vector<behavior::HsmEntry>& entries) {
        const auto entry = find_entry(entries, template_device_id, hsm::model_tag);
        ASSERT_NE(entry, entries.end());
        entry->tag = hsm::SectionTag::make(hsm::model_tag.id(), /*is_inline=*/true);
        entry->payload.resize(8);
    });
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory model struct");
}

TEST_F(TemplateHsmBlobTest, blob_info_stored_as_pointer_is_not_recognized) {
    const auto blob = behavior::hsm_repack(export_hsm(), [](std::vector<behavior::HsmEntry>& entries) {
        const auto entry = find_entry(entries, template_device_id, blob_info_tag);
        ASSERT_NE(entry, entries.end());
        entry->tag = hsm::SectionTag::make(blob_info_tag.id(), /*is_inline=*/false);
    });
    std::stringstream stream(blob);

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing or unsupported blob info");
}

TEST_F(TemplateHsmBlobTest, unknown_sections_are_skipped) {
    const auto blob = behavior::hsm_repack(export_hsm(), [](std::vector<behavior::HsmEntry>& entries) {
        entries.push_back({template_device_id, hsm::SectionTag::make_device_tag(100, false), {std::byte{1}}});
        entries.push_back({hsm::any_device_id, hsm::SectionTag::make(hsm::core_tag_id_range_end - 1, true), {}});
    });
    std::stringstream stream(blob);

    OV_ASSERT_NO_THROW(core.import_model(stream, device));
}

TEST_F(TemplateHsmBlobTest, model_id_section_is_optional) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.include_model_id = false;
    const auto blob = build_blob(spec);
    std::stringstream stream(blob);

    EXPECT_NO_THROW(core.import_model(stream, device));
}

TEST_F(TemplateHsmBlobTest, weights_and_model_id_sections_decode_under_any_device_id) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.weights_device = hsm::any_device_id;
    spec.model_id_device = hsm::any_device_id;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, weights_and_model_id_sections_decode_under_template_device_id) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.weights_device = template_device_id;
    spec.model_id_device = template_device_id;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, section_order_does_not_affect_import) {
    const auto serialized = serialize_model(model);
    BlobSpec spec;
    spec.model_xml = serialized.xml;
    spec.weights = serialized.weights;
    spec.reverse_order = true;
    const auto blob = build_blob(spec);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, generated_blob_is_imported_from_file) {
    generated_blob();
    const auto blob = read_file(generated_blob_path);
    const auto& header = hsm::Header::view(blob.data());
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);

    std::stringstream stream(blob);
    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, generated_blob_is_imported_from_file_via_tensor) {
    generated_blob();
    const auto blob_tensor = ov::read_tensor_data(generated_blob_path);

    auto imported = core.import_model(blob_tensor, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, import_restores_compiled_options) {
    std::stringstream stream(export_hsm({ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}));

    auto imported = core.import_model(stream, device);

    EXPECT_EQ(imported.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::THROUGHPUT);
    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, compiled_options_hold_the_cache_relevant_properties) {
    // A non-default value, so a persisted option can be told apart from the plugin default.
    std::stringstream stream(export_hsm({ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}));
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader);
    const auto section = reader->section(template_device_id, hsm::compiled_options);
    ASSERT_TRUE(section);
    const auto bytes = section->to_bytes();
    ASSERT_TRUE(bytes);
    // The section is `key=value` text, one option per line.
    const std::string options(reinterpret_cast<const char*>(bytes->data()), bytes->size());

    // Every property the plugin reports as cache-relevant is stored.
    for (const auto& name : core.get_property(device, ov::internal::caching_properties)) {
        EXPECT_NE(options.find(name + "="), std::string::npos) << name;
    }
    // The stored value is the one used for compilation, not the default.
    EXPECT_NE(options.find(std::string(ov::hint::performance_mode.name()) + "=THROUGHPUT"), std::string::npos)
        << options;
}

TEST_F(TemplateHsmBlobTest, model_id_is_stable_for_a_compiled_model) {
    auto compiled = core.compile_model(model, device, {ov::template_plugin::export_hsm_format(true)});
    const auto model_id = [&] {
        std::stringstream stream(export_blob(compiled));
        const auto reader = hsm::Reader::open(stream);
        OPENVINO_ASSERT(reader, "Not an HSM blob");
        const auto bytes = reader->section(hsm::any_device_id, hsm::model_id)->to_bytes();
        OPENVINO_ASSERT(bytes, "Cannot read the model id section");
        return std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size());
    };

    EXPECT_EQ(model_id(), model_id());
}

TEST_F(TemplateHsmBlobTest, exported_blob_has_all_sections) {
    std::stringstream stream(export_hsm());
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader);

    EXPECT_EQ(reader->entries().size(), 6u);
    EXPECT_EQ(reader->count(hsm::any_device_id, hsm::model_id), 1u);
    EXPECT_EQ(reader->count(hsm::any_device_id, hsm::runtime_requirements), 1u);
    EXPECT_EQ(reader->count(template_device_id, hsm::model), 1u);
    EXPECT_EQ(reader->count(template_device_id, hsm::weights), 1u);
    EXPECT_EQ(reader->count(template_device_id, hsm::compiled_options), 1u);
    EXPECT_EQ(reader->count(template_device_id, blob_info_tag.id()), 1u);
}

TEST_F(TemplateHsmBlobTest, weights_section_starts_on_a_page_boundary) {
    std::stringstream stream(export_hsm());
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader);

    const auto& entries = reader->entries();
    const auto weights = std::find_if(entries.begin(), entries.end(), [](const hsm::ManifestEntry& entry) {
        return entry.device == template_device_id && entry.tag.id() == hsm::weights;
    });

    ASSERT_NE(weights, entries.end());
    EXPECT_EQ(weights->offset % 4096, 0u);
}

TEST_F(TemplateHsmBlobTest, cache_mode_optimize_size_omits_weights_section) {
    std::stringstream stream(export_hsm({ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)}));
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader);

    EXPECT_EQ(reader->count(template_device_id, hsm::model), 1u);
    EXPECT_EQ(reader->count(template_device_id, hsm::weights), 0u);
}

TEST_F(TemplateHsmBlobTest, weightless_blob_is_imported_with_weights_path_hint) {
    const auto weights_path = save_weights(export_hsm());
    std::stringstream stream(export_hsm({ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)}));

    auto imported = core.import_model(stream, device, {ov::weights_path(weights_path)});

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, weightless_blob_restores_weights_path_from_compiled_options) {
    const auto weights_path = save_weights(export_hsm());
    std::stringstream stream(
        export_hsm({ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE), ov::weights_path(weights_path)}));

    auto imported = core.import_model(stream, device);

    utils::compare(reference, infer(imported));
}

TEST_F(TemplateHsmBlobTest, weightless_blob_without_weights_is_rejected) {
    std::stringstream stream(export_hsm({ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)}));

    OV_EXPECT_THROW_HAS_SUBSTRING(core.import_model(stream, device), ov::Exception, "missing mandatory weights");
}

TEST_F(TemplateHsmBlobTest, cache_uses_hsm_format_selected_at_plugin_level) {
    check_cache_format(true);
}

TEST_F(TemplateHsmBlobTest, cache_uses_legacy_format_selected_at_plugin_level) {
    check_cache_format(false);
}
}  // namespace ov::test
