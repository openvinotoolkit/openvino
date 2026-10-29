// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"
#include "hsm_common.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "properties.hpp"

namespace ov::hetero::tests {

namespace {

struct HsmEntry {
    runtime::hsm::DeviceId device;
    runtime::hsm::SectionTagReserved tag;
    std::vector<std::byte> payload;
};

// A blob together with the id of the section that carries the payload of its submodels.
struct ExportedBlob {
    std::string name;
    std::string blob;
    hsm::SectionId payload;
    size_t submodels;
};

// Mock devices cannot cache, so their submodels are stored as IR; CPU stores its own blob.
constexpr auto ir_variant = "ir";
constexpr auto cpu_variant = "cpu_blob";

const std::byte* as_bytes(const std::string& blob) {
    return reinterpret_cast<const std::byte*>(blob.data());
}

bool is_hsm_blob(const std::string& blob) {
    return blob.size() >= sizeof(runtime::hsm::Header) &&
           runtime::hsm::Header::view(blob.data()).magic == runtime::hsm::BlobMagic::single;
}

bool has_section_id(const runtime::hsm::ManifestEntry& entry, const hsm::SectionId& id) {
    return hsm::section_key(entry.device, entry.tag.id()) == hsm::section_key(id);
}

bool has_section_id(const HsmEntry& entry, const hsm::SectionId& id) {
    return hsm::section_key(entry.device, entry.tag.tag.id()) == hsm::section_key(id);
}

// Rebuilds an HSM blob from its sections after `modify` edited them.
std::string hsm_repack(const std::string& blob, const std::function<void(std::vector<HsmEntry>&)>& modify) {
    const auto reader = runtime::hsm::Reader::open(as_bytes(blob), blob.size());
    OPENVINO_ASSERT(reader, "Not an HSM blob");

    std::vector<HsmEntry> entries;
    for (const auto& entry : reader->entries()) {
        const auto section =
            entry.tag.is_inline()
                ? runtime::hsm::Section{entry}
                : runtime::hsm::Section{entry, {as_bytes(blob) + entry.offset, static_cast<size_t>(entry.size)}};
        entries.push_back({entry.device, {entry.tag, entry.tag_reserved}, section.to_bytes().value()});
    }
    modify(entries);

    std::stringstream out;
    auto writer = runtime::hsm::DeferredWriter::open(out);
    OPENVINO_ASSERT(writer, "Cannot open HSM writer");
    for (const auto& entry : entries) {
        OPENVINO_ASSERT(
            writer->add_section(entry.device, entry.tag, {entry.payload.data(), entry.payload.size()}, {4096}),
            "Cannot add HSM section");
    }
    OPENVINO_ASSERT(!writer->finalize(), "HSM repack failed");
    return out.str();
}

// Rebuilds an HSM blob with the payload of every section with the given id replaced by `edit(payload)`.
std::string hsm_edit_payload(const std::string& blob,
                             const hsm::SectionId& id,
                             const std::function<std::string(const std::string&)>& edit) {
    return hsm_repack(blob, [&](std::vector<HsmEntry>& entries) {
        for (auto& entry : entries) {
            if (has_section_id(entry, id)) {
                const auto text = reinterpret_cast<const char*>(entry.payload.data());
                const auto edited = edit(std::string(text, entry.payload.size()));
                entry.payload.assign(reinterpret_cast<const std::byte*>(edited.data()),
                                     reinterpret_cast<const std::byte*>(edited.data()) + edited.size());
            }
        }
    });
}

}  // namespace

// IR frontend is needed for import
#ifdef IR_FRONTEND_ENABLED
class HeteroHsmBlobTests : public HeteroTests {
protected:
    ov::CompiledModel compile(const std::shared_ptr<ov::Model>& model,
                              const std::string& priorities,
                              bool hsm_format = true) {
        return core.compile_model(model,
                                  ov::test::utils::DEVICE_HETERO,
                                  ov::device::priorities(priorities),
                                  ov::hetero::export_hsm_format(hsm_format));
    }

    static std::string export_blob(ov::CompiledModel compiled_model) {
        std::stringstream stream;
        compiled_model.export_model(stream);
        return stream.str();
    }

    ov::CompiledModel import_from_stream(const std::string& blob) {
        std::stringstream stream(blob);
        return core.import_model(stream, ov::test::utils::DEVICE_HETERO, {});
    }

    ov::CompiledModel import_from_tensor(const std::string& blob) {
        ov::Tensor tensor(ov::element::u8, ov::Shape{blob.size()});
        if (!blob.empty()) {
            std::memcpy(tensor.data(), blob.data(), blob.size());
        }
        return core.import_model(tensor, ov::test::utils::DEVICE_HETERO, {});
    }

    void expect_runs(ov::CompiledModel compiled_model) {
        ASSERT_EQ(1, compiled_model.inputs().size());
        ASSERT_EQ(1, compiled_model.outputs().size());
        auto request = compiled_model.create_infer_request();
        auto input =
            create_and_fill_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape());
        request.set_input_tensor(input);
        ASSERT_NO_THROW(request.infer());
        auto output = request.get_output_tensor();
        EXPECT_EQ(compiled_model.output().get_shape(), output.get_shape());
        EXPECT_EQ(compiled_model.output().get_element_type(), output.get_element_type());
    }

    void expect_import_rejected(const std::string& blob) {
        EXPECT_THROW(import_from_stream(blob), ov::Exception);
        EXPECT_THROW(import_from_tensor(blob), ov::Exception);
    }

    ExportedBlob export_variant(const std::string& name) {
        if (name == cpu_variant) {
            return {name,
                    export_blob(compile(create_model_with_reshape(), ov::test::utils::DEVICE_CPU)),
                    hsm::submodel_blob,
                    1};
        }
        return {name, export_blob(compile(create_model_with_subtract(), "MOCK0,MOCK1")), hsm::submodel_ir_xml, 2};
    }
};

class HeteroHsmBlobVariantTests : public HeteroHsmBlobTests, public ::testing::WithParamInterface<std::string> {
protected:
    void SetUp() override {
        HeteroHsmBlobTests::SetUp();
        const auto devices = core.get_available_devices();
        if (GetParam() == cpu_variant &&
            std::find(devices.begin(), devices.end(), ov::test::utils::DEVICE_CPU) == devices.end()) {
            GTEST_SKIP() << "CPU plugin is not available";
        }
        variant = export_variant(GetParam());
    }

    ExportedBlob variant;
};

INSTANTIATE_TEST_SUITE_P(Hsm,
                         HeteroHsmBlobVariantTests,
                         ::testing::Values(ir_variant, cpu_variant),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

TEST_P(HeteroHsmBlobVariantTests, export_is_hsm_with_hetero_sections_per_submodel) {
    ASSERT_TRUE(is_hsm_blob(variant.blob));

    const auto reader = runtime::hsm::Reader::open(as_bytes(variant.blob), variant.blob.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(1u, reader->count(hsm::runtime_requirements.device, hsm::runtime_requirements.tag.id()));
    EXPECT_EQ(1u, reader->count(hsm::metadata.device, hsm::metadata.tag.id()));
    EXPECT_EQ(variant.submodels, reader->count(variant.payload.device, variant.payload.tag.id()));

    std::vector<uint32_t> indexes;
    for (const auto& entry : reader->entries()) {
        if (has_section_id(entry, variant.payload)) {
            indexes.push_back(hsm::decode_section_index(entry.tag_reserved));
        }
        if (has_section_id(entry, hsm::submodel_blob) || has_section_id(entry, hsm::submodel_ir_weights)) {
            EXPECT_EQ(0u, entry.offset % ov::util::min_page_alignment);
        }
    }
    std::vector<uint32_t> expected(variant.submodels);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(expected, indexes);
}

TEST_P(HeteroHsmBlobVariantTests, import_from_stream_and_tensor_runs) {
    expect_runs(import_from_stream(variant.blob));
    expect_runs(import_from_tensor(variant.blob));
}

TEST_F(HeteroHsmBlobTests, export_to_non_seekable_stream) {
    auto compiled_model = compile(create_model_with_subtract(), "MOCK0,MOCK1");
    NonSeekableOutputBuffer buffer;
    std::ostream stream(&buffer);

    compiled_model.export_model(stream);

    ASSERT_TRUE(stream);
    EXPECT_TRUE(is_hsm_blob(buffer.str()));
    expect_runs(import_from_stream(buffer.str()));
}

TEST_F(HeteroHsmBlobTests, legacy_blob_imports_in_hsm_mode) {
    const auto legacy = export_blob(compile(create_model_with_subtract(), "MOCK0,MOCK1", false));
    EXPECT_FALSE(is_hsm_blob(legacy));

    core.set_property(ov::test::utils::DEVICE_HETERO, ov::hetero::export_hsm_format(true));
    expect_runs(import_from_stream(legacy));
    expect_runs(import_from_tensor(legacy));
}

TEST_F(HeteroHsmBlobTests, plugin_property_selects_format_and_is_reported) {
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::hetero::export_hsm_format(false));
    EXPECT_FALSE(core.get_property(ov::test::utils::DEVICE_HETERO, ov::internal::emit_hsm_format));
    auto model = create_model_with_subtract();
    const auto default_blob =
        export_blob(core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1")));
    EXPECT_FALSE(is_hsm_blob(default_blob));

    core.set_property(ov::test::utils::DEVICE_HETERO, ov::hetero::export_hsm_format(true));
    EXPECT_TRUE(core.get_property(ov::test::utils::DEVICE_HETERO, ov::internal::emit_hsm_format));
    const auto hsm_blob =
        export_blob(core.compile_model(model, ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1")));
    EXPECT_TRUE(is_hsm_blob(hsm_blob));
}

TEST_F(HeteroHsmBlobTests, re_export_of_imported_model_round_trips) {
    // Only the IR variant: the CPU plugin cannot re-export a model it has imported.
    const auto variant = export_variant(ir_variant);
    std::stringstream stream(variant.blob);
    const auto first = core.import_model(stream, ov::test::utils::DEVICE_HETERO, {ov::hetero::export_hsm_format(true)});

    const auto re_exported = export_blob(first);

    EXPECT_TRUE(is_hsm_blob(re_exported));
    expect_runs(import_from_tensor(re_exported));
}

TEST_P(HeteroHsmBlobVariantTests, unknown_sections_are_skipped) {
    const auto blob = hsm_repack(variant.blob, [](std::vector<HsmEntry>& entries) {
        entries.push_back({runtime::hsm::any_device_id,
                           {runtime::hsm::SectionTag::make(runtime::hsm::core_tag_id_range_end - 1, false), {}},
                           std::vector<std::byte>(19, std::byte{1})});
        entries.push_back({250, {runtime::hsm::SectionTag::make_device_tag(0x123, false), {}}, {std::byte{1}}});
        entries.push_back({hsm::device_id,
                           {runtime::hsm::SectionTag::make_device_tag(0x456, false), {}},
                           std::vector<std::byte>(7, std::byte{2})});
    });

    expect_runs(import_from_stream(blob));
    expect_runs(import_from_tensor(blob));
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_blob_without_mandatory_sections) {
    for (const auto& id : {hsm::runtime_requirements, hsm::metadata, variant.payload}) {
        SCOPED_TRACE(id.tag.id());
        expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
            entries.erase(std::remove_if(entries.begin(),
                                         entries.end(),
                                         [&](const HsmEntry& entry) {
                                             return has_section_id(entry, id);
                                         }),
                          entries.end());
        }));
    }
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_missing_submodel_section) {
    expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
        const auto found = std::find_if(entries.begin(), entries.end(), [&](const HsmEntry& entry) {
            return has_section_id(entry, variant.payload) && hsm::decode_section_index(entry.tag.bytes) == 0;
        });
        ASSERT_NE(found, entries.end());
        entries.erase(found);
    }));
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_duplicated_sections) {
    for (const auto& id : {hsm::runtime_requirements, hsm::metadata, variant.payload}) {
        SCOPED_TRACE(id.tag.id());
        expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
            const auto found = std::find_if(entries.begin(), entries.end(), [&](const HsmEntry& entry) {
                return has_section_id(entry, id);
            });
            ASSERT_NE(found, entries.end());
            entries.push_back(*found);
        }));
    }
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_submodel_with_both_blob_and_ir_sections) {
    const bool has_blob = hsm::section_key(variant.payload) == hsm::section_key(hsm::submodel_blob);
    const auto& other = has_blob ? hsm::submodel_ir_xml : hsm::submodel_blob;
    expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
        entries.push_back(
            {hsm::device_id, {other.tag, hsm::encode_section_index(0)}, std::vector<std::byte>(4, std::byte{'x'})});
    }));
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_submodel_index_out_of_range) {
    expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
        for (auto& entry : entries) {
            if (has_section_id(entry, variant.payload)) {
                entry.tag.bytes = hsm::encode_section_index(7);
            }
        }
    }));
}

TEST_F(HeteroHsmBlobTests, import_rejects_incompatible_runtime_requirements) {
    const auto blob = export_variant(ir_variant).blob;
    for (const std::string requirements : {"hetero=2", "hetero=", "hetero=1 ", "garbage", ""}) {
        SCOPED_TRACE(requirements);
        expect_import_rejected(hsm_edit_payload(blob, hsm::runtime_requirements, [&](const std::string&) {
            return requirements;
        }));
    }
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_truncated_blob) {
    auto blob = variant.blob;
    blob.pop_back();

    expect_import_rejected(blob);
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_blob_truncated_at_any_point) {
    const auto size = variant.blob.size();
    for (const size_t cut : {size_t{0},
                             size_t{1},
                             size_t{31},
                             size_t{32},
                             size_t{33},
                             size_t{4095},
                             size_t{4096},
                             size / 2,
                             size - 33,
                             size - 32,
                             size - 31,
                             size - 1}) {
        SCOPED_TRACE(cut);
        expect_import_rejected(variant.blob.substr(0, cut));
    }
}

TEST_F(HeteroHsmBlobTests, import_rejects_tiny_or_garbage_input) {
    const auto blob = export_variant(ir_variant).blob;
    const auto magic_only = blob.substr(0, sizeof(runtime::hsm::Header));
    for (const auto& input :
         {std::string{}, std::string("x"), std::string(4096, '\0'), std::string(4096, '\xFF'), magic_only}) {
        SCOPED_TRACE(input.size());
        expect_import_rejected(input);
    }
}

TEST_F(HeteroHsmBlobTests, import_never_crashes_on_corrupted_header_or_manifest) {
    const auto blob = export_variant(ir_variant).blob;
    const auto header = runtime::hsm::Header::view(blob.data());
    std::vector<size_t> positions(sizeof(runtime::hsm::Header));
    std::iota(positions.begin(), positions.end(), size_t{0});
    for (auto pos = static_cast<size_t>(header.manifest_offset);
         pos < static_cast<size_t>(header.manifest_offset + header.manifest_size);
         ++pos) {
        positions.push_back(pos);
    }

    for (const auto pos : positions) {
        for (const auto value : {'\x00', '\xFF'}) {
            auto corrupted = blob;
            corrupted[pos] = value;
            SCOPED_TRACE("byte " + std::to_string(pos) + " = " + std::to_string(static_cast<int>(value)));
            // A corrupted copy may still be valid; the only requirement is an ov exception instead of a crash.
            try {
                import_from_stream(corrupted);
            } catch (const ov::Exception&) {
            }
            try {
                import_from_tensor(corrupted);
            } catch (const ov::Exception&) {
            }
        }
    }
}

TEST_F(HeteroHsmBlobTests, import_rejects_malformed_metadata) {
    const auto blob = export_variant(ir_variant).blob;
    const auto edit_metadata = [&](const std::function<std::string(const std::string&)>& edit) {
        return hsm_edit_payload(blob, hsm::metadata, edit);
    };

    expect_import_rejected(edit_metadata([](const std::string&) {
        return std::string{};
    }));
    expect_import_rejected(edit_metadata([](const std::string&) {
        return std::string{"not xml"};
    }));
    expect_import_rejected(edit_metadata([](const std::string&) {
        return std::string{"<hetero/>"};
    }));
    // Fewer submodels in the metadata than sections in the blob.
    expect_import_rejected(edit_metadata([](std::string text) {
        const auto begin = text.rfind("<compiled_submodel ");
        const auto end = text.find("/>", begin);
        EXPECT_NE(std::string::npos, begin);
        EXPECT_NE(std::string::npos, end);
        return text.erase(begin, end + 2 - begin);
    }));
    // Mapping that points to a submodel that does not exist.
    expect_import_rejected(edit_metadata([](std::string text) {
        const std::string from = "submodel_idx=\"0\"";
        const auto pos = text.find(from);
        EXPECT_NE(std::string::npos, pos);
        return text.replace(pos, from.size(), "submodel_idx=\"99\"");
    }));
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_corrupted_submodel_payload) {
    for (const auto& payload : {std::string{}, std::string{"x"}, std::string(64, '\xFF')}) {
        SCOPED_TRACE(payload.size());
        expect_import_rejected(hsm_edit_payload(variant.blob, variant.payload, [&](const std::string&) {
            return payload;
        }));
    }
}

TEST_P(HeteroHsmBlobVariantTests, import_rejects_inline_sections) {
    for (const auto& id : {hsm::runtime_requirements, hsm::metadata, variant.payload}) {
        SCOPED_TRACE(id.tag.id());
        expect_import_rejected(hsm_repack(variant.blob, [&](std::vector<HsmEntry>& entries) {
            for (auto& entry : entries) {
                if (has_section_id(entry, id)) {
                    entry.tag.tag = runtime::hsm::SectionTag::make(entry.tag.tag.id(), true);
                    entry.payload.resize(8);
                }
            }
        }));
    }
}

TEST_P(HeteroHsmBlobVariantTests, import_from_multi_dimensional_tensor_runs) {
    ov::Tensor tensor(ov::element::u8, ov::Shape{1, variant.blob.size()});
    std::memcpy(tensor.data(), variant.blob.data(), variant.blob.size());
    expect_runs(core.import_model(tensor, ov::test::utils::DEVICE_HETERO, {}));
}
#endif

}  // namespace ov::hetero::tests
