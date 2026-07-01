// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "plugin/ops/moe_legacy_xml_offset_resolver.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using ov::intel_gpu::moe_offload::MoeLegacyXmlOffsetResolver;

namespace {

// RAII helper that writes a synthetic IR xml (only the Const layers the resolver
// cares about) to a unique temp path and removes it on destruction.
class TempIr {
public:
    explicit TempIr(const std::string& tag, const std::string& layers_xml) {
        auto dir = std::filesystem::temp_directory_path();
        m_xml = (dir / ("moe_xml_resolver_" + tag + ".xml")).string();
        m_bin = (dir / ("moe_xml_resolver_" + tag + ".bin")).string();

        std::ofstream ofs(m_xml, std::ios::trunc);
        ofs << "<net name=\"test\" version=\"11\">\n  <layers>\n"
            << layers_xml
            << "  </layers>\n  <edges/>\n</net>\n";
    }

    ~TempIr() {
        std::error_code ec;
        std::filesystem::remove(m_xml, ec);
        std::filesystem::remove(m_bin, ec);
    }

    // The resolver derives the xml path from the weights path by swapping the
    // extension, so handing it the .bin sibling exercises that logic too.
    const std::string& weights_path() const { return m_bin; }

private:
    std::string m_xml;
    std::string m_bin;
};

std::string const_layer(const std::string& name, size_t offset, size_t size) {
    return "    <layer name=\"" + name + "\" type=\"Const\">\n"
           "      <data element_type=\"u8\" offset=\"" + std::to_string(offset) +
           "\" size=\"" + std::to_string(size) + "\"/>\n"
           "    </layer>\n";
}

// Builds a u8 constant of exactly `byte_size` bytes with the given friendly name.
std::shared_ptr<ov::op::v0::Constant> make_const(const std::string& name, size_t byte_size) {
    auto c = std::make_shared<ov::op::v0::Constant>(ov::element::u8,
                                                    ov::Shape{byte_size},
                                                    std::vector<uint8_t>(byte_size, 0));
    c->set_friendly_name(name);
    return c;
}

}  // namespace

// ──────────────────────────────────────────────────
// Case 1: exact friendly-name match
// ──────────────────────────────────────────────────
TEST(moe_xml_offset_resolver, exact_name_match) {
    TempIr ir("exact", const_layer("const_gate", 1000, 16));
    MoeLegacyXmlOffsetResolver resolver(ir.weights_path());
    ASSERT_TRUE(resolver.is_ready());

    auto c = make_const("const_gate", 16);
    EXPECT_EQ(resolver.resolve(c, /*moe_name=*/"layers.0.experts", /*offset_slot=*/0), 1000U);
}

// ──────────────────────────────────────────────────
// Case 2: fused-name match (exact name absent, size ambiguous on purpose)
// ──────────────────────────────────────────────────
TEST(moe_xml_offset_resolver, fused_name_match) {
    // Two size-16 entries → by-size alone is ambiguous, so a successful resolve
    // proves the fused-name path fired before falling through to by-size.
    TempIr ir("fused", const_layer("orig_up_proj", 2000, 16) + const_layer("other_const", 9999, 16));
    MoeLegacyXmlOffsetResolver resolver(ir.weights_path());
    ASSERT_TRUE(resolver.is_ready());

    auto c = make_const("renamed_const", 16);
    c->get_rt_info()[ov::FusedNames::get_type_info_static()] = ov::FusedNames("orig_up_proj");

    EXPECT_EQ(resolver.resolve(c, "layers.0.experts", 1), 2000U);
}

// ──────────────────────────────────────────────────
// Case 3: by-size fallback (unique size, name/fused-name absent)
// ──────────────────────────────────────────────────
TEST(moe_xml_offset_resolver, by_size_match) {
    TempIr ir("bysize", const_layer("weird_name_xyz", 3000, 24));
    MoeLegacyXmlOffsetResolver resolver(ir.weights_path());
    ASSERT_TRUE(resolver.is_ready());

    auto c = make_const("const_down", 24);
    EXPECT_EQ(resolver.resolve(c, "layers.0.experts", 2), 3000U);
}

// ──────────────────────────────────────────────────
// Case 4: ambiguity throws (duplicate name, same size)
// ──────────────────────────────────────────────────
TEST(moe_xml_offset_resolver, ambiguous_name_throws) {
    TempIr ir("ambig", const_layer("dup_const", 4000, 16) + const_layer("dup_const", 5000, 16));
    MoeLegacyXmlOffsetResolver resolver(ir.weights_path());
    ASSERT_TRUE(resolver.is_ready());

    auto c = make_const("dup_const", 16);
    EXPECT_THROW(resolver.resolve(c, "layers.0.experts", 0), ov::Exception);
}

// ──────────────────────────────────────────────────
// Empty weights path → resolver stays not-ready (xml fallback unavailable)
// ──────────────────────────────────────────────────
TEST(moe_xml_offset_resolver, empty_path_not_ready) {
    MoeLegacyXmlOffsetResolver resolver("");
    EXPECT_FALSE(resolver.is_ready());
}
