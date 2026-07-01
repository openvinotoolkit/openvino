// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <filesystem>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <pugixml.hpp>
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov::intel_gpu::moe_offload {

// Resolves byte offsets of weight constants from the IR XML file.
// This is a legacy fallback for when WeightlessCacheAttribute and source_buffer
// are not available (older IR formats or non-standard export paths).
class MoeLegacyXmlOffsetResolver {
public:
    explicit MoeLegacyXmlOffsetResolver(const std::string& weights_path);

    bool is_ready() const { return m_ready; }

    // Resolves the byte offset for a constant identified by name, fused names,
    // expected byte size, and MOE context (layer pattern, projection slot).
    // Throws on ambiguity or failure.
    size_t resolve(const std::shared_ptr<ov::op::v0::Constant>& const_op,
                   const std::string& moe_name,
                   size_t offset_slot);

private:
    struct XmlConstEntry {
        size_t offset = 0;
        size_t size = 0;
        bool used = false;
    };

    struct ProjHint {
        std::vector<std::string> patterns;
        std::vector<std::string> suffixes;
    };

    struct SizeCandidate {
        std::string name;
        XmlConstEntry* entry = nullptr;
    };

    std::unordered_map<std::string, std::vector<XmlConstEntry>> m_entries_by_name;
    bool m_ready = false;

    static std::string extract_layer_pattern(const std::string& moe_name);
    static ProjHint get_proj_hint(size_t offset_slot);
    bool resolve_from_name(const std::string& lookup_name, size_t expected_size, size_t& resolved_offset);
    size_t resolve_by_size(const std::string& const_name, size_t expected_size,
                           const std::string& moe_name, size_t offset_slot);
};

}  // namespace ov::intel_gpu::moe_offload