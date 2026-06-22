// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pugixml.hpp>
#include "openvino/core/except.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov::intel_gpu::moe_offload {

// Resolves byte offsets of weight constants from the IR XML file.
// This is a legacy fallback for when WeightlessCacheAttribute and source_buffer
// are not available (older IR formats or non-standard export paths).
class XmlOffsetResolver {
public:
    explicit XmlOffsetResolver(const std::string& weights_path) {
        if (weights_path.empty())
            return;

        std::filesystem::path xml_path(weights_path);
        xml_path.replace_extension(".xml");
        OPENVINO_ASSERT(std::filesystem::exists(xml_path), "IR xml file is not found: ", xml_path.string());

        pugi::xml_document doc;
        OPENVINO_ASSERT(doc.load_file(xml_path.string().c_str()), "Failed to parse IR xml file: ", xml_path.string());

        auto net = doc.child("net");
        auto layers = net.child("layers");
        for (auto layer = layers.child("layer"); layer; layer = layer.next_sibling("layer")) {
            const auto type_attr = layer.attribute("type");
            if (!type_attr || std::string(type_attr.value()) != "Const")
                continue;

            const auto data = layer.child("data");
            const auto name_attr = layer.attribute("name");
            const auto offset_attr = data.attribute("offset");
            const auto size_attr = data.attribute("size");
            if (!data || !name_attr || !offset_attr || !size_attr)
                continue;

            XmlConstEntry entry;
            try {
                entry.offset = static_cast<size_t>(std::stoull(offset_attr.value()));
                entry.size = static_cast<size_t>(std::stoull(size_attr.value()));
            } catch (const std::exception& e) {
                OPENVINO_THROW("Failed to parse MOE weight offset/size from XML attribute: ", e.what(),
                               " (name=", name_attr.value(), ", offset='", offset_attr.value(),
                               "', size='", size_attr.value(), "')");
            }
            m_entries_by_name[name_attr.value()].push_back(entry);
        }
        m_ready = true;
    }

    bool is_ready() const { return m_ready; }

    // Resolves the byte offset for a constant identified by name, fused names,
    // expected byte size, and MOE context (layer pattern, projection slot).
    // Throws on ambiguity or failure.
    size_t resolve(const std::shared_ptr<ov::op::v0::Constant>& const_op,
                   const std::string& moe_name,
                   size_t offset_slot) {
        const auto& name = const_op->get_friendly_name();
        const size_t expected_size = const_op->get_byte_size();
        size_t resolved_offset = 0;

        // 1. Exact name match
        if (resolve_from_name(name, expected_size, resolved_offset))
            return resolved_offset;

        // 2. Fused names
        std::set<std::string> fused_names_unique;
        for (const auto& fused_name : ov::getFusedNamesVector(const_op)) {
            if (!fused_name.empty() && fused_name != name)
                fused_names_unique.insert(fused_name);
        }
        for (const auto& fused_name : fused_names_unique) {
            if (resolve_from_name(fused_name, expected_size, resolved_offset))
                return resolved_offset;
        }

        // 3. By-size with disambiguation
        return resolve_by_size(name, expected_size, moe_name, offset_slot);
    }

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

    std::unordered_map<std::string, std::vector<XmlConstEntry>> m_entries_by_name;
    bool m_ready = false;

    static std::string extract_layer_pattern(const std::string& moe_name) {
        auto pos = moe_name.find("layers.");
        if (pos == std::string::npos) return {};
        auto end = moe_name.find(".experts", pos);
        if (end == std::string::npos) {
            end = moe_name.find("/experts", pos);
            if (end == std::string::npos) return {};
            return moe_name.substr(pos, end - pos + 8);
        }
        return moe_name.substr(pos, end - pos + 8);
    }

    static ProjHint get_proj_hint(size_t offset_slot) {
        size_t proj_idx = offset_slot % 3;
        ProjHint hint;
        if (proj_idx == 0) {
            hint.patterns = {"VariadicSplit.0", "gate_proj", "gate"};
        } else if (proj_idx == 1) {
            hint.patterns = {"VariadicSplit.1", "up_proj", "up"};
        } else {
            hint.patterns = {"down_proj", "VariadicSplit.2"};
        }
        if (offset_slot < 3) {
            hint.suffixes = {};
        } else if (offset_slot < 6) {
            hint.suffixes = {"/scale"};
        } else {
            hint.suffixes = {"/zero_point"};
        }
        return hint;
    }

    bool resolve_from_name(const std::string& lookup_name, size_t expected_size, size_t& resolved_offset) {
        auto it = m_entries_by_name.find(lookup_name);
        if (it == m_entries_by_name.end())
            return false;

        size_t match_count = 0;
        XmlConstEntry* matched_entry = nullptr;
        for (auto& entry : it->second) {
            if (!entry.used && entry.size == expected_size) {
                match_count++;
                if (matched_entry == nullptr)
                    matched_entry = &entry;
            }
        }

        if (match_count == 1 && matched_entry != nullptr) {
            matched_entry->used = true;
            resolved_offset = matched_entry->offset;
            return true;
        }

        if (match_count > 1) {
            OPENVINO_THROW("Ambiguous xml offset resolution for MOE3GemmFusedCompressed constant input: ",
                           lookup_name, ", byte_size=", expected_size, ", candidates=", match_count);
        }
        return false;
    }

    struct SizeCandidate {
        std::string name;
        XmlConstEntry* entry = nullptr;
    };

    size_t resolve_by_size(const std::string& const_name, size_t expected_size,
                           const std::string& moe_name, size_t offset_slot) {
        std::vector<SizeCandidate> size_candidates;
        for (auto& kv : m_entries_by_name) {
            for (auto& entry : kv.second) {
                if (!entry.used && entry.size == expected_size)
                    size_candidates.push_back(SizeCandidate{kv.first, &entry});
            }
        }

        if (size_candidates.size() == 1 && size_candidates[0].entry != nullptr) {
            size_candidates[0].entry->used = true;
            return size_candidates[0].entry->offset;
        }

        if (size_candidates.size() > 1) {
            std::string layer_pat = extract_layer_pattern(moe_name);
            ProjHint hint = get_proj_hint(offset_slot);

            std::vector<SizeCandidate> layer_filtered;
            if (!layer_pat.empty()) {
                for (auto& sc : size_candidates) {
                    if (sc.name.find(layer_pat) != std::string::npos)
                        layer_filtered.push_back(sc);
                }
            }

            if (layer_filtered.size() == 1 && layer_filtered[0].entry != nullptr) {
                layer_filtered[0].entry->used = true;
                return layer_filtered[0].entry->offset;
            }

            auto& search_pool = layer_filtered.empty() ? size_candidates : layer_filtered;
            for (const auto& pat : hint.patterns) {
                std::vector<SizeCandidate> proj_filtered;
                for (auto& sc : search_pool) {
                    if (sc.name.find(pat) != std::string::npos)
                        proj_filtered.push_back(sc);
                }
                if (proj_filtered.size() == 1 && proj_filtered[0].entry != nullptr) {
                    proj_filtered[0].entry->used = true;
                    return proj_filtered[0].entry->offset;
                }
                if (proj_filtered.size() > 1 && !hint.suffixes.empty()) {
                    for (const auto& suffix : hint.suffixes) {
                        std::vector<SizeCandidate> suffix_filtered;
                        for (auto& sc : proj_filtered) {
                            if (sc.name.size() >= suffix.size() &&
                                sc.name.compare(sc.name.size() - suffix.size(), suffix.size(), suffix) == 0)
                                suffix_filtered.push_back(sc);
                        }
                        if (suffix_filtered.size() == 1 && suffix_filtered[0].entry != nullptr) {
                            suffix_filtered[0].entry->used = true;
                            return suffix_filtered[0].entry->offset;
                        }
                    }
                }
            }

            if (offset_slot < 3 && !search_pool.empty()) {
                std::vector<SizeCandidate> weight_filtered;
                for (auto& sc : search_pool) {
                    bool is_scale_or_zp = (sc.name.find("/scale") != std::string::npos) ||
                                          (sc.name.find("/zero_point") != std::string::npos);
                    if (!is_scale_or_zp)
                        weight_filtered.push_back(sc);
                }
                if (weight_filtered.size() == 1 && weight_filtered[0].entry != nullptr) {
                    weight_filtered[0].entry->used = true;
                    return weight_filtered[0].entry->offset;
                }
                for (const auto& pat : hint.patterns) {
                    std::vector<SizeCandidate> proj_wt_filtered;
                    for (auto& sc : weight_filtered) {
                        if (sc.name.find(pat) != std::string::npos)
                            proj_wt_filtered.push_back(sc);
                    }
                    if (proj_wt_filtered.size() == 1 && proj_wt_filtered[0].entry != nullptr) {
                        proj_wt_filtered[0].entry->used = true;
                        return proj_wt_filtered[0].entry->offset;
                    }
                }
            }

            std::ostringstream oss;
            const size_t max_log = 8;
            for (size_t i = 0; i < std::min(max_log, size_candidates.size()); i++) {
                if (i > 0) oss << ';';
                oss << size_candidates[i].name << '@' << size_candidates[i].entry->offset;
            }

            OPENVINO_THROW("Ambiguous xml offset resolution for MOE3GemmFusedCompressed constant input: ",
                           const_name, ", byte_size=", expected_size,
                           ", size_candidates=", size_candidates.size(),
                           ", layer_pat=", layer_pat,
                           ", offset_slot=", offset_slot,
                           ", moe_name=", moe_name,
                           ", sample_candidates=", oss.str());
        }

        OPENVINO_THROW("Unable to resolve xml offset for MOE3GemmFusedCompressed constant input: ",
                       const_name, ", byte_size=", expected_size);
    }
};

}  // namespace ov::intel_gpu::moe_offload
