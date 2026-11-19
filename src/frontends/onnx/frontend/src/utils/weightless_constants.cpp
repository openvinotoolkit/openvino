// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/weightless_constants.hpp"

#include <unordered_map>
#include <vector>

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"

namespace ov::frontend::onnx::weightless {
namespace {

// FNV-1a 64-bit, part of the scheme 1 ID definition. Changing it requires a new kIdScheme.
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

struct Fnv1a64 {
    uint64_t value = kFnvOffsetBasis;

    void add_byte(uint8_t byte) {
        value ^= byte;
        value *= kFnvPrime;
    }

    void add_u64_le(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            add_byte(static_cast<uint8_t>(v >> (8 * i)));
        }
    }

    // Length-prefixed, so that adjacent fields can never be confused.
    void add_string(std::string_view str) {
        add_u64_le(static_cast<uint64_t>(str.size()));
        for (const char c : str) {
            add_byte(static_cast<uint8_t>(c));
        }
    }
};

// 64-bit finalizer of MurmurHash3. FNV-1a alone mixes its low bits poorly, and the ID keeps the low 60 bits.
constexpr uint64_t fmix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

constexpr uint64_t kTagBit = 1ull << 63;
constexpr uint64_t kHashMask = (1ull << 60) - 1;

template <typename Func>
void for_each_node(const std::shared_ptr<ov::Model>& model, Func&& func) {
    for (const auto& node : model->get_ordered_ops()) {
        func(node);
        if (const auto multi_subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (const auto& body : multi_subgraph->get_functions()) {
                if (body) {
                    for_each_node(body, func);
                }
            }
        }
    }
}

const OnnxInitializerId* get_initializer_id(const ov::Node& node) {
    const auto& rt_info = node.get_rt_info();
    const auto it = rt_info.find(OnnxInitializerId::get_type_info_static());
    return it == rt_info.end() ? nullptr : &it->second.as<OnnxInitializerId>();
}

const ov::WeightlessCacheAttribute* get_weightless_cache_attribute(const ov::Node& node) {
    const auto& rt_info = node.get_rt_info();
    const auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
    return it == rt_info.end() ? nullptr : &it->second.as<ov::WeightlessCacheAttribute>();
}

const char* to_string(SourceModelLocation location) {
    switch (location) {
    case SourceModelLocation::UNKNOWN:
        return "UNKNOWN";
    case SourceModelLocation::NONE:
        return "NONE";
    case SourceModelLocation::INTERNAL:
        return "INTERNAL";
    case SourceModelLocation::EXTERNAL:
        return "EXTERNAL";
    }
    return "unrecognized";
}

struct Stats {
    uint64_t count = 0;
    uint64_t bytes = 0;

    void add(uint64_t byte_size) {
        ++count;
        bytes += byte_size;
    }
};
}  // namespace

uint64_t make_synthetic_id(std::string_view name, uint64_t byte_size, const ov::element::Type& element_type) {
    Fnv1a64 hash;
    hash.add_string(name);
    hash.add_u64_le(byte_size);
    hash.add_string(element_type.get_type_name());
    return kTagBit | (kIdScheme << 60) | (fmix64(hash.value) & kHashMask);
}

void mark_initializer(const std::shared_ptr<ov::op::v0::Constant>& constant,
                      const std::string& name,
                      SourceModelLocation location,
                      const std::optional<SourceModelExternalData>& declared,
                      bool in_memory_marker) {
    if (!constant) {
        return;
    }
    auto& rt_info = constant->get_rt_info();
    const auto& wca_type = ov::WeightlessCacheAttribute::get_type_info_static();
    const bool from_file = rt_info.count(wca_type) != 0;
    const uint64_t byte_size = constant->get_byte_size();
    bool backed = false;

    switch (location) {
    case SourceModelLocation::UNKNOWN:
        // Today's rules: a file-backed constant keeps its real offset, and inline data is backed.
        backed = !from_file && !in_memory_marker;
        break;
    case SourceModelLocation::INTERNAL:
        if (from_file) {
            OPENVINO_WARN("[ONNX Frontend] Initializer '",
                          name,
                          "' is reported as stored inside the source model but is read from an external file; "
                          "it is kept in the compiled model.");
            rt_info.erase(wca_type);
        } else {
            backed = !in_memory_marker;
        }
        break;
    case SourceModelLocation::EXTERNAL:
        // The declared location is what the source model provides at import, even when the data is in memory now.
        if (declared && (declared->length == 0 || declared->length == byte_size) &&
            !is_synthetic_id(declared->offset)) {
            rt_info[wca_type] = ov::WeightlessCacheAttribute(byte_size,
                                                             static_cast<size_t>(declared->offset),
                                                             constant->get_element_type());
        } else {
            if (declared) {
                OPENVINO_WARN("[ONNX Frontend] Initializer '",
                              name,
                              "' declares ",
                              declared->length,
                              " bytes of external data but holds ",
                              byte_size,
                              " bytes; it is kept in the compiled model.");
            } else if (from_file) {
                OPENVINO_WARN("[ONNX Frontend] Initializer '",
                              name,
                              "' is reported as external without a declared location; it is kept in the compiled "
                              "model.");
            }
            rt_info.erase(wca_type);
        }
        break;
    case SourceModelLocation::NONE:
    default:
        // Not in the source model: a file offset would point into a file the source model doesn't reference.
        rt_info.erase(wca_type);
        break;
    }

    if (name.empty() || byte_size == 0 || constant->get_element_type() == ov::element::string) {
        backed = false;
    }
    rt_info[OnnxInitializerId::get_type_info_static()] = OnnxInitializerId(name, location, backed);
}

void stamp_weightless_constants(const std::shared_ptr<ov::Model>& model,
                                ov::WeightlessConstants mode,
                                bool has_origin_provider) {
    if (!model) {
        return;
    }
    const bool all = mode == ov::WeightlessConstants::ALL;
    if (!all && !has_origin_provider) {
        return;  // Default behavior: nothing to record.
    }

    // Collect the synthetic IDs first, so that colliding constants can be excluded together.
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<ov::op::v0::Constant>>> by_id;
    if (all) {
        for_each_node(model, [&](const std::shared_ptr<ov::Node>& node) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!constant) {
                return;
            }
            const auto* id = get_initializer_id(*constant);
            if (!id || !id->backed || get_weightless_cache_attribute(*constant)) {
                return;
            }
            by_id[make_synthetic_id(id->name, constant->get_byte_size(), constant->get_element_type())].push_back(
                constant);
        });
        for (const auto& [id, constants] : by_id) {
            if (constants.size() > 1) {
                OPENVINO_WARN("[ONNX Frontend] ",
                              constants.size(),
                              " constants share the weightless ID ",
                              id,
                              " (e.g. '",
                              get_initializer_id(*constants.front())->name,
                              "'); they are kept in the compiled model.");
                continue;
            }
            const auto& constant = constants.front();
            constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(constant->get_byte_size(),
                                             static_cast<size_t>(id),
                                             constant->get_element_type());
        }
        model->set_rt_info(kIdScheme, kRtInfoIdScheme);
    }
    model->set_rt_info(mode, kRtInfoWeightlessConstants);

    // Diagnostics: what the compiled model can reference, and which initializers it must embed.
    Stats synthetic, file, unbacked;
    std::string unbacked_example;
    for_each_node(model, [&](const std::shared_ptr<ov::Node>& node) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant) {
            return;
        }
        const auto byte_size = constant->get_byte_size();
        if (const auto* wca = get_weightless_cache_attribute(*constant)) {
            (is_synthetic_id(wca->bin_offset) ? synthetic : file).add(byte_size);
        } else if (const auto* id = get_initializer_id(*constant); id && byte_size > 0) {
            if (unbacked.count == 0) {
                unbacked_example = id->name + " (" + to_string(id->location) + ")";
            }
            unbacked.add(byte_size);
        }
    });
    model->set_rt_info(synthetic.count, "weightless_stats", "synthetic_count");
    model->set_rt_info(synthetic.bytes, "weightless_stats", "synthetic_bytes");
    model->set_rt_info(file.count, "weightless_stats", "file_count");
    model->set_rt_info(file.bytes, "weightless_stats", "file_bytes");
    model->set_rt_info(unbacked.count, "weightless_stats", "unbacked_count");
    model->set_rt_info(unbacked.bytes, "weightless_stats", "unbacked_bytes");
    if (!unbacked_example.empty()) {
        model->set_rt_info(unbacked_example, "weightless_stats", "unbacked_example");
    }
    OPENVINO_DEBUG("[ONNX Frontend] Weightless constants (",
                   mode,
                   "): ",
                   synthetic.count,
                   " synthetic IDs (",
                   synthetic.bytes,
                   " bytes), ",
                   file.count,
                   " file offsets (",
                   file.bytes,
                   " bytes), ",
                   unbacked.count,
                   " initializers kept in the compiled model (",
                   unbacked.bytes,
                   " bytes",
                   unbacked_example.empty() ? "" : ", e.g. " + unbacked_example,
                   ")");
}

bool has_weightless_constants_mode(const std::shared_ptr<ov::Model>& model) {
    return model && model->has_rt_info(kRtInfoWeightlessConstants);
}

void remove_initializer_ids(const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        return;
    }
    for_each_node(model, [](const std::shared_ptr<ov::Node>& node) {
        node->get_rt_info().erase(OnnxInitializerId::get_type_info_static());
    });
}

}  // namespace ov::frontend::onnx::weightless
