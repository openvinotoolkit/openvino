// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/except.hpp"

#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cldnn {
namespace ocl {

// Self-contained text form of device_info, produced on a HW-attached machine during enumeration and
// consumed by the HW-free compiler process, which has no device to query.
constexpr const char* device_descriptor_version = "2";

namespace detail {

inline std::map<std::string, std::string> parse_kv(const std::string& descriptor) {
    std::map<std::string, std::string> kv;
    std::istringstream is(descriptor);
    std::string line;
    while (std::getline(is, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

inline std::string to_hex(const uint8_t* data, size_t count) {
    std::ostringstream os;
    for (size_t i = 0; i < count; i++)
        os << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]);
    return os.str();
}

inline void from_hex(const std::string& s, uint8_t* data, size_t count) {
    for (size_t i = 0; i < count && (i * 2 + 1) < s.size(); i++)
        data[i] = static_cast<uint8_t>(std::stoul(s.substr(i * 2, 2), nullptr, 16));
}

inline std::string join_sizes(const std::vector<size_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); i++)
        os << (i ? "," : "") << v[i];
    return os.str();
}

inline std::vector<size_t> split_sizes(const std::string& s) {
    std::vector<size_t> v;
    std::istringstream is(s);
    std::string tok;
    while (std::getline(is, tok, ','))
        if (!tok.empty())
            v.push_back(std::stoull(tok));
    return v;
}

}  // namespace detail

// Every field that can influence a compile-time decision must be listed here; anything omitted
// silently falls back to a default in the compiler process and can change kernel/layout selection.
#define OV_GPU_DEVICE_DESCRIPTOR_FIELDS(NUM, STR)                     \
    NUM(execution_units_count)                                        \
    NUM(gpu_frequency)                                                \
    NUM(max_work_group_size)                                          \
    NUM(max_local_mem_size)                                           \
    NUM(max_global_mem_size)                                          \
    NUM(max_alloc_mem_size)                                           \
    NUM(max_global_cache_size)                                        \
    NUM(max_image2d_width)                                            \
    NUM(max_image2d_height)                                           \
    NUM(supports_fp16)                                                \
    NUM(supports_fp64)                                                \
    NUM(supports_fp16_denorms)                                        \
    NUM(supports_khr_subgroups)                                       \
    NUM(supports_intel_subgroups)                                     \
    NUM(supports_intel_subgroups_short)                               \
    NUM(supports_intel_subgroups_char)                                \
    NUM(supports_intel_required_subgroup_size)                        \
    NUM(supports_queue_families)                                      \
    NUM(supports_image)                                               \
    NUM(supports_intel_planar_yuv)                                    \
    NUM(supports_work_group_collective_functions)                     \
    NUM(supports_imad)                                                \
    NUM(supports_immad)                                               \
    NUM(supports_mutable_command_list)                                \
    NUM(supports_usm)                                                 \
    NUM(has_separate_cache)                                           \
    NUM(supports_cp_offload)                                          \
    NUM(supports_counter_based_events)                                \
    NUM(supports_leo)                                                 \
    NUM(vendor_id)                                                    \
    NUM(ip_version)                                                   \
    NUM(device_id)                                                    \
    NUM(num_slices)                                                   \
    NUM(num_sub_slices_per_slice)                                     \
    NUM(num_eus_per_sub_slice)                                        \
    NUM(num_threads_per_eu)                                           \
    NUM(num_ccs)                                                      \
    NUM(sub_device_idx)                                               \
    NUM(timer_resolution)                                             \
    NUM(kernel_timestamp_valid_bits)                                  \
    NUM(compute_queue_group_ordinal)                                  \
    NUM(device_memory_ordinal)                                        \
    STR(dev_name)                                                     \
    STR(driver_version)

// oneDNN keeps its own device_info with its own private layout, so the only robust way to reproduce
// it offline is to carry the blob oneDNN itself produced. It has no version tag of its own, hence
// onednn_version alongside it: a oneDNN upgrade silently changes the field order.
inline std::string serialize_device_info(const device_info& info,
                                         const std::string& onednn_version = "",
                                         const std::vector<uint8_t>& onednn_device_info = {}) {
    std::ostringstream os;
    os << "version=" << device_descriptor_version << "\n";
    os << "onednn_version=" << onednn_version << "\n";
    os << "onednn_device_info=" << detail::to_hex(onednn_device_info.data(), onednn_device_info.size()) << "\n";
#define OV_GPU_DD_NUM(field) os << #field "=" << static_cast<uint64_t>(info.field) << "\n";
#define OV_GPU_DD_STR(field) os << #field "=" << info.field << "\n";
    OV_GPU_DEVICE_DESCRIPTOR_FIELDS(OV_GPU_DD_NUM, OV_GPU_DD_STR)
#undef OV_GPU_DD_NUM
#undef OV_GPU_DD_STR
    os << "dev_type=" << static_cast<int>(info.dev_type) << "\n";
    os << "arch=" << static_cast<int>(info.arch) << "\n";
    os << "gfx_ver=" << info.gfx_ver.major << "," << static_cast<unsigned>(info.gfx_ver.minor) << ","
       << static_cast<unsigned>(info.gfx_ver.revision) << "\n";
    os << "pci_info=" << info.pci_info.pci_domain << "," << info.pci_info.pci_bus << ","
       << info.pci_info.pci_device << "," << info.pci_info.pci_function << "\n";
    os << "supported_simd_sizes=" << detail::join_sizes(info.supported_simd_sizes) << "\n";
    os << "cacheline_size=" << (info.cacheline_size ? std::to_string(*info.cacheline_size) : "") << "\n";
    os << "sub_buffer_base_alignment="
       << (info.sub_buffer_base_alignment ? std::to_string(*info.sub_buffer_base_alignment) : "") << "\n";
    os << "uuid=" << detail::to_hex(info.uuid.uuid.data(), info.uuid.uuid.size()) << "\n";
    os << "luid=" << detail::to_hex(info.luid.luid.data(), info.luid.luid.size()) << "\n";
    return os.str();
}

// Opaque oneDNN device_info blob captured during enumeration; empty when oneDNN was not available.
inline std::vector<uint8_t> deserialize_onednn_device_info(const std::string& descriptor) {
    auto kv = detail::parse_kv(descriptor);
    const auto& hex = kv["onednn_device_info"];
    std::vector<uint8_t> blob(hex.size() / 2);
    detail::from_hex(hex, blob.data(), blob.size());
    return blob;
}

inline std::string deserialize_onednn_version(const std::string& descriptor) {
    return detail::parse_kv(descriptor)["onednn_version"];
}

inline device_info deserialize_device_info(const std::string& descriptor) {
    auto kv = detail::parse_kv(descriptor);

    OPENVINO_ASSERT(kv.count("version") && kv["version"] == device_descriptor_version,
                    "[GPU] Unsupported device descriptor version '", kv["version"], "' (expected ",
                    device_descriptor_version, ")");

    auto num = [&](const char* k) -> uint64_t {
        auto it = kv.find(k);
        OPENVINO_ASSERT(it != kv.end(), "[GPU] Device descriptor is missing field '", k, "'");
        return std::stoull(it->second);
    };

    device_info info{};
#define OV_GPU_DD_NUM(field) info.field = static_cast<decltype(info.field)>(num(#field));
#define OV_GPU_DD_STR(field) info.field = kv[#field];
    OV_GPU_DEVICE_DESCRIPTOR_FIELDS(OV_GPU_DD_NUM, OV_GPU_DD_STR)
#undef OV_GPU_DD_NUM
#undef OV_GPU_DD_STR
    info.dev_type = static_cast<device_type>(num("dev_type"));
    info.arch = static_cast<gpu_arch>(num("arch"));

    auto gfx = detail::split_sizes(kv["gfx_ver"]);
    OPENVINO_ASSERT(gfx.size() == 3, "[GPU] Device descriptor has a malformed gfx_ver");
    info.gfx_ver = {static_cast<uint16_t>(gfx[0]), static_cast<uint8_t>(gfx[1]), static_cast<uint8_t>(gfx[2])};

    auto pci = detail::split_sizes(kv["pci_info"]);
    OPENVINO_ASSERT(pci.size() == 4, "[GPU] Device descriptor has a malformed pci_info");
    info.pci_info = {static_cast<uint32_t>(pci[0]), static_cast<uint32_t>(pci[1]),
                     static_cast<uint32_t>(pci[2]), static_cast<uint32_t>(pci[3])};

    info.supported_simd_sizes = detail::split_sizes(kv["supported_simd_sizes"]);
    if (!kv["cacheline_size"].empty())
        info.cacheline_size = static_cast<uint32_t>(std::stoul(kv["cacheline_size"]));
    if (!kv["sub_buffer_base_alignment"].empty())
        info.sub_buffer_base_alignment = static_cast<uint32_t>(std::stoul(kv["sub_buffer_base_alignment"]));
    detail::from_hex(kv["uuid"], info.uuid.uuid.data(), info.uuid.uuid.size());
    detail::from_hex(kv["luid"], info.luid.luid.data(), info.luid.luid.size());
    return info;
}

#undef OV_GPU_DEVICE_DESCRIPTOR_FIELDS

}  // namespace ocl
}  // namespace cldnn
