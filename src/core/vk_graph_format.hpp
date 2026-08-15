// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_graph_format: binary serialization of the plugin-agnostic Vulkan IR.
//
// FB (Float Binary) v1: one ir_graph, f32 payloads. Self-describing,
// little-endian, bounds-checked so truncated/corrupt blobs fail loudly.
//
// PB (Parallel Binary) v1: container of several FB graphs in one blob (parallel
// branches, per-device/per-rank graphs). The header carries the graph count;
// each entry is a full FB blob.
//
// The module has no openvino core dependency: only vk_ir.hpp. It is the entry
// point for loading models without ov::Model (the OV plugin uses FB as its
// compiled-blob format; VkModelConverter stays a pure OV adapter).
//
// Written for C++23: blobs are std::span<const std::byte>, ids travel as
// std::string_view, and the magic/version constants are compile-time.

#pragma once

#include "vk_ir.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ov::core::vulkan::cross_platform {

// FB magic, 8 bytes: "VKFB0001". PB magic: "VKPB0001".
inline constexpr std::array<char, 8> k_fb_magic{'V', 'K', 'F', 'B', '0', '0', '0', '1'};
inline constexpr std::array<char, 8> k_pb_magic{'V', 'K', 'P', 'B', '0', '0', '0', '1'};
inline constexpr uint32_t k_format_version = 1;

// Serializes a single graph (nodes, tensors, constants, ports) to FB bytes.
[[nodiscard]] std::vector<std::byte> serialize_fb(const ir_graph& graph);

// Deserializes FB bytes into |out|. Throws std::runtime_error on any
// malformed input (bad magic, truncation, trailing garbage, unknown op).
void deserialize_fb(std::span<const std::byte> blob, ir_graph& out);

// PB: container of graphs. Serializes in order; deserialize_pb returns the
// same order and count.
[[nodiscard]] std::vector<std::byte> serialize_pb(std::span<const ir_graph> graphs);
[[nodiscard]] std::vector<ir_graph> deserialize_pb(std::span<const std::byte> blob);

}  // namespace ov::core::vulkan::cross_platform