// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vk_graph_format.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ov::core::vulkan::cross_platform {

namespace {

// Thrown on malformed blobs; keeps this module free of openvino core includes.
[[noreturn]] void format_error(const std::string& msg) {
    throw std::runtime_error("[GPU] vk_graph_format: " + msg);
}

[[nodiscard]] constexpr std::byte to_byte(uint8_t v) {
    return std::byte{v};
}
[[nodiscard]] constexpr uint8_t to_u8(std::byte b) {
    return std::to_integer<uint8_t>(b);
}

// ---- little-endian writer / reader ---------------------------------------

class fb_writer {
public:
    void put_u8(uint8_t v) { _buf.push_back(to_byte(v)); }
    void put_u32(uint32_t v) {
        for (int i = 0; i < 4; ++i)
            _buf.push_back(to_byte(static_cast<uint8_t>((v >> (8 * i)) & 0xff)));
    }
    void put_u64(uint64_t v) {
        for (int i = 0; i < 8; ++i)
            _buf.push_back(to_byte(static_cast<uint8_t>((v >> (8 * i)) & 0xff)));
    }
    void put_f32(float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        put_u32(bits);
    }
    void put_str(std::string_view s) {
        put_u32(static_cast<uint32_t>(s.size()));
        for (const char c : s)
            _buf.push_back(to_byte(static_cast<uint8_t>(c)));
    }
    void put_u64_vec(std::span<const size_t> v) {
        put_u32(static_cast<uint32_t>(v.size()));
        for (const size_t x : v)
            put_u64(static_cast<uint64_t>(x));
    }
    void put_bytes(std::span<const std::byte> data) {
        _buf.insert(_buf.end(), data.begin(), data.end());
    }
    void put_bytes(std::span<const char> data) {
        // std::byte has an explicit constructor; insert()'s construct_at cannot
        // convert char implicitly, so convert element-wise.
        for (const char c : data)
            _buf.push_back(to_byte(static_cast<uint8_t>(c)));
    }
    [[nodiscard]] std::vector<std::byte> take() && { return std::move(_buf); }

private:
    std::vector<std::byte> _buf;
};

class fb_reader {
public:
    explicit fb_reader(std::span<const std::byte> blob) : _blob(blob) {}

    uint8_t get_u8() {
        require(1);
        return to_u8(_blob[_pos++]);
    }
    uint32_t get_u32() {
        require(4);
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<uint32_t>(to_u8(_blob[_pos + i])) << (8 * i);
        _pos += 4;
        return v;
    }
    uint64_t get_u64() {
        require(8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= static_cast<uint64_t>(to_u8(_blob[_pos + i])) << (8 * i);
        _pos += 8;
        return v;
    }
    float get_f32() {
        const uint32_t bits = get_u32();
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    std::string get_str() {
        const uint32_t len = get_u32();
        require(len);
        std::string s{reinterpret_cast<const char*>(_blob.data() + _pos), len};
        _pos += len;
        return s;
    }
    std::vector<size_t> get_u64_vec() {
        const uint32_t count = get_u32();
        std::vector<size_t> v;
        v.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            v.push_back(static_cast<size_t>(get_u64()));
        return v;
    }
    void skip(size_t n) {
        require(n);
        _pos += n;
    }
    [[nodiscard]] size_t remaining() const { return _blob.size() - _pos; }
    [[nodiscard]] size_t pos() const { return _pos; }
    [[nodiscard]] std::span<const std::byte> rest() const { return _blob.subspan(_pos); }

private:
    void require(size_t n) const {
        if (_pos + n > _blob.size())
            format_error("truncated blob (need " + std::to_string(n) + " bytes at offset " + std::to_string(_pos) +
                         ", have " + std::to_string(_blob.size() - _pos) + ")");
    }

    std::span<const std::byte> _blob;
    size_t _pos = 0;
};

[[nodiscard]] constexpr uint8_t op_to_u8(ir_op op) {
    return static_cast<uint8_t>(op);
}
[[nodiscard]] ir_op u8_to_op(uint8_t v) {
    switch (static_cast<ir_op>(v)) {
        case ir_op::parameter:
        case ir_op::constant:
        case ir_op::result:
        case ir_op::relu:
        case ir_op::add:
        case ir_op::max_pool:
        case ir_op::avg_pool:
        case ir_op::convolution:
        case ir_op::matmul:
            return static_cast<ir_op>(v);
    }
    format_error("unknown op code " + std::to_string(v));
}

void write_fb_body(fb_writer& w, const ir_graph& graph) {
    w.put_u32(k_format_version);
    w.put_u32(static_cast<uint32_t>(graph.tensor_shapes.size()));
    w.put_u32(static_cast<uint32_t>(graph.constant_data.size()));
    w.put_u32(static_cast<uint32_t>(graph.nodes.size()));
    w.put_u32(static_cast<uint32_t>(graph.inputs.size()));
    w.put_u32(static_cast<uint32_t>(graph.outputs.size()));

    for (const auto& [id, shape] : graph.tensor_shapes) {
        w.put_str(id);
        w.put_u64_vec(shape);
    }
    for (const auto& [id, data] : graph.constant_data) {
        w.put_str(id);
        w.put_u32(static_cast<uint32_t>(data.size()));
        for (const float f : data)
            w.put_f32(f);
    }
    for (const auto& node : graph.nodes) {
        w.put_u8(op_to_u8(node.op));
        w.put_str(node.id);
        w.put_u8(node.matmul_transpose_b ? 1 : 0);
        w.put_u64_vec(node.pool.kernel);
        w.put_u64_vec(node.pool.strides);
        w.put_u64_vec(node.pool.pads_begin);
        w.put_u32(static_cast<uint32_t>(node.inputs.size()));
        for (const auto& in : node.inputs)
            w.put_str(in);
    }
    for (const auto& id : graph.inputs)
        w.put_str(id);
    for (const auto& id : graph.outputs)
        w.put_str(id);
}

void read_fb_body(fb_reader& r, ir_graph& out) {
    const uint32_t version = r.get_u32();
    if (version != k_format_version)
        format_error("unsupported FB version " + std::to_string(version));

    const uint32_t num_tensors = r.get_u32();
    const uint32_t num_constants = r.get_u32();
    const uint32_t num_nodes = r.get_u32();
    const uint32_t num_inputs = r.get_u32();
    const uint32_t num_outputs = r.get_u32();

    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Note: read both operands into locals first; in C++17 and later the
        // right-hand side of operator[]/operator= is sequenced before the
        // left-hand side, so `out.tensor_shapes[r.get_str()] = r.get_u64_vec()`
        // would read the vector before the key.
        const auto id = r.get_str();
        out.tensor_shapes[id] = r.get_u64_vec();
    }
    for (uint32_t i = 0; i < num_constants; ++i) {
        const auto id = r.get_str();
        const uint32_t count = r.get_u32();
        std::vector<float> data;
        data.reserve(count);
        for (uint32_t j = 0; j < count; ++j)
            data.push_back(r.get_f32());
        out.constant_data[id] = std::move(data);
    }
    for (uint32_t i = 0; i < num_nodes; ++i) {
        ir_node node;
        node.op = u8_to_op(r.get_u8());
        node.id = r.get_str();
        node.matmul_transpose_b = r.get_u8() != 0;
        node.pool.kernel = r.get_u64_vec();
        node.pool.strides = r.get_u64_vec();
        node.pool.pads_begin = r.get_u64_vec();
        const uint32_t n_inputs = r.get_u32();
        node.inputs.reserve(n_inputs);
        for (uint32_t j = 0; j < n_inputs; ++j)
            node.inputs.push_back(r.get_str());
        out.nodes.push_back(std::move(node));
    }
    out.inputs.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
        out.inputs.push_back(r.get_str());
    out.outputs.reserve(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i)
        out.outputs.push_back(r.get_str());
}

}  // namespace

std::vector<std::byte> serialize_fb(const ir_graph& graph) {
    fb_writer w;
    w.put_bytes(std::span<const char>{k_fb_magic.data(), k_fb_magic.size()});
    write_fb_body(w, graph);
    return std::move(w).take();
}

void deserialize_fb(std::span<const std::byte> blob, ir_graph& out) {
    if (blob.size() < k_fb_magic.size())
        format_error("blob too small for FB header");
    if (std::memcmp(blob.data(), k_fb_magic.data(), k_fb_magic.size()) != 0)
        format_error("not an FB blob (bad magic)");
    fb_reader r{blob.subspan(k_fb_magic.size())};
    read_fb_body(r, out);
    if (!r.rest().empty())
        format_error("trailing bytes after FB payload (" + std::to_string(r.remaining()) + ")");
}

std::vector<std::byte> serialize_pb(std::span<const ir_graph> graphs) {
    fb_writer w;
    w.put_bytes(std::span<const char>{k_pb_magic.data(), k_pb_magic.size()});
    w.put_u32(k_format_version);
    w.put_u32(static_cast<uint32_t>(graphs.size()));
    for (const auto& graph : graphs) {
        auto blob = serialize_fb(graph);
        w.put_u32(static_cast<uint32_t>(blob.size()));
        w.put_bytes(blob);
    }
    return std::move(w).take();
}

std::vector<ir_graph> deserialize_pb(std::span<const std::byte> blob) {
    if (blob.size() < k_pb_magic.size() + 8)
        format_error("blob too small for PB header");
    if (std::memcmp(blob.data(), k_pb_magic.data(), k_pb_magic.size()) != 0)
        format_error("not a PB blob (bad magic)");
    fb_reader r{blob.subspan(k_pb_magic.size())};
    const uint32_t version = r.get_u32();
    if (version != k_format_version)
        format_error("unsupported PB version " + std::to_string(version));
    const uint32_t count = r.get_u32();
    std::vector<ir_graph> graphs;
    graphs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t blob_size = r.get_u32();
        if (blob_size > r.remaining())
            format_error("PB graph " + std::to_string(i) + " size " + std::to_string(blob_size) +
                         " exceeds remaining " + std::to_string(r.remaining()));
        ir_graph graph;
        deserialize_fb(r.rest().first(blob_size), graph);
        r.skip(blob_size);
        graphs.push_back(std::move(graph));
    }
    return graphs;
}

}  // namespace ov::core::vulkan::cross_platform