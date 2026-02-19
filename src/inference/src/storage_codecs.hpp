// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/tlv_format.hpp"
#include "storage_traits.hpp"

namespace ov {

struct SharedContextStreamCodec {
    SharedContext* ctx;

    friend std::istream& operator>>(std::istream& stream, SharedContextStreamCodec& codec) {
        if (!codec.ctx) {
            return stream;
        }
        TLVStorage::Tag tag{};
        do {
            TLVFormat::length_type ctx_size{};
            stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            if (!stream.good()) {
                break;
            }
            stream.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
            if (!stream.good() || ctx_size == 0) {
                break;
            }
            if (tag == TLVStorage::Tag::SharedContext) {
                const auto end_pos = stream.tellg() + static_cast<std::streamoff>(ctx_size);
                do {
                    size_t id, const_id, offset, byte_size;
                    stream.read(reinterpret_cast<char*>(&id), sizeof(id));
                    stream.read(reinterpret_cast<char*>(&const_id), sizeof(const_id));
                    stream.read(reinterpret_cast<char*>(&offset), sizeof(offset));
                    stream.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
                    if (auto id_it = codec.ctx->find(id); id_it != codec.ctx->end()) {
                        id_it->second[const_id] = std::make_tuple(offset, byte_size);
                    } else {
                        (*codec.ctx)[id] = {{const_id, std::make_tuple(offset, byte_size)}};
                    }
                } while (stream.good() && stream.tellg() < end_pos);
            } else {
                stream.seekg(ctx_size ? ctx_size : 1, std::ios::cur);
            }
        } while (stream.good());

        return stream;
    }

    friend std::ostream& operator<<(std::ostream& stream, const SharedContextStreamCodec& codec) {
        if (!codec.ctx || codec.ctx->empty()) {
            return stream;
        }
        constexpr auto sc_tag = TLVStorage::Tag::SharedContext;
        stream.write(reinterpret_cast<const char*>(&sc_tag), sizeof(sc_tag));
        TLVFormat::length_type ctx_size = 0;
        const auto size_offset = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
        for (const auto& [id, consts] : *codec.ctx) {
            for (const auto& [const_id, props] : consts) {
                const auto& [offset, size] = props;
                stream.write(reinterpret_cast<const char*>(&id), sizeof(id));
                stream.write(reinterpret_cast<const char*>(&const_id), sizeof(const_id));
                stream.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
                stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                ctx_size += sizeof(id) + sizeof(const_id) + sizeof(size) + sizeof(offset);
            }
        }
        const auto end_pos = stream.tellp();
        stream.seekp(size_offset);
        stream.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
        stream.seekp(end_pos);
        return stream;
    }
};

namespace {}  // namespace
}  // namespace ov
