// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#include "openvino/runtime/internal_properties.hpp"
#include "storage_traits.hpp"

namespace ov {
struct SharedContextStreamCodec {
    ov::SharedContext* ctx;

    friend std::istream& operator>>(std::istream& stream, SharedContextStreamCodec& cache) {
        storage::tag_type tag{};
        do {
            storage::length_type ctx_size{};
            stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
            if (!stream.good()) {
                break;
            }
            stream.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
            if (!stream.good() || ctx_size == 0) {
                break;
            }
            if (tag == storage::shared_context_tag) {
                const auto end_pos = stream.tellg() + static_cast<std::streamoff>(ctx_size);
                do {
                    size_t id, const_id, offset, byte_size;
                    stream.read(reinterpret_cast<char*>(&id), sizeof(id));
                    stream.read(reinterpret_cast<char*>(&const_id), sizeof(const_id));
                    stream.read(reinterpret_cast<char*>(&offset), sizeof(offset));
                    stream.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size));
                    if (auto id_it = cache.ctx->find(id); id_it != cache.ctx->end()) {
                        id_it->second[const_id] = std::make_tuple(offset, byte_size);
                    } else {
                        (*cache.ctx)[id] = {{const_id, std::make_tuple(offset, byte_size)}};
                    }
                } while (stream.good() && stream.tellg() < end_pos);
            } else {
                stream.seekg(ctx_size ? ctx_size : 1, std::ios::cur);
            }
        } while (stream.good());

        return stream;
    }

    friend std::ostream& operator<<(std::ostream& stream, const SharedContextStreamCodec& cache) {
        if (cache.ctx == nullptr || cache.ctx->empty()) {
            return stream;
        }
        stream.write(reinterpret_cast<const char*>(&storage::shared_context_tag), sizeof(storage::shared_context_tag));
        storage::length_type ctx_size = 0;
        const auto size_offset = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
        for (const auto& [id, consts] : *cache.ctx) {
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

}  // namespace ov
