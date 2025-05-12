// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "buffer.hpp"
#include "helpers.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace cldnn {

namespace {
bool is_alloc_host_accessible(const cldnn::allocation_type& alloc_type) {
    return alloc_type == cldnn::allocation_type::usm_host || alloc_type == cldnn::allocation_type::usm_shared;
}
}

template <typename BufferType>
class Serializer<BufferType, memory::ptr, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const memory::ptr& mem) {
        // copy from data.hpp
        buffer << mem->get_layout();

        const auto allocation_type = mem->get_allocation_type();
        buffer << make_data(&allocation_type, sizeof(allocation_type));

        size_t data_size = mem->size();
        buffer << make_data(&data_size, sizeof(size_t));

        if (is_alloc_host_accessible(allocation_type)) {
            buffer << make_data(mem->buffer_ptr(), data_size);
        } else {
            std::vector<uint8_t> _buf;
            _buf.resize(data_size);
            stream* strm = reinterpret_cast<stream*>(buffer.get_stream());
            mem->copy_to(*strm, _buf.data());
            buffer << make_data(_buf.data(), data_size);
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, memory::ptr, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, memory::ptr& mem) {
        layout output_layout = layout();
        buffer >> output_layout;

        allocation_type allocation_type = allocation_type::unknown;
        buffer >> make_data(&allocation_type, sizeof(allocation_type));

        size_t data_size = 0;
        buffer >> make_data(&data_size, sizeof(size_t));

        mem = buffer.get_engine().allocate_memory(output_layout, allocation_type, false);

        if (is_alloc_host_accessible(allocation_type)) {
            buffer >> make_data(mem->buffer_ptr(), data_size);
        } else {
            const size_t DATA_BLOCK_SIZE = 2 * 1024 * 1024;
            auto& strm = buffer.get_engine().get_service_stream();
            if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                std::vector<uint8_t> _buf(data_size);
                buffer >> make_data(_buf.data(), data_size);
                mem->copy_from(strm, _buf.data());
            } else {
                std::vector<uint8_t> _buf1(DATA_BLOCK_SIZE);
                std::vector<uint8_t> _buf2(DATA_BLOCK_SIZE);
                bool buf_flag = true;
                event::ptr ev1, ev2;
                ev1 = ev2 = nullptr;
                size_t dst_offset = 0;
                while (dst_offset < data_size) {
                    const bool is_blocking = false;
                    const size_t src_offset = 0;
                    size_t copy_size =
                        (data_size > (dst_offset + DATA_BLOCK_SIZE)) ? DATA_BLOCK_SIZE : (data_size - dst_offset);
                    if (buf_flag) {
                        buffer >> make_data(_buf1.data(), copy_size);
                        if (ev2 != nullptr) {
                            ev2->wait();
                            ev2 = nullptr;
                        }
                        ev1 = mem->copy_from(strm, _buf1.data(), src_offset, dst_offset, copy_size, is_blocking);
                    } else {
                        buffer >> make_data(_buf2.data(), copy_size);
                        if (ev1 != nullptr) {
                            ev1->wait();
                            ev1 = nullptr;
                        }
                        ev2 = mem->copy_from(strm, _buf2.data(), src_offset, dst_offset, copy_size, is_blocking);
                    }
                    dst_offset += DATA_BLOCK_SIZE;
                    buf_flag = !buf_flag;
                }
                if (ev2 != nullptr) {
                    ev2->wait();
                }
                if (ev1 != nullptr) {
                    ev1->wait();
                }
            }
        }
    }
};

}  // namespace cldnn
