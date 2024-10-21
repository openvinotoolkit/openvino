// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <climits>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Provides input data to topology.
/// @details This primitive allows to pass data which is known at topology creation.
/// For example, weights and biases for scoring networks.
/// @note Passing data at topology may improve network performance if data optimization is enabled.
struct data : public primitive_base<data> {
    CLDNN_DECLARE_PRIMITIVE(data)

    data() : primitive_base("", {}) {}

    /// @brief Constructs data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    data(const primitive_id& id, memory::ptr mem)
        : primitive_base(id, {}), mem(std::move(mem)) {}

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    size_t original_size = SIZE_MAX;
    size_t bin_offset = SIZE_MAX;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<data>::save(ob);

        ob << mem->get_layout();

        const auto _allocation_type = mem->get_allocation_type();
        ob << make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = mem->size();
        ob << make_data(&data_size, sizeof(size_t));

        bool is_cache_without_weights = bin_offset != SIZE_MAX && data_size == original_size;

        if (is_cache_without_weights) {
            ob << true;
            ob << bin_offset;
        } else {
            ob << false;
            if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
                ob << make_data(mem->buffer_ptr(), data_size);
            } else {
                std::vector<uint8_t> _buf;
                _buf.resize(data_size);
                stream* strm = reinterpret_cast<stream*>(ob.get_stream());
                mem->copy_to(*strm, _buf.data());
                ob << make_data(_buf.data(), data_size);
            }
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<data>::load(ib);
    }

    void load_weights(BinaryInputBuffer& ib, std::shared_ptr<ov::MappedMemory> mapped_weights) {
        layout output_layout = layout();
        ib >> output_layout;

        allocation_type _allocation_type = allocation_type::unknown;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = 0;
        ib >> make_data(&data_size, sizeof(size_t));

        mem = ib.get_engine().allocate_memory(output_layout, _allocation_type, false);

        bool is_cache_without_weights;
        ib >> is_cache_without_weights;
        if (is_cache_without_weights && mapped_weights == nullptr) {
            OPENVINO_THROW("mmap object is null");
        }

        std::shared_ptr<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>> shared_buf;
        if (is_cache_without_weights) {
            ib >> bin_offset;
            original_size = data_size;

            shared_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mapped_weights->data() + bin_offset,
                data_size,
                mapped_weights);
        }

        if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
            if (is_cache_without_weights) {
                std::memcpy(reinterpret_cast<uint8_t*>(mem->buffer_ptr()), shared_buf->get_ptr<uint8_t>(), data_size);
            } else {
                ib >> make_data(mem->buffer_ptr(), data_size);
            }
        } else {
            const size_t DATA_BLOCK_SIZE = 2 * 1024 * 1024;
            auto& strm = ib.get_engine().get_service_stream();
            if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                std::vector<uint8_t> _buf(data_size);
                if (is_cache_without_weights) {
                    std::memcpy(reinterpret_cast<uint8_t*>(_buf.data()), shared_buf->get_ptr<uint8_t>(), data_size);
                } else {
                    ib >> make_data(_buf.data(), data_size);
                }
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
                        if (is_cache_without_weights) {
                            std::memcpy(reinterpret_cast<uint8_t*>(_buf1.data()),
                                        shared_buf->get_ptr<uint8_t>() + dst_offset,
                                        copy_size);
                        } else {
                            ib >> make_data(_buf1.data(), copy_size);
                        }
                        if (ev2 != nullptr) {
                            ev2->wait();
                            ev2 = nullptr;
                        }
                        ev1 = mem->copy_from(strm, _buf1.data(), src_offset, dst_offset, copy_size, is_blocking);
                    } else {
                        if (is_cache_without_weights) {
                            std::memcpy(reinterpret_cast<uint8_t*>(_buf2.data()),
                                        shared_buf->get_ptr<uint8_t>() + dst_offset,
                                        copy_size);
                        } else {
                            ib >> make_data(_buf2.data(), copy_size);
                        }
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
