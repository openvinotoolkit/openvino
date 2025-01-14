// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <climits>
#include <algorithm>

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "primitive.hpp"
#include "transformations/convert_precision.hpp"

namespace cldnn {

struct weights_mem {
    std::shared_ptr<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>> shared_buf = nullptr;
    std::shared_ptr<ov::op::v0::Constant> transformed_constant = nullptr;

    const uint8_t* get_loaded_data() {
        if (transformed_constant) {
            return reinterpret_cast<const uint8_t*>(transformed_constant->get_data_ptr());
        }
        OPENVINO_ASSERT(shared_buf);
        return shared_buf->get_ptr<uint8_t>();
    }
};

struct weightless_cache_manager {
    void set_constant_info(size_t bin_offset,
                           size_t original_size,
                           ov::element::Type original_dtype,
                           ov::element::Type curr_dtype,
                           ov::Shape shape) {
        this->bin_offset = bin_offset;
        this->original_size = original_size;
        this->original_dtype = original_dtype;
        this->curr_dtype = curr_dtype;
        this->shape = shape;
        do_weightless_caching = true;

        if (original_dtype != curr_dtype) {
            do_precision_conversion = true;
        }
    }

    void invalidate() {
        do_weightless_caching = false;
    }

    void set_new_dtype(ov::element::Type curr_dtype) {
        this->curr_dtype = curr_dtype;
        do_precision_conversion = original_dtype != curr_dtype;
    }

    bool save(BinaryOutputBuffer& ob, size_t data_size) const {
        if (!do_weightless_caching) {
            ob << false;
            return false;
        }

        ob << true;
        ob << bin_offset;
        ob << do_precision_conversion;
        if (do_precision_conversion) {
            ob << original_size;
            ob << make_data(&original_dtype, sizeof(ov::element::Type));
            ob << make_data(&curr_dtype, sizeof(ov::element::Type));

            size_t num_dims = shape.size();
            ob << make_data(&num_dims, sizeof(size_t));
            ob << make_data(shape.data(), num_dims * sizeof(ov::Shape::value_type));
        }
        return true;
    }

    std::shared_ptr<weights_mem> load(BinaryInputBuffer& ib,
                                      std::shared_ptr<ov::MappedMemory> mapped_weights,
                                      size_t data_size) {
        ib >> do_weightless_caching;
        if (!do_weightless_caching) {
            return nullptr;
        }

        OPENVINO_ASSERT(mapped_weights != nullptr, "mmap object is null");

        ib >> bin_offset;
        ib >> do_precision_conversion;
        if (do_precision_conversion) {
            ib >> original_size;
            ib >> make_data(&original_dtype, sizeof(ov::element::Type));
            ib >> make_data(&curr_dtype, sizeof(ov::element::Type));

            size_t num_dims = 0;
            ib >> make_data(&num_dims, sizeof(size_t));
            shape.resize(num_dims);
            ib >> make_data(shape.data(), num_dims * sizeof(ov::Shape::value_type));
        } else {
            original_size = data_size;
        }

        auto mem_obj = std::make_shared<weights_mem>();
        mem_obj->shared_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
            mapped_weights->data() + bin_offset,
            original_size,
            mapped_weights);

        if (should_run_transformations()) {
            run_transformations(mem_obj);
        }
        return mem_obj;
    }

private:
    bool do_weightless_caching = false;
    bool do_precision_conversion = false;

    size_t bin_offset = SIZE_MAX;
    size_t original_size = SIZE_MAX;
    ov::element::Type original_dtype = ov::element::Type_t::undefined;
    ov::element::Type curr_dtype = ov::element::Type_t::undefined;
    ov::Shape shape;

    bool should_run_transformations() {
        return do_precision_conversion;
    }

    void run_transformations(std::shared_ptr<weights_mem> mem_obj) {
        auto orig_constant = std::make_shared<ov::op::v0::Constant>(original_dtype,
                                                                    shape,
                                                                    mem_obj->shared_buf->get_ptr(),
                                                                    mem_obj->shared_buf);

        ov::ParameterVector inputParams;
        ov::ResultVector results;
        results.push_back(std::make_shared<ov::op::v0::Result>(orig_constant->output(0)));
        auto model = std::make_shared<ov::Model>(results, inputParams, "aux");

        ov::pass::Manager manager("Plugin:GPU:weightless_cache_transformations");

        if (do_precision_conversion) {
            precisions_map fp_convert_precision_map = {
                {original_dtype, curr_dtype}};
            type_to_fuse_map empty_fuse_map = {};
            const bool keep_precision_sensitive_in_fp32 = false;
            const bool convert_input_output_precision = false;
            const bool store_original_precision_as_rt_attribute = true;
            manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                              empty_fuse_map,
                                                              keep_precision_sensitive_in_fp32,
                                                              convert_input_output_precision,
                                                              store_original_precision_as_rt_attribute);
        }

        manager.run_passes(model);
        const auto& ops = model->get_ops();
        auto it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
            return ov::op::util::is_constant(node);
        });
        OPENVINO_ASSERT(it != ops.end());
        mem_obj->transformed_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(*it);
        OPENVINO_ASSERT(mem_obj->transformed_constant->get_element_type() == curr_dtype);
    }
};

/// @brief Provides input data to topology.
/// @details This primitive allows to pass data which is known at topology creation.
/// For example, weights and biases for scoring networks.
/// @note Passing data at topology may improve network performance if data optimization is enabled.
struct data : public primitive_base<data> {
    CLDNN_DECLARE_PRIMITIVE(data)

    data() : primitive_base("", {}) {
        cache_info = std::make_shared<weightless_cache_manager>();
    }

    /// @brief Constructs data primitive.
    /// @param id This primitive id.
    /// @param mem @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    data(const primitive_id& id, memory::ptr mem) : primitive_base(id, {}), mem(std::move(mem)) {
        cache_info = std::make_shared<weightless_cache_manager>();
    }

    data(const primitive_id& id, memory::ptr mem, std::shared_ptr<weightless_cache_manager> cache_info)
        : primitive_base(id, {}),
          mem(std::move(mem)),
          cache_info(cache_info) {
        if (!cache_info) {
            this->cache_info = std::make_shared<weightless_cache_manager>();
        }
    }

    /// @brief @ref memory object which contains data.
    /// @note If memory is attached by memory::attach(), the attached buffer should be valid till network build.
    memory::ptr mem;

    std::shared_ptr<weightless_cache_manager> cache_info;

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

        bool do_weightless_caching = cache_info->save(ob, data_size);
        if (!do_weightless_caching) {
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

        auto mem_obj = cache_info->load(ib, mapped_weights, data_size);
        bool is_weightless_caching_enabled = mem_obj != nullptr;

        if (_allocation_type == allocation_type::usm_host || _allocation_type == allocation_type::usm_shared) {
            if (is_weightless_caching_enabled) {
                std::memcpy(reinterpret_cast<uint8_t*>(mem->buffer_ptr()), mem_obj->get_loaded_data(), data_size);
            } else {
                ib >> make_data(mem->buffer_ptr(), data_size);
            }
        } else {
            const size_t DATA_BLOCK_SIZE = 2 * 1024 * 1024;
            auto& strm = ib.get_engine().get_service_stream();
            if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                std::vector<uint8_t> _buf(data_size);
                if (is_weightless_caching_enabled) {
                    std::memcpy(reinterpret_cast<uint8_t*>(_buf.data()), mem_obj->get_loaded_data(), data_size);
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
                        if (is_weightless_caching_enabled) {
                            std::memcpy(reinterpret_cast<uint8_t*>(_buf1.data()),
                                        mem_obj->get_loaded_data() + dst_offset,
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
                        if (is_weightless_caching_enabled) {
                            std::memcpy(reinterpret_cast<uint8_t*>(_buf2.data()),
                                        mem_obj->get_loaded_data() + dst_offset,
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
