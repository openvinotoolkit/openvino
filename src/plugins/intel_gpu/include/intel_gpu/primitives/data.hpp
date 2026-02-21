// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <variant>
#include <vector>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "primitive.hpp"
#include "transformations/convert_precision.hpp"

using weights_memory_ptr = std::variant<std::shared_ptr<ov::MappedMemory>, std::shared_ptr<const ov::Model>>;
using offset_const_map_t = std::map<size_t, std::shared_ptr<ov::op::v0::Constant>>;
using shared_mapped_memory_ptr = std::shared_ptr<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>;
using constant_memory_ptr = std::variant<shared_mapped_memory_ptr, std::shared_ptr<ov::op::v0::Constant>>;

namespace {

bool is_alloc_host_accessible(const cldnn::allocation_type& alloc_type) {
    return alloc_type == cldnn::allocation_type::usm_host || alloc_type == cldnn::allocation_type::usm_shared;
}

void copy_to_dst_mem(cldnn::memory::ptr mem_ptr, const uint8_t* data_ptr) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "copy_to_dst_mem");
    if (is_alloc_host_accessible(mem_ptr->get_allocation_type())) {
        size_t data_size = mem_ptr->size();
        std::memcpy(reinterpret_cast<uint8_t*>(mem_ptr->buffer_ptr()), data_ptr, data_size);
    } else {
        auto& strm = mem_ptr->get_engine()->get_service_stream();
        mem_ptr->copy_from(strm, data_ptr);
    }
}

}  // namespace

namespace cldnn {

class WeightsMemory {
public:
    WeightsMemory(std::shared_ptr<const ov::Model> model, std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr)
        : weights_memory(model) {
        fill_offset_to_constant_map(model, cache_attr_map);
    }

    WeightsMemory(std::shared_ptr<ov::MappedMemory> mapped_memory) : weights_memory(mapped_memory) {}

    constant_memory_ptr get_constant_buf(size_t bin_offset, size_t original_size) {
        if (std::holds_alternative<std::shared_ptr<ov::MappedMemory>>(weights_memory)) {
            auto mapped_memory = std::get<std::shared_ptr<ov::MappedMemory>>(weights_memory);
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + bin_offset, original_size, mapped_memory);
        } else {
            auto model_ptr = std::get<std::shared_ptr<const ov::Model>>(weights_memory);
            auto const_it = offset_to_constant_map.find(bin_offset);
            if (const_it == offset_to_constant_map.end()) {
                OPENVINO_THROW("Constant with bin_offset ", bin_offset, " not found in the model");
            }
            auto const_ptr = const_it->second;
            return const_ptr;
        }
    }

private:
    void fill_offset_to_constant_map(std::shared_ptr<const ov::Model> model, std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr) {
        const auto& ops = model->get_ops();

        if (cache_attr_map != nullptr && cache_attr_map->size() > 0) {
            for (const auto& node : ops) {
                if (ov::op::util::is_constant(node)) {
                    auto it = cache_attr_map->find(node->get_instance_id());
                    if (it != cache_attr_map->end()) {
                        auto attr = it->second;
                        auto const_ptr = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                        offset_to_constant_map.emplace(attr.bin_offset, const_ptr);
                    }
                }
            }
        } else {
            for (const auto& node : ops) {
                if (ov::op::util::is_constant(node)) {
                    auto rt_info = node->get_rt_info();
                    auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
                    if (weightless_cache_attr != rt_info.end()) {
                        auto& attr = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>();
                        auto const_ptr = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                        offset_to_constant_map.emplace(attr.bin_offset, const_ptr);
                    }
                } else if (auto ti = ov::as_type<const ov::op::v0::TensorIterator>(node.get())) {
                    auto ti_body = ti->get_body();
                    fill_offset_to_constant_map(ti_body);
                }
            }
        }
    }

    weights_memory_ptr weights_memory;
    offset_const_map_t offset_to_constant_map{};
};

struct reorder_replication {
    std::shared_ptr<cldnn::layout> input_layout = nullptr;
    std::shared_ptr<cldnn::reorder> reorder = nullptr;
};

struct weightless_cache_manager {
    void set_constant_info(size_t bin_offset, size_t original_size, ov::element::Type original_dtype, ov::element::Type curr_dtype, ov::Shape shape) {
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

    void apply_reorder(std::shared_ptr<layout> input_layout, std::shared_ptr<reorder> reorder) {
        reorder_rep = {input_layout, reorder};
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

        bool do_reorder = should_run_reorder();
        if (do_reorder) {
            ob << true;
            ob << *reorder_rep.input_layout;
            ob << *reorder_rep.reorder;
        } else {
            ob << false;
        }
        return true;
    }

    bool load(BinaryInputBuffer& ib, memory::ptr dst_mem, std::shared_ptr<WeightsMemory> weights_memory) {
        ib >> do_weightless_caching;
        if (!do_weightless_caching) {
            return false;
        }

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
            original_size = dst_mem->size();
        }

        bool do_reorder = false;
        ib >> do_reorder;
        if (do_reorder) {
            reorder_rep.input_layout = std::make_shared<layout>();
            ib >> *reorder_rep.input_layout;
            reorder_rep.reorder = std::make_shared<reorder>();
            ib >> *reorder_rep.reorder;
        }

        OPENVINO_ASSERT(weights_memory != nullptr, "weights_memory is nullptr!!!");
        auto constant_ptr = weights_memory->get_constant_buf(bin_offset, original_size);

        if (should_run_transformations()) {
            run_transformations(ib.get_engine(), dst_mem, constant_ptr);
        } else {
            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                auto cptr = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
                copy_to_dst_mem(dst_mem, reinterpret_cast<const uint8_t*>(cptr->get_data_ptr()));
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                copy_to_dst_mem(dst_mem, shared_buf->get_ptr<uint8_t>());
            }
        }
        return true;
    }

private:
    bool do_weightless_caching = false;
    bool do_precision_conversion = false;
    reorder_replication reorder_rep{};

    size_t bin_offset = SIZE_MAX;
    size_t original_size = SIZE_MAX;
    ov::element::Type original_dtype = ov::element::Type_t::dynamic;
    ov::element::Type curr_dtype = ov::element::Type_t::dynamic;
    ov::Shape shape{};

    bool should_run_reorder() const {
        return reorder_rep.reorder != nullptr;
    }

    bool should_run_transformations() {
        return do_precision_conversion || should_run_reorder();
    }

    void run_transformations(engine& engine, memory::ptr dst_mem, constant_memory_ptr constant_ptr) {
        std::shared_ptr<ov::op::v0::Constant> transformed_constant = nullptr;
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "weightless_cache_manager::run_transformations");

        // Note: this works only until the data is copied to dst_mem.
        auto get_intermediate_data = [&]() -> const uint8_t* {
            if (transformed_constant) {
                return reinterpret_cast<const uint8_t*>(transformed_constant->get_data_ptr());
            }

            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                auto cptr = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
                return reinterpret_cast<const uint8_t*>(cptr->get_data_ptr());
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                return shared_buf->get_ptr<uint8_t>();
            }
        };

        // Note: this works only until the data is copied to dst_mem.
        auto get_current_data_size = [&]() -> size_t {
            if (transformed_constant) {
                return transformed_constant->get_byte_size();
            }
            return original_size;
        };

        if (do_precision_conversion) {
            std::shared_ptr<ov::op::v0::Constant> orig_constant = nullptr;
            if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)) {
                orig_constant = std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr);
            } else {
                auto shared_buf = std::get<shared_mapped_memory_ptr>(constant_ptr);
                orig_constant = std::make_shared<ov::op::v0::Constant>(original_dtype, shape, get_intermediate_data(), shared_buf);
            }

            ov::ParameterVector inputParams;
            ov::ResultVector results;
            ov::pass::Manager manager("Plugin:GPU:weightless_cache_transformations");
            std::shared_ptr<ov::Model> model = nullptr;

            auto convert_op = std::make_shared<ov::op::v0::Convert>(orig_constant, curr_dtype);
            results.push_back(std::make_shared<ov::op::v0::Result>(convert_op->output(0)));
            model = std::make_shared<ov::Model>(results, inputParams, "aux");
            manager.register_pass<ov::pass::ConstantFolding>();

            manager.run_passes(model);
            const auto& ops = model->get_ops();
            auto it = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
                return ov::op::util::is_constant(node);
            });
            OPENVINO_ASSERT(it != ops.end());
            transformed_constant = ov::as_type_ptr<ov::op::v0::Constant>(*it);
            OPENVINO_ASSERT(transformed_constant->get_element_type() == curr_dtype);
        }

        if (should_run_reorder()) {
            const auto allocation_type = dst_mem->get_allocation_type();
            memory::ptr input_mem = engine.allocate_memory(*reorder_rep.input_layout, allocation_type, false);

            if (is_alloc_host_accessible(allocation_type)) {
                std::memcpy(reinterpret_cast<uint8_t*>(input_mem->buffer_ptr()), get_intermediate_data(), get_current_data_size());
            } else {
                auto& strm = engine.get_service_stream();
                input_mem->copy_from(strm, get_intermediate_data());
            }

            reorder_rep.reorder->input = {input_info("input")};
            topology topology(input_layout("input", *reorder_rep.input_layout), *reorder_rep.reorder);

            ExecutionConfig config;
            config.set_property(ov::intel_gpu::optimize_data(false));
            cldnn::network network(engine, topology, config, true);
            network.set_input_data("input", input_mem);
            network.set_output_memory(reorder_rep.reorder->id, dst_mem);
            auto outputs = network.execute();
            for (const auto& output : outputs) {
                auto ev = output.second.get_event();
                if (ev) {
                    ev->wait();
                }
            }

            OPENVINO_ASSERT(outputs.size() == 1);
        } else {
            copy_to_dst_mem(dst_mem, get_intermediate_data());
        }
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
            if (is_alloc_host_accessible(_allocation_type)) {
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

    void load_weights(BinaryInputBuffer& ib, std::shared_ptr<WeightsMemory> weights_memory, const std::string& weights_path = "") {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "load_weights");
        layout output_layout = layout();
        ib >> output_layout;

        allocation_type _allocation_type = allocation_type::unknown;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));
        // std::cout << "load weights: allocation_type = " << static_cast<int>(_allocation_type) << ", weights_path = " << weights_path << std::endl;

        size_t data_size = 0;
        ib >> make_data(&data_size, sizeof(size_t));

        mem = ib.get_engine().allocate_memory(output_layout, _allocation_type, false);

        bool is_weightless_caching = cache_info->load(ib, mem, weights_memory);

        if (!is_weightless_caching) {
            const size_t DATA_BLOCK_SIZE = 4 * 1024 * 1024;
            const size_t FAST_IO_THRESHOLD = 4 * 1024 * 1024;  // Use FAST IO for >4MB

            if (is_alloc_host_accessible(_allocation_type)) {
                bool used_fast_io = false;
                if (!used_fast_io && !weights_path.empty()) {
                    if (data_size >= FAST_IO_THRESHOLD) {
                        auto cur_offset = ib.get_stream().tellg();

                        // Auto-detect header offset compensation for path-based loading
                        // This applies to both Windows and Linux Parallel loaders which open by path
                        size_t offset_compensation = 0;

                        // Save current position
                        auto restore_pos = ib.get_stream().tellg();
                        ib.get_stream().seekg(0, std::ios::end);
                        auto stream_end = (size_t)ib.get_stream().tellg();
                        ib.get_stream().seekg(restore_pos, std::ios::beg);

                        int64_t phys_size = ov::util::file_size(ov::util::make_path(weights_path));
                        size_t physical_size = (phys_size >= 0) ? static_cast<size_t>(phys_size) : 0;

                        if (physical_size > stream_end) {
                            offset_compensation = physical_size - stream_end;
                        }

                        used_fast_io = ov::util::read_binary_file_parallel(ov::util::make_path(weights_path),
                                                                             mem->buffer_ptr(),
                                                                             data_size,
                                                                             (size_t)cur_offset + offset_compensation);
                        if (used_fast_io) {
                            ib.get_stream().seekg(data_size, std::ios::cur);
                        }
                    }
                }

                if (!used_fast_io) {
                    ib >> make_data(mem->buffer_ptr(), data_size);
                }
            } else {
                auto& strm = ib.get_engine().get_service_stream();
                if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                    std::vector<uint8_t> _buf(data_size);
                    ib >> make_data(_buf.data(), data_size);
                    mem->copy_from(strm, _buf.data());
                } else {
                    bool used_fast_io = false;

                    // Try parallel file read + pipelined async GPU copy for non-host-accessible memory
                    if (!weights_path.empty() && data_size >= FAST_IO_THRESHOLD) {
                        auto cur_offset = ib.get_stream().tellg();

                        // Auto-detect header offset compensation (same as host-accessible path)
                        size_t offset_compensation = 0;
                        auto restore_pos = ib.get_stream().tellg();
                        ib.get_stream().seekg(0, std::ios::end);
                        auto stream_end = (size_t)ib.get_stream().tellg();
                        ib.get_stream().seekg(restore_pos, std::ios::beg);

                        int64_t phys_size = ov::util::file_size(ov::util::make_path(weights_path));
                        size_t physical_size = (phys_size >= 0) ? static_cast<size_t>(phys_size) : 0;
                        if (physical_size > stream_end) {
                            offset_compensation = physical_size - stream_end;
                        }

                        size_t file_base_offset = (size_t)cur_offset + offset_compensation;
                        auto file_path = ov::util::make_path(weights_path);

                        // Pipelined: parallel file read into double-buffered CPU staging + async GPU copy
                        // This overlaps reading chunk N+1 from disk with GPU DMA of chunk N
                        const size_t PIPE_CHUNK_SIZE = 32 * 1024 * 1024;  // 32MB per pipeline stage
                        size_t actual_chunk = std::min(PIPE_CHUNK_SIZE, data_size);
                        std::vector<uint8_t> staging_a(actual_chunk);
                        std::vector<uint8_t> staging_b(actual_chunk);

                        size_t num_blocks = (data_size + PIPE_CHUNK_SIZE - 1) / PIPE_CHUNK_SIZE;
                        size_t first_chunk = std::min(PIPE_CHUNK_SIZE, data_size);

                        // Use ITaskExecutor to dispatch file read tasks instead of std::async
                        auto io_executor_cfg = ov::threading::IStreamsExecutor::Config{"GPUWeightLoadIO", 1};
                        auto io_executor = ov::threading::executor_manager()->get_idle_cpu_streams_executor(io_executor_cfg);

                        // Helper: submit a file read task via executor, return a future<bool>
                        auto submit_read = [&io_executor](const std::filesystem::path& path, uint8_t* buf,
                                                          size_t size, size_t offset) {
                            auto promise = std::make_shared<std::promise<bool>>();
                            auto future = promise->get_future();
                            io_executor->run([promise, path, buf, size, offset]() {
                                try {
                                    promise->set_value(
                                        ov::util::read_binary_file_parallel(path, buf, size, offset));
                                } catch (...) {
                                    promise->set_exception(std::current_exception());
                                }
                            });
                            return future;
                        };

                        // Kick off first parallel file read
                        auto read_future = submit_read(file_path, staging_a.data(),
                                                       first_chunk, file_base_offset);

                        event::ptr gpu_ev = nullptr;
                        size_t dst_offset = 0;
                        bool cur_is_a = true;
                        used_fast_io = true;

                        for (size_t block = 0; block < num_blocks && used_fast_io; block++) {
                            size_t chunk_size = std::min(PIPE_CHUNK_SIZE, data_size - dst_offset);
                            uint8_t* cur_buf = cur_is_a ? staging_a.data() : staging_b.data();

                            // Wait for current chunk's file read to complete
                            if (!read_future.get()) {
                                used_fast_io = false;
                                break;
                            }

                            // Start next chunk's parallel file read into the OTHER staging buffer
                            // This overlaps with the GPU copy of the current chunk below
                            if (block + 1 < num_blocks) {
                                size_t next_offset = dst_offset + chunk_size;
                                size_t next_chunk = std::min(PIPE_CHUNK_SIZE, data_size - next_offset);
                                uint8_t* next_buf = cur_is_a ? staging_b.data() : staging_a.data();
                                size_t next_file_off = file_base_offset + next_offset;
                                read_future = submit_read(file_path, next_buf,
                                                          next_chunk, next_file_off);
                            }

                            // Wait for previous GPU copy to complete before reusing its staging buffer
                            if (gpu_ev != nullptr) {
                                gpu_ev->wait();
                                gpu_ev = nullptr;
                            }

                            // Issue async GPU copy from current staging buffer
                            gpu_ev = mem->copy_from(strm, cur_buf, 0, dst_offset, chunk_size, false);

                            dst_offset += chunk_size;
                            cur_is_a = !cur_is_a;
                        }

                        if (gpu_ev != nullptr) {
                            gpu_ev->wait();
                        }

                        if (used_fast_io) {
                            ib.get_stream().seekg(data_size, std::ios::cur);
                        }
                    }

                    if (!used_fast_io) {
                        // Fallback: double-buffered sequential stream read + async GPU copy
                        std::vector<uint8_t> _buf1(DATA_BLOCK_SIZE);
                        std::vector<uint8_t> _buf2(DATA_BLOCK_SIZE);
                        bool buf_flag = true;
                        event::ptr ev1, ev2;
                        ev1 = ev2 = nullptr;
                        size_t dst_offset = 0;
                        while (dst_offset < data_size) {
                            const bool is_blocking = false;
                            const size_t src_offset = 0;
                            size_t copy_size = (data_size > (dst_offset + DATA_BLOCK_SIZE)) ? DATA_BLOCK_SIZE : (data_size - dst_offset);
                            if (buf_flag) {
                                ib >> make_data(_buf1.data(), copy_size);
                                if (ev2 != nullptr) {
                                    ev2->wait();
                                    ev2 = nullptr;
                                }
                                ev1 = mem->copy_from(strm, _buf1.data(), src_offset, dst_offset, copy_size, is_blocking);
                            } else {
                                ib >> make_data(_buf2.data(), copy_size);
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
        }
    }
};
}  // namespace cldnn
