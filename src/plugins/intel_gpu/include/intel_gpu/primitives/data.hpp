// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <algorithm>
#include <climits>
#include <variant>
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/polymorphic_serializer.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
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
    if (is_alloc_host_accessible(mem_ptr->get_allocation_type())) {
        size_t data_size = mem_ptr->size();
        std::memcpy(reinterpret_cast<uint8_t*>(mem_ptr->buffer_ptr()),
                    data_ptr,
                    data_size);
    } else {
        auto& strm = mem_ptr->get_engine()->get_service_stream();
        mem_ptr->copy_from(strm, data_ptr);
    }
}

}  // namespace

namespace cldnn {

class WeightsMemory {
public:
    WeightsMemory(std::shared_ptr<const ov::Model> model,
                  std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr) : weights_memory(model) {
        fill_offset_to_constant_map(model, cache_attr_map);
    }

    WeightsMemory(std::shared_ptr<ov::MappedMemory> mapped_memory) : weights_memory(mapped_memory) {}

    constant_memory_ptr get_constant_buf(size_t bin_offset, size_t original_size) {
        if (std::holds_alternative<std::shared_ptr<ov::MappedMemory>>(weights_memory)) {
            auto mapped_memory = std::get<std::shared_ptr<ov::MappedMemory>>(weights_memory);
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mapped_memory->data() + bin_offset,
                original_size,
                mapped_memory);
        }
        auto model_ptr = std::get<std::shared_ptr<const ov::Model>>(weights_memory);
        auto const_it = offset_to_constant_map.find(bin_offset);
        if (const_it == offset_to_constant_map.end()) {
            OPENVINO_THROW("Constant with bin_offset ", bin_offset, " not found in the model");
        }
            auto const_ptr = const_it->second;
            return const_ptr;
    }

private:
    void fill_offset_to_constant_map(std::shared_ptr<const ov::Model> model,
                                     std::shared_ptr<ov::intel_gpu::GpuWeightlessCacheMap> cache_attr_map = nullptr) {
        const auto& ops = model->get_ops();

        if (cache_attr_map != nullptr && !cache_attr_map->empty()) {
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
                } else if (const auto* ti = ov::as_type<const ov::op::v0::TensorIterator>(node.get())) {
                    auto ti_body = ti->get_body();
                    fill_offset_to_constant_map(ti_body);
                }
            }
        }
    }

    weights_memory_ptr weights_memory;
    offset_const_map_t offset_to_constant_map;
};

struct reorder_replication {
    std::shared_ptr<cldnn::layout> input_layout = nullptr;
    std::shared_ptr<cldnn::reorder> reorder = nullptr;
};

// A constant sub-graph kept as a recipe: leaf constants plus the primitives that turn them into the
// folded value. Replayed at load time, which is where the GPU is.
struct const_chain {
    struct leaf {
        primitive_id id;
        cldnn::layout lay;
        std::vector<uint8_t> bytes;
    };
    std::vector<leaf> leaves;
    std::vector<std::shared_ptr<primitive>> prims;
    // Graph edges captured per prim: a primitive's own `input` can be stale after passes rewired the
    // node, so the dependencies are what the replayed topology must be built from.
    std::vector<std::vector<input_info>> prim_inputs;
    primitive_id output_id;

    bool empty() const { return prims.empty(); }
};

// Source the constant is rebuilt from: `original_model` needs the source model at load time,
// `chain` is self-contained and covers constants created by transformations.
enum class const_source : uint8_t {
    original_model = 0,
    chain = 1,     // leaves carry their own bytes; see const_chain
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

    void apply_reorder(std::shared_ptr<layout> input_layout, std::shared_ptr<reorder> reorder) {
        reorder_rep = {input_layout, reorder};
    }

    // Multi-primitive counterpart of apply_reorder: the whole constant sub-graph travels in the blob.
    void apply_chain(const_chain chain) { this->chain = std::move(chain); }

    // Drop the original-model recipe once the constant is modified in place, so the current value is serialized.
    void invalidate() { do_weightless_caching = false; }

    // True when this manager owns the value, so data::save must not write the (possibly empty) buffer.
    bool is_managed() const { return do_weightless_caching || !chain.empty(); }

    bool save(BinaryOutputBuffer& ob, size_t data_size) const {
        if (!is_managed()) {
            ob << false;
            return false;
        }

        ob << true;
        const auto source = !chain.empty() ? const_source::chain : const_source::original_model;
        ob << make_data(&source, sizeof(source));
        if (source == const_source::chain) {
            ob << chain.leaves.size();
            for (const auto& l : chain.leaves) {
                ob << l.id;
                ob << l.lay;
                ob << l.bytes.size();
                ob << make_data(l.bytes.data(), l.bytes.size());
            }
            ob << chain.prims.size();
            for (size_t i = 0; i < chain.prims.size(); i++) {
                ob << chain.prims[i];
                ob << chain.prim_inputs[i].size();
                for (const auto& in : chain.prim_inputs[i]) {
                    ob << in.pid;
                    ob << in.idx;
                }
            }
            ob << chain.output_id;
        } else {
            ob << bin_offset;
        }
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

    bool load(BinaryInputBuffer& ib, memory::ptr dst_mem, std::shared_ptr<WeightsMemory> weights_memory, bool weightless_caching) {
        if (!weightless_caching) {
            do_weightless_caching = false;
            return false;
        }

        const_source source = const_source::original_model;
        ib >> make_data(&source, sizeof(source));
        do_weightless_caching = (source == const_source::original_model);
        if (source == const_source::chain) {
            size_t n = 0;
            ib >> n;
            chain.leaves.resize(n);
            for (auto& l : chain.leaves) {
                size_t len = 0;
                ib >> l.id;
                ib >> l.lay;
                ib >> len;
                l.bytes.resize(len);
                ib >> make_data(l.bytes.data(), len);
            }
            ib >> n;
            chain.prims.resize(n);
            chain.prim_inputs.resize(n);
            for (size_t i = 0; i < n; i++) {
                size_t num_in = 0;
                ib >> chain.prims[i];
                ib >> num_in;
                chain.prim_inputs[i].resize(num_in);
                for (auto& in : chain.prim_inputs[i]) {
                    ib >> in.pid;
                    ib >> in.idx;
                }
            }
            ib >> chain.output_id;
        } else {
            ib >> bin_offset;
        }
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
            // dst_mem carries the reordered layout, which may be padded. What we read back from the
            // original weights is the reorder input, so its size is the one to use.
            if (!do_precision_conversion)
                original_size = reorder_rep.input_layout->bytes_count();
        }

        constant_memory_ptr constant_ptr;
        if (do_weightless_caching) {
            OPENVINO_ASSERT(weights_memory != nullptr, "weights_memory is nullptr!!!");
            constant_ptr = weights_memory->get_constant_buf(bin_offset, original_size);
        }

        if (should_run_transformations()) {
            run_transformations(ib.get_engine(), dst_mem, constant_ptr);
        } else {
            copy_to_dst_mem(dst_mem, source_data(constant_ptr));
        }
        return true;
    }


private:
    bool do_weightless_caching = false;
    bool do_precision_conversion = false;
    reorder_replication reorder_rep{};

    const_chain chain;                     // multi-primitive constant sub-graph

    size_t bin_offset = SIZE_MAX;
    size_t original_size = SIZE_MAX;
    ov::element::Type original_dtype = ov::element::Type_t::dynamic;
    ov::element::Type curr_dtype = ov::element::Type_t::dynamic;
    ov::Shape shape;

    // Start of the value the recipe transforms, whichever source it came from.
    const uint8_t* source_data(const constant_memory_ptr& constant_ptr) const {
        if (std::holds_alternative<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr))
            return reinterpret_cast<const uint8_t*>(
                std::get<std::shared_ptr<ov::op::v0::Constant>>(constant_ptr)->get_data_ptr());
        return std::get<shared_mapped_memory_ptr>(constant_ptr)->get_ptr<uint8_t>();
    }

    bool should_run_reorder() const {
        return reorder_rep.reorder != nullptr;
    }

    bool should_run_transformations() {
        return do_precision_conversion || should_run_reorder() || !chain.empty();
    }

    // Rebuilds the constant sub-graph as a throwaway network and writes its output into dst_mem.
    void run_chain(engine& engine, memory::ptr dst_mem) {
        const auto alloc_type = dst_mem->get_allocation_type();
        topology t;
        std::map<primitive_id, memory::ptr> inputs;
        for (const auto& l : chain.leaves) {
            auto mem = engine.allocate_memory(l.lay, alloc_type, false);
            copy_to_dst_mem(mem, l.bytes.data());
            t.add(input_layout(l.id, l.lay));
            inputs[l.id] = mem;
        }
        for (size_t i = 0; i < chain.prims.size(); i++) {
            chain.prims[i]->input = chain.prim_inputs[i];
            t.add_primitive(chain.prims[i]);
        }

        ExecutionConfig config;
        config.set_property(ov::intel_gpu::optimize_data(false));
        cldnn::network network(engine, t, config, true);
        for (const auto& [id, mem] : inputs)
            network.set_input_data(id, mem);
        network.set_output_memory(chain.output_id, dst_mem);
        for (const auto& output : network.execute()) {
            if (auto ev = output.second.get_event())
                ev->wait();
        }
    }

    void run_transformations(engine& engine,
                             memory::ptr dst_mem,
                             constant_memory_ptr constant_ptr) {
        if (!chain.empty()) {
            run_chain(engine, dst_mem);
            return;
        }
        std::shared_ptr<ov::op::v0::Constant> transformed_constant = nullptr;

        // Note: this works only until the data is copied to dst_mem.
        auto get_intermediate_data = [&]() -> const uint8_t* {
            if (transformed_constant) {
                return reinterpret_cast<const uint8_t*>(transformed_constant->get_data_ptr());
            }
            return source_data(constant_ptr);
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
                orig_constant =
                    std::make_shared<ov::op::v0::Constant>(original_dtype, shape, get_intermediate_data(), shared_buf);
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
                std::memcpy(reinterpret_cast<uint8_t*>(input_mem->buffer_ptr()),
                            get_intermediate_data(),
                            get_current_data_size());
            } else {
                auto& strm = engine.get_service_stream();
                input_mem->copy_from(strm, get_intermediate_data());
            }

            reorder_rep.reorder->input = {input_info("input")};
            topology topology(input_layout("input", *reorder_rep.input_layout),
                              *reorder_rep.reorder);

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
    data(const primitive_id& id, memory::ptr mem, bool skip_device_transfer = false)
        : primitive_base(id, {}), mem(std::move(mem)), _skip_device_transfer(skip_device_transfer) {
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

    /// @brief Whether transfer_memory_to_device skips this node (e.g. OTD partial upload).
    bool skip_device_transfer() const { return _skip_device_transfer; }

private:
    bool _skip_device_transfer = false;

public:

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
        const auto& engine = mem->get_engine();
        const auto& dev_info = engine->get_device_info();
        if (!do_weightless_caching) {
            if (engine->can_use_host_usm_zero_copy() && !ob.is_encrypted()) {
                if (const auto pad = ov::util::align_padding_size(dev_info.sub_buffer_base_alignment.value_or(0), ob.get_offset()); pad > 0) {
                    std::vector<uint8_t> zeros(pad, 0);
                    ob << make_data(zeros.data(), zeros.size());
                }
            }
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

    void load_weights(BinaryInputBuffer& ib, std::shared_ptr<WeightsMemory> weights_memory, memory_ptr host_buffer_base_ptr) {
        layout output_layout = layout();
        ib >> output_layout;

        allocation_type _allocation_type = allocation_type::unknown;
        ib >> make_data(&_allocation_type, sizeof(_allocation_type));

        size_t data_size = 0;
        ib >> make_data(&data_size, sizeof(size_t));

        OPENVINO_ASSERT(data_size == output_layout.bytes_count(),
               "[GPU] Corrupt cache blob: data_size=", data_size,
               ", expected=", output_layout.bytes_count());

        bool weightless_caching = false;
        ib >> weightless_caching;
        const auto& dev_info = ib.get_engine().get_device_info();
        const bool can_use_zero_copy_mode = ib.get_engine().can_use_host_usm_zero_copy() && ib.is_tensor_aligned(ov::util::min_page_alignment) &&
                                            host_buffer_base_ptr != nullptr && !weightless_caching;
        if (!can_use_zero_copy_mode) {
            mem = ib.get_engine().allocate_memory(output_layout, _allocation_type, false);
        }

        bool is_weightless_caching = cache_info->load(ib, mem, weights_memory, weightless_caching);

        if (!is_weightless_caching) {
            if (ib.get_engine().can_use_host_usm_zero_copy() && !ib.is_encrypted()) {
                // Advance input stream past padding so subbuffer base offset meets use_host_ptr() alignment requirements.
                if (const auto pad = ov::util::align_padding_size(dev_info.sub_buffer_base_alignment.value_or(0), ib.get_offset()); pad > 0) {
                    ib.seek_current_ptr(pad);
                }
            }
            // Zero-copy: create subbuffer referencing mmap'd cache without host-to-device transfer.
            if (can_use_zero_copy_mode) {
                mem = ib.get_engine().create_subbuffer(*host_buffer_base_ptr, output_layout, ib.get_offset());
                ib.seek_current_ptr(data_size);
            } else if (is_alloc_host_accessible(_allocation_type)) {
                ib >> make_data(std::move(mem->buffer_ptr()), data_size);
            } else {
                const size_t DATA_BLOCK_SIZE = 4 * 1024 * 1024;
                auto& eng = ib.get_engine();
                auto& strm = eng.get_service_stream();
                if (data_size < DATA_BLOCK_SIZE || output_layout.format.is_image_2d()) {
                    std::vector<uint8_t> _buf(data_size);
                    ib >> make_data(_buf.data(), data_size);
                    mem->copy_from(strm, _buf.data());
                } else if (eng.supports_allocation(allocation_type::usm_host)) {
                    // Use USM host staging buffers so the driver can DMA directly
                    // from pinned host memory to device, avoiding the hidden
                    // pageable-to-pinned bounce copy that std::vector staging incurs.
                    // The pipeline is the same ping-pong as the fallback below:
                    //   CPU reads into host_bufA  while  GPU copies from host_bufB
                    // Safety: each staging buffer is reused every other iteration;
                    // the wait() before re-submitting a copy ensures the previous
                    // GPU read from that buffer has finished before the CPU writes
                    // new data into it (the OTHER buffer's event is waited, but that
                    // event was submitted after waiting for THIS buffer's previous
                    // event, so this buffer is always free by the time we reuse it).
                    const layout staging_layout{{static_cast<int64_t>(DATA_BLOCK_SIZE)},
                                                data_types::u8, format::bfyx};
                    auto host_buf1 = eng.allocate_memory(staging_layout, allocation_type::usm_host, false);
                    auto host_buf2 = eng.allocate_memory(staging_layout, allocation_type::usm_host, false);
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
                            ib >> make_data(static_cast<uint8_t*>(host_buf1->buffer_ptr()), copy_size);
                            if (ev2 != nullptr) {
                                ev2->wait();
                                ev2 = nullptr;
                            }
                            ev1 = mem->copy_from(strm, *host_buf1, src_offset, dst_offset, copy_size, is_blocking);
                        } else {
                            ib >> make_data(static_cast<uint8_t*>(host_buf2->buffer_ptr()), copy_size);
                            if (ev1 != nullptr) {
                                ev1->wait();
                                ev1 = nullptr;
                            }
                            ev2 = mem->copy_from(strm, *host_buf2, src_offset, dst_offset, copy_size, is_blocking);
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
                } else {
                    // Fallback for devices that do not support USM host allocation:
                    // use pageable std::vector staging buffers (original behaviour).
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
};
}  // namespace cldnn
