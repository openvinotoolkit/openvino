// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/loop.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "primitive_inst.h"
#include <string>
#include <memory>
#include <vector>

namespace cldnn {
template<>
struct typed_program_node<loop> : public typed_program_node_base<loop> {
private:
    using parent = typed_program_node_base<loop>;

    std::vector<loop::io_primitive_map>& input_primitive_maps;
    std::vector<loop::io_primitive_map>& output_primitive_maps;
    std::vector<loop::backedge_mapping>& back_edges;

public:
    typed_program_node(std::shared_ptr<loop> prim, program& prog) :
        parent(prim, prog),
        input_primitive_maps(prim->input_primitive_maps),
        output_primitive_maps(prim->output_primitive_maps),
        back_edges(prim->back_edges) {}

    program::ptr get_body_program() const { return get_primitive()->body_program; }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->first_execution_condition_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->body_current_iteration_id; }
    const primitive_id& get_execution_condition_id() const { return get_primitive()->body_execution_condition_id; }
    const primitive_id& get_num_iterations_id() const { return get_primitive()->num_iteration_id; }
    const int32_t get_max_num_iteration() const { return get_primitive()->max_num_iterations; }

    const std::vector<loop::io_primitive_map>& get_input_primitive_maps() const { return input_primitive_maps; }
    const std::vector<loop::io_primitive_map>& get_output_primitive_maps() const { return output_primitive_maps; }
    const std::vector<loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID, bool external_id = true) {
        if (external_id) {
            for (auto& pm : input_primitive_maps) {
                if (pm.external_id.pid == prevID) {
                    pm.external_id.pid = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.external_id.pid == prevID) {
                    pm.external_id.pid = newID;
                }
            }
        } else {
            for (auto& pm : input_primitive_maps) {
                if (pm.internal_id.pid == prevID) {
                    pm.internal_id.pid = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.internal_id.pid == prevID) {
                    pm.internal_id.pid = newID;
                }
            }
            for (auto& back_edge : back_edges) {
                if (back_edge.from == prevID) {
                    back_edge.from = newID;
                }
                if (back_edge.to == prevID) {
                    back_edge.to = newID;
                }
            }
        }
    }

    // current_iteration is necessary to calculate output layout in dynamic shape
    std::vector<size_t> get_shape_infer_dependencies() const override { return {0}; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->inner_progs = { get_primitive()->body_program };
        // Set memory_deps using custom get_memory_deps to add current_iteration(mutable_data) into memory_deps
        params->memory_deps = get_memory_deps();
        return params;
    }

private:
    std::map<size_t, memory::ptr> get_memory_deps() const;
};

using loop_node = typed_program_node<loop>;

template <>
class typed_primitive_inst<loop> : public typed_primitive_inst_base<loop> {
    using parent = typed_primitive_inst_base<loop>;
    using parent::parent;

public:
    struct concatenated_memory_mapping {
        using ptr = std::shared_ptr<concatenated_memory_mapping>;
        using cptr = std::shared_ptr<const concatenated_memory_mapping>;
        concatenated_memory_mapping(int64_t axis,
                                    memory::ptr concatenated_mem,
                                    std::vector<memory::ptr> sliced_mems, // To change shared ptr vector
                                    stream& stream,
                                    engine& engine,
                                    int64_t iteration_elements = 0,
                                    int64_t stride = 0,
                                    int64_t initial_offset = 0) :
            axis(axis),
            concatenated_mem(concatenated_mem),
            sliced_mems(sliced_mems),
            stream(stream),
            engine(engine),
            iteration_elements(iteration_elements),
            stride(stride),
            initial_offset(initial_offset) {
                calculate_concatenated_mem();
            }

        concatenated_memory_mapping(const concatenated_memory_mapping& o) :
            axis(o.axis),
            concat_data_prim(o.concat_data_prim),
            sliced_data_prim(o.sliced_data_prim),

            concatenated_mem(o.concatenated_mem),
            sliced_mems(o.sliced_mems),
            stream(o.stream),
            engine(o.engine),
            iteration_elements(o.iteration_elements),
            stride(o.stride),
            initial_offset(o.initial_offset),

            bytes_per_element(o.bytes_per_element),
            batch_size(o.batch_size),
            bytes_batch_stride(o.bytes_batch_stride),
            bytes_iteration(o.bytes_iteration),
            bytes_iteration_stride(o.bytes_iteration_stride),
            bytes_iteration_initial_offset(o.bytes_iteration_initial_offset) {}


        static int64_t get_batch_size(layout mem_layout, int64_t axis) {
            if (axis < 0) {
                throw std::runtime_error("axis should be positive integer or zero");
            }

            if (mem_layout.is_dynamic()) {
                return -1;
            }

            int64_t batch_size = 1;
            for (int64_t i = 0; i < axis; ++i) {
                batch_size *= mem_layout.get_tensor().raw[i];
            }
            for (int64_t i = axis-1; i >= 2; --i) {
                batch_size *= mem_layout.get_tensor().raw[i];
            }
            return batch_size;
        }

        void calculate_concatenated_mem() const {
            if (!sliced_mems.empty() && concatenated_mem != nullptr) {
                auto& sliced_layout = sliced_mems.front()->get_layout();
                const int64_t num_elements_batch = get_batch_size(sliced_layout, axis);
                iteration_elements = sliced_layout.count() / num_elements_batch;
                bytes_per_element = data_type_traits::size_of(concatenated_mem->get_layout().data_type);
                batch_size = get_batch_size(concatenated_mem->get_layout(), axis);
                bytes_batch_stride = (static_cast<int64_t>(concatenated_mem->get_layout().count()) / batch_size) * bytes_per_element;
                bytes_iteration = iteration_elements * bytes_per_element;
                bytes_iteration_stride = stride * bytes_iteration;
                bytes_iteration_initial_offset = initial_offset * bytes_iteration;
            }
        }

        void update_concatenated_mem(memory::ptr mem) {
            if (concatenated_mem != nullptr && concatenated_mem->get_layout() == mem->get_layout()) {
                concatenated_mem = mem;
            } else {
                concatenated_mem = mem;
                calculate_concatenated_mem();
            }
        }

        void restore_concatenated_mem() const {
            OPENVINO_ASSERT(concatenated_mem != nullptr, "concatenated_mem should not be nullptr");
            mem_lock<uint8_t> concat_mem_lock{ concatenated_mem, stream };
            int64_t iteration_offset = bytes_iteration_initial_offset;
            for (const auto& sliced_mem : sliced_mems) {
                // To support multi-batch, just repeat memcpy for each batch
                for (int64_t batch = 0; batch < batch_size; ++batch) {
                    const int64_t src_offset = batch * bytes_iteration;
                    const int64_t dst_offset = batch * bytes_batch_stride + iteration_offset;
                    mem_lock<uint8_t> sliced_mem_lock{ sliced_mem, stream };
                    uint8_t* src = sliced_mem_lock.data() + src_offset;
                    uint8_t* dst = concat_mem_lock.data() + dst_offset;
                    std::copy(src, src + bytes_iteration, dst);
                }
                iteration_offset += bytes_iteration_stride;
            }
        }

        // Get sliced mem for the iteration idx and copy data from external input to sliced mem
        // In the case of dynamic model, concatenated_mem is always non nullptr.
        memory::ptr get_sliced_mem(int64_t iteration) const {
            OPENVINO_ASSERT(!sliced_mems.empty(), "For input data, sliced_mems should not be empty");
            mem_lock<uint8_t, mem_lock_type::read> from_lock{ concatenated_mem, stream };
            int64_t batch_offset = 0;
            auto sliced_mem = get_or_create_sliced_mem(iteration, sliced_mems.front()->get_layout());
            const int64_t iteration_offset = bytes_iteration_initial_offset +
                bytes_iteration_stride * iteration;
            // To support multi-batch, just repeat memcpy for each batch
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                const int64_t src_offset = batch_offset + iteration_offset;
                const int64_t dst_offset = batch * bytes_iteration;
                mem_lock<uint8_t> to_lock{ sliced_mem, stream };
                const auto src = from_lock.begin() + src_offset;
                const auto dst = to_lock.begin() + dst_offset;
                std::copy(src, src + bytes_iteration, dst);
                batch_offset += bytes_batch_stride;
            }
            return sliced_mem;
        }

        memory::ptr get_or_create_sliced_mem(int64_t idx, const layout& mem_layout) const {
            bool recalc_data = !sliced_mems.empty();
            while (sliced_mems.size() <= static_cast<size_t>(idx)) {
                memory::ptr sliced_mem = engine.allocate_memory(mem_layout, 0);
                sliced_mems.push_back(sliced_mem);
            }
            if (recalc_data) {
                calculate_concatenated_mem();
            }
            return sliced_mems.at(idx);
        }

        void setup_sliced_output_memory(uint64_t iteration) const {
            if (sliced_data_prim) {
                OPENVINO_ASSERT(iteration < sliced_mems.size(), "invalid index");
                const auto& sliced_output_mem = sliced_mems.at(iteration);
                sliced_data_prim->set_output_memory(sliced_output_mem);
            }
        }

        std::vector<memory::ptr>& get_sliced_mems() const { return sliced_mems; }

        void reset_data_for_shape_changed() {
            bytes_per_element = 0;
            batch_size = 0;
            bytes_batch_stride = 0;
            bytes_iteration = 0;
            bytes_iteration_stride = 0;
            bytes_iteration_initial_offset = 0;
            if (concatenated_mem) concatenated_mem = nullptr;
            iteration_elements = 0;
            sliced_mems.clear();
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "concatenated_memory_mapping [" << std::endl;
            ss << "* axis                           : " << axis << std::endl;
            ss << "* bytes_per_element              : " << bytes_per_element << std::endl;
            ss << "* batch_size                     : " << batch_size << std::endl;
            if (concatenated_mem != nullptr && concatenated_mem->get_layout().is_static()) {
                ss << "* bytes_batch_stride             : " << bytes_batch_stride << " = (static_cast<int64_t>("
                    << concatenated_mem->get_layout().count() << ") / batch_size:" << batch_size << ") * bytes_per_element:" << bytes_per_element << std::endl;
            } else {
                ss << "* bytes_batch_stride             : " << bytes_batch_stride << std::endl;
            }
            ss << "* bytes_iteration                : " << bytes_iteration << " = (iteration_elements:"
                << iteration_elements << " * bytes_per_element:" << bytes_per_element << ")" << std::endl;
            ss << "* bytes_iteration_stride         : " << bytes_iteration_stride << std::endl;
            ss << "* bytes_iteration_initial_offset : " << bytes_iteration_initial_offset << std::endl;
            ss << "* concat_data_prim               : " << ((concat_data_prim != nullptr)? concat_data_prim->id() : "nullptr") << std::endl;
            ss << "* sliced_data_prim               : " << ((sliced_data_prim != nullptr)? sliced_data_prim->id()  : "nullptr") << std::endl;
            if (concatenated_mem) {
                ss << "* concatenated_mem               : " << concatenated_mem->get_layout().to_short_string() << std::endl;
            } else {
                ss << "* concatenated_mem               : nullptr" << std::endl;
            }
            ss << "* iteration_elements             : " << iteration_elements << std::endl;
            ss << "* stride                         : " << stride << std::endl;
            ss << "* initial_offset                 : " << initial_offset << std::endl;
            ss << "* sliced_mems                    :{ ";
            for (auto mem : sliced_mems) {
                ss << mem->get_layout().to_short_string() << ",";
            }
            ss << "}]" << std::endl;
            return ss.str();
        }

        const int64_t axis;
        std::shared_ptr<primitive_inst> concat_data_prim;
        std::shared_ptr<primitive_inst> sliced_data_prim;

private:
        mutable memory::ptr concatenated_mem;
        mutable std::vector<memory::ptr> sliced_mems;
        cldnn::stream& stream;
        cldnn::engine& engine;
        mutable int64_t iteration_elements = 0;
        const int64_t stride = 0;
        const int64_t initial_offset = 0;

        // element size
        mutable int64_t bytes_per_element;
        // number of higher level of dimension of slicing axis
        mutable int64_t batch_size;
        // stride of batch in concatenated memory
        mutable int64_t bytes_batch_stride;
        // byte size of each iteration per batch in a sliced memory
        mutable int64_t bytes_iteration;
        // byte size of each iteration (bytes_iteration * batch_size) in a sliced memory
        mutable int64_t bytes_iteration_stride;
        // byte offset of 1st iteration in a batch in a sliced memory
        mutable int64_t bytes_iteration_initial_offset;
    };

    struct backedge_memory_mapping {
        enum backedge_type {
            // output memory(from_primitive) of body network needs to be concatenated
            CONCAT_OUTPUT,
            // output memory(from_primitive) of body network does not need to be concatenated
            // input memory is shared by output memory
            SINGLE_SHARED,
            // output memory(from_primitive) of body network does not need to be concatenated
            // input memory is not shared by output memory
            // each iteration input memory and output memory are swapped
            SINGLE,
        };
        std::shared_ptr<primitive_inst> from_primitive;
        std::shared_ptr<primitive_inst> to_primitive;
        std::shared_ptr<concatenated_memory_mapping> concat_mem_mapping;
        mutable memory::ptr from_mem;
        memory::ptr initial_mem;
        cldnn::stream& stream;
        backedge_type type;
        size_t total_bytes;

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            std::shared_ptr<concatenated_memory_mapping> _concat_mem_mapping, memory::ptr _initial_mem,
            cldnn::stream& _stream, backedge_type _type = CONCAT_OUTPUT):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            concat_mem_mapping(std::move(_concat_mem_mapping)),
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _from_mem, memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE_SHARED):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            from_mem{std::move(_from_mem)},
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE):
            from_primitive(_from_primitive),
            to_primitive(std::move(_to_primitive)),
            initial_mem(std::move(_initial_mem)),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

private:
        void validate_backedge_memory() {
            if (from_mem) {
                const size_t from_mem_bytes = from_mem->get_layout().bytes_count();
                OPENVINO_ASSERT((from_mem_bytes == total_bytes), "Invalid backedge memory layout: size(",
                        from_mem_bytes, ",", from_mem->get_layout().to_short_string(),
                        ") not matched with that of initial_mem(", total_bytes,
                        ",", initial_mem->get_layout().to_short_string(), ")");
            }
            if (concat_mem_mapping) {
                for (const auto& from_mem : concat_mem_mapping->get_sliced_mems()) {
                    const size_t from_mem_bytes = from_mem->get_layout().bytes_count();
                    OPENVINO_ASSERT((from_mem_bytes == total_bytes), "Invalid backedge memory layout: size(",
                        from_mem_bytes, ",", from_mem->get_layout().to_short_string(),
                        ") not matched with that of initial_mem(", total_bytes,
                        ",", initial_mem->get_layout().to_short_string(), ")");
                }
            }
        }
    };

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(loop_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(const loop_node& /*node*/, kernel_impl_params const& impl_param);
    bool preproc_memories_done = false;
    std::vector<backedge_memory_mapping> backedge_memory_mappings;
    std::vector<concatenated_memory_mapping::ptr> concatenated_input_mem_mappings;
    std::vector<concatenated_memory_mapping::ptr> concatenated_output_mem_mappings;

    static std::string to_string(const loop_node& node);

public:
    typed_primitive_inst(network& network, const loop_node& node);
    network::ptr get_body_network() const { return body_network; }
    void preprocess_input_memory(const int64_t trip_count);
    void preprocess_output_memory(const int64_t trip_count);
    void preprocess_backedge_memory();
    void update_mapped_memory();
    void update_input_mapped_memory();
    void update_output_mapped_memory();
    void update_backedge_mapped_memory();
    void postprocess_output_memory(bool is_dynamic);
    concatenated_memory_mapping::ptr create_concat_memory_map(const input_info& id,
                                                                const cldnn::loop::io_primitive_map& io_prim_map,
                                                                memory::ptr mem_ptr,
                                                                const int64_t trip_count);
    event::ptr set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0) override;
    void reset_memory();

    void save(BinaryOutputBuffer& ob) const override;
    void load(BinaryInputBuffer& ib) override;
    void validate_backedges(loop_node const & node) const;

    void update_shape() override { primitive_inst::update_shape(); }
    void update_output_layout();

private:
    network::ptr body_network;
    memory::ptr get_external_memory(const primitive_id& external_id, size_t mem_idx = 0) const;
    layout get_external_output_layout(const primitive_id& external_id, size_t mem_idx = 0) const;
    std::shared_ptr<concatenated_memory_mapping> get_sliced_mem(const primitive_id& internal_id) const;
    std::vector<loop::io_primitive_map> _input_primitive_maps;
    std::vector<loop::io_primitive_map> _output_primitive_maps;
    std::vector<loop::backedge_mapping> _back_edges;
    primitive_id _trip_count_id;
    primitive_id _initial_execution_id;
    primitive_id _current_iteration_id;
    primitive_id _condition_id;
    primitive_id _num_iterations_id;
};

using loop_inst = typed_primitive_inst<loop>;

static inline std::ostream& operator<< (std::ostream& os, loop_inst::concatenated_memory_mapping& map) {
    os << map.to_string();
    return os;
}
}  // namespace cldnn
