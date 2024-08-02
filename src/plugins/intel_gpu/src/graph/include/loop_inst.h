// Copyright (C) 2018-2024 Intel Corporation
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

    primitive_id trip_count_id;
    primitive_id initial_execution_id;
    primitive_id current_iteration_id;
    primitive_id execution_condition_id;
    primitive_id num_iterations_id;

    std::vector<loop::io_primitive_map>& input_primitive_maps;
    std::vector<loop::io_primitive_map>& output_primitive_maps;
    std::vector<loop::backedge_mapping>& back_edges;

public:
    typed_program_node(std::shared_ptr<loop> prim, program& prog) :
        parent(prim, prog),
        input_primitive_maps(prim->input_primitive_maps),
        output_primitive_maps(prim->output_primitive_maps),
        back_edges(prim->back_edges) {
        set_primitive_ids(prim);
    }

    program::ptr get_body_program() const { return get_primitive()->body_program; }

    const primitive_id& get_trip_count_id() const { return trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return initial_execution_id; }
    const primitive_id& get_current_iteration_id() const { return current_iteration_id; }
    const primitive_id& get_execution_condition_id() const { return execution_condition_id; }
    const primitive_id& get_num_iterations_id() const { return num_iterations_id; }

    const int32_t get_max_num_iteration() const { return get_primitive()->max_num_iterations; }

    const std::vector<loop::io_primitive_map>& get_input_primitive_maps() const { return input_primitive_maps; }
    const std::vector<loop::io_primitive_map>& get_output_primitive_maps() const { return output_primitive_maps; }
    const std::vector<loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    void set_primitive_ids(std::shared_ptr<loop> prim) {
        trip_count_id           = prim->trip_count_id;
        initial_execution_id    = prim->first_execution_condition_id;
        current_iteration_id    = prim->body_current_iteration_id;
        execution_condition_id  = prim->body_execution_condition_id;
        num_iterations_id       = prim->num_iteration_id;
    }

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

        // Update ids
        if (get_trip_count_id() == prevID)
            trip_count_id = newID;
        if (get_initial_execution_id() == prevID)
            initial_execution_id = newID;
        if (get_current_iteration_id() == prevID)
            current_iteration_id = newID;
        if (get_execution_condition_id() == prevID)
            execution_condition_id = newID;
        if (get_num_iterations_id() == prevID)
            num_iterations_id = newID;
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
        concatenated_memory_mapping(memory::ptr concatenated_mem,
                                    std::vector<memory::ptr> sliced_mems,
                                    stream& stream,
                                    engine& engine,
                                    std::shared_ptr<primitive_inst> concat_data_prim,
                                    std::shared_ptr<primitive_inst> sliced_data_prim,
                                    const cldnn::loop::io_primitive_map& io_prim_map) :
            concatenated_mem(concatenated_mem),
            sliced_mems(sliced_mems),
            stream(stream),
            engine(engine),
            concat_data_prim(std::move(concat_data_prim)),
            sliced_data_prim(std::move(sliced_data_prim)),
            io_prim_map(io_prim_map) {}

        concatenated_memory_mapping(const concatenated_memory_mapping& o) :
            concatenated_mem(o.concatenated_mem),
            sliced_mems(o.sliced_mems),
            stream(o.stream),
            engine(o.engine),
            concat_data_prim(o.concat_data_prim),
            sliced_data_prim(o.sliced_data_prim),
            io_prim_map(o.io_prim_map) {}

        void update_concatenated_mem(memory::ptr mem) {
            concatenated_mem = mem;
        }

        void slice_mem(const int64_t num_iteration) const;
        void concat_mem(const int64_t curent_iterations) const;

        // Get sliced mem for the iteration idx and copy data from external input to sliced mem
        // In the case of dynamic model, concatenated_mem is always non nullptr.
        memory::ptr get_sliced_mem(int64_t iteration) const {
            OPENVINO_ASSERT(static_cast<size_t>(iteration) < sliced_mems.size(), "invalid itertion(", iteration,
                                ") for sliced_mes(", sliced_mems.size(), ")");
            return sliced_mems.at(iteration);;
        }

        memory::ptr get_or_create_sliced_mem(int64_t idx, const layout& mem_layout) const {
            while (sliced_mems.size() <= static_cast<size_t>(idx)) {
                memory::ptr sliced_mem = engine.allocate_memory(mem_layout, 0);
                sliced_mems.push_back(sliced_mem);
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
            if (concatenated_mem) concatenated_mem = nullptr;
            sliced_mems.clear();
        }

        const input_info& get_external_id() {
            return io_prim_map.external_id;
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "concatenated_memory_mapping [" << std::endl;
            ss << "* concat_data_prim               : " << ((concat_data_prim != nullptr)? concat_data_prim->id() : "nullptr") << std::endl;
            ss << "* sliced_data_prim               : " << ((sliced_data_prim != nullptr)? sliced_data_prim->id()  : "nullptr") << std::endl;
            ss << "* concatenated_mem               : "
                << ((concatenated_mem != nullptr)? concatenated_mem->get_layout().to_short_string() : "nullptr") << std::endl;
            ss << "* sliced_mems                    :{ ";
            for (auto mem : sliced_mems) {
                ss << mem->get_layout().to_short_string() << ",";
            }
            ss << "* io_prim_map                    : " << io_prim_map.to_string() << std::endl;
            ss << "}]" << std::endl;
            return ss.str();
        }

        std::shared_ptr<primitive_inst> get_sliced_data_prim() {
            OPENVINO_ASSERT(sliced_data_prim != nullptr, "sliced_data_prim should not be nullptr");
            return sliced_data_prim;
        }

        primitive_id get_sliced_data_prim_id() {
            OPENVINO_ASSERT(sliced_data_prim != nullptr, "sliced_data_prim should not be nullptr");
            return sliced_data_prim->id();
        }

private:
        mutable memory::ptr concatenated_mem;
        mutable std::vector<memory::ptr> sliced_mems;
        cldnn::stream& stream;
        cldnn::engine& engine;
        std::shared_ptr<primitive_inst> concat_data_prim;
        std::shared_ptr<primitive_inst> sliced_data_prim;
        const cldnn::loop::io_primitive_map& io_prim_map;
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
            bool is_dynamic = (from_primitive->is_dynamic() || to_primitive->is_dynamic());
            if (!is_dynamic && from_mem) {
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
    void preprocess_input_memory(const int64_t num_iteration);
    void preprocess_output_memory(const int64_t num_iteration);
    void preprocess_backedge_memory();
    void update_mapped_memory();
    void update_input_mapped_memory();
    void update_output_mapped_memory();
    void update_backedge_mapped_memory();
    void postprocess_output_memory(bool is_dynamic, int64_t current_iteration);
    concatenated_memory_mapping::ptr create_concat_memory_map(const cldnn::loop::io_primitive_map& io_prim_map,
                                                                memory::ptr mem_ptr,
                                                                const int64_t num_iteration);
    event::ptr set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0) override;
    void reset_memory();

    void validate_backedges(loop_node const & node) const;

    void update_shape() override { primitive_inst::update_shape(); }
    void update_output_layout();

    // num_iteration is used for slicing input memory
    int64_t get_num_iterations();

    std::vector<event::ptr> preprocess_memory_for_body_network(int64_t current_iteration_idx);
    std::vector<event::ptr> postprocess_memory_for_body_network(int64_t current_iteration_idx);

    primitive_id get_trip_count_id()        { return _trip_count_id; }
    primitive_id get_initial_execution_id() { return _initial_execution_id; }
    primitive_id get_current_iteration_id() { return _current_iteration_id; }
    primitive_id get_condition_id()         { return _condition_id; }
    primitive_id get_num_iterations_id()    { return _num_iterations_id; }

private:
    network::ptr body_network;
    memory::ptr get_external_memory(const primitive_id& external_id, size_t mem_idx = 0) const;
    layout get_external_output_layout(const primitive_id& external_id, size_t mem_idx = 0) const;
    std::shared_ptr<concatenated_memory_mapping> get_sliced_mem(const primitive_id& internal_id) const;
    int64_t calculate_num_iterations(const cldnn::loop::io_primitive_map& io_prim_map, ov::PartialShape& pshape);
    std::vector<event::ptr> handle_buffers_for_next_iteration(const backedge_memory_mapping& mapping,
                                                                network::ptr body_network, int64_t iter);
    void set_memory_in_body_network(cldnn::network::ptr body_network, const std::shared_ptr<cldnn::primitive_inst>& inst,
                                        memory::ptr mem);

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
