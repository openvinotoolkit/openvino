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
    mutable topology body;

    std::vector<loop::io_primitive_map> input_primitive_maps;
    std::vector<loop::io_primitive_map> output_primitive_maps;
    mutable std::vector<loop::backedge_mapping> back_edges;
    bool use_current_iteration;
    bool use_execution_condition;
    mutable program::ptr body_program;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog) :
        parent(prim, prog),
        body(this->get_primitive()->body),
        input_primitive_maps(this->get_primitive()->input_primitive_maps),
        output_primitive_maps(this->get_primitive()->output_primitive_maps),
        back_edges(this->get_primitive()->back_edges),
        use_current_iteration(!this->get_primitive()->current_iteration_id.empty()),
        use_execution_condition(!this->get_primitive()->condition_id.empty()),
        iteration_axis(0),
        max_iteration(this->get_primitive()->max_iteration < 0 ? DEFAULT_MAX_NUM_ITERATION : this->get_primitive()->max_iteration) {}

    mutable size_t iteration_axis;
    int64_t max_iteration;

    int64_t get_max_iteration() const { return max_iteration; }
    program::ptr get_body_program() const { return body_program; }
    bool is_current_iteration_used() const { return use_current_iteration; }
    bool is_execution_condition_used() const { return use_execution_condition; }

    static size_t convert_to_raw_axis(size_t axis, size_t ndim) {
        // convert between bfyx, bfzyx, bfzyxw and tensor.size.raw
        if (axis >= ndim) {
            throw std::runtime_error("axis should be less than ndim");
        }

        if (axis < 2) {
            return axis;
        }
        return (ndim - 1) - (axis - 2);
    }

    // read scala value from data primitive
    static int64_t read_scalar_value(memory::ptr mem, stream& stream) {
        int64_t trip_count = 0;
        const layout& prim_layout = mem->get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            mem_lock<uint8_t> lock_prim_output{mem, stream};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i8: {
            mem_lock<int8_t> lock_prim_output{mem, stream};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i32: {
            mem_lock<int32_t> lock_prim_output{mem, stream};
            trip_count = *lock_prim_output.data();
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem, stream};
            trip_count = *lock_prim_output.data();
            break;
        }
        default:
            throw std::runtime_error("Invalid data type : " + data_type_traits::name(prim_layout.data_type));
        }
        return trip_count;
    }

    template<typename T>
    static inline void validate_input_value(int64_t input) {
        if (input < std::numeric_limits<T>::min() || input > std::numeric_limits<T>::max()) {
            throw std::runtime_error("Invalid data value : " + std::to_string(input));
        }
    }

    static void write_scalar_value(memory::ptr mem, stream& stream, int64_t input) {
        const layout& prim_layout = mem->get_layout();

        switch (prim_layout.data_type) {
        case data_types::u8: {
            validate_input_value<uint8_t>(input);
            mem_lock<uint8_t> lock_prim_output{mem, stream};
            lock_prim_output[0] = static_cast<uint8_t>(input);
            break;
        }
        case data_types::i8: {
            validate_input_value<int8_t>(input);
            mem_lock<int8_t> lock_prim_output{mem, stream};
            lock_prim_output[0] = static_cast<int8_t>(input);
            break;
        }
        case data_types::i32: {
            validate_input_value<int32_t>(input);
            mem_lock<int32_t> lock_prim_output{mem, stream};
            lock_prim_output[0] = static_cast<int32_t>(input);
            break;
        }
        case data_types::i64: {
            mem_lock<int64_t> lock_prim_output{mem, stream};
            lock_prim_output[0] = input;
            break;
        }
        default:
            throw std::runtime_error("Invalid data type : " + data_type_traits::name(prim_layout.data_type));
        }
    }

    layout calc_body_input_layout(const loop::io_primitive_map& inputDesc) const {
        const auto& dependency_list = this->get_dependencies();
        auto input = std::find_if(dependency_list.begin(), dependency_list.end(), [&inputDesc](const std::pair<program_node*, int32_t>& dep){
            return dep.first->id() == inputDesc.external_id;
        });
        if (input == dependency_list.end()) {
            throw std::runtime_error("Can't find input from dependency_list");
        }
        layout calculated_layout = (*input).first->get_output_layout();
        auto shape = calculated_layout.get_tensor().sizes(calculated_layout.format);

        if (inputDesc.axis >= 0) {
            iteration_axis = convert_to_raw_axis(static_cast<size_t>(inputDesc.axis), shape.size());
            auto calculated_size = calculated_layout.get_tensor();
            calculated_size.raw[iteration_axis] = 1; // cropped inputs shape
            calculated_layout.set_tensor(calculated_size);
        }

        return calculated_layout;
    }

    const std::vector<loop::io_primitive_map>& get_input_primitive_maps() const { return input_primitive_maps; }
    const std::vector<loop::io_primitive_map>& get_output_primitive_maps() const { return output_primitive_maps; }

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID, bool external_id = true) {
        if (external_id) {
            for (auto& pm : input_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
        } else {
            for (auto& pm : input_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
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

    const std::vector<cldnn::loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    static bool is_integer(const data_types& data_type) {
        switch (data_type) {
            case data_types::u8:
            case data_types::i8:
            case data_types::i32:
            case data_types::i64:
                return true;
            default:
                return false;
        }
    }

    void process_current_iteration() const {
        const primitive_id& current_iteration_id = get_current_iteration_id();
        if (current_iteration_id.empty()) {
            return;
        }

        const topology_map& body_topology_map = body.get_primitives();
        const layout body_input_layout(data_types::i64, format::bfyx, {1, 1, 1, 1});

        // add current_iteration primitive if current_iteration primitive is not exist in body
        if (body_topology_map.find(current_iteration_id) == body_topology_map.end()) {
            body.add_primitive(std::make_shared<input_layout>(current_iteration_id, body_input_layout));
        } else {
            const auto& body_input_prim = body.at(current_iteration_id);
            const auto input_layout_prim = std::dynamic_pointer_cast<input_layout>(body_input_prim);
            OPENVINO_ASSERT(input_layout_prim, "[GPU] current_iteration primitive should be cldnn::input_layout in node", this->id());
            input_layout_prim->change_layout(body_input_layout);
        }

        // add incremental data: 1
        // it is used to update current_iteration in body network
        const primitive_id increment_value_id = current_iteration_id + "_inc";
        auto mem = get_program().get_engine().allocate_memory(body_input_layout);
        auto& stream = get_program().get_stream();
        write_scalar_value(mem, stream, 1);
        body.add_primitive(std::make_shared<data>(increment_value_id, mem));

        // add eltwise sum updating current_iteration with incremental data
        const primitive_id updated_currnet_iteration_id = current_iteration_id + "_update";
        body.add_primitive(std::make_shared<eltwise>(updated_currnet_iteration_id,
            current_iteration_id, increment_value_id, eltwise_mode::sum));

        // set backedge
        back_edges.emplace_back(updated_currnet_iteration_id, current_iteration_id);
    }

    void process_single_int_output(const primitive_id& id) const {
        // add mutable if not exist
        const topology_map& body_topology_map = body.get_primitives();
        layout body_output_layout(data_types::i64, format::bfyx, {1, 1, 1, 1});
        if (!id.empty()) {
            auto body_output = body_topology_map.find(id);
            if (body_output == body_topology_map.end()) {
                auto mem = get_program().get_engine().allocate_memory(body_output_layout);
                auto md = std::make_shared<data>(id, mem);
                body.add_primitive(md);
            } else {
                auto body_output_prim = body.at(body_output->first);
                auto mem = get_program().get_engine().allocate_memory(body_output_layout);
                body_output_prim.reset(new mutable_data(body_output->first, mem));
            }
        }
    }

    void build_body_program() const {
        for (const auto& pm : input_primitive_maps) {
            layout calculated_layout = calc_body_input_layout(pm);
            const primitive_id& internal_input_id = pm.internal_id;

            // add inputs for body network if not exist
            if (body.get_primitives().count(internal_input_id) == 0) {
                body.add_primitive(std::make_shared<input_layout>(internal_input_id, calculated_layout));
            } else {
                body.change_input_layout(internal_input_id, calculated_layout);
            }
        }

        // setup internal output
        OPENVINO_ASSERT(!output_primitive_maps.empty(), "[GPU] Output primitive map should have at least 1 mapping in primitive ", this->id());
        std::set<primitive_id> output_names;
        output_names.insert(output_primitive_maps.front().internal_id);

        // add current_iteration_id in body network, condition_id if exist
        process_current_iteration();
        process_single_int_output(get_condition_id());

        // setup outputs for backedges
        for (auto& back_edge : back_edges) {
            // check whether the back_edge.to has its corresponding io_primitive_map
            const auto& input_map = std::find_if(input_primitive_maps.begin(), input_primitive_maps.end(),
                [&](const loop::io_primitive_map& pm) {
                    return pm.internal_id == back_edge.to;
                });

            // backedge which is current_iteration does not have
            // input primitive map because its initial value is always
            // zero and the value will be set in execute_impl()
            if (back_edge.to != get_current_iteration_id() && input_map == input_primitive_maps.end()) {
                std::string msg = "[GPU] No primitive mapping for backedge (internal_id: " + back_edge.to + ") for primitive " + this->id();
                OPENVINO_ASSERT(false, msg.c_str());
            }

            output_names.insert(back_edge.from);
        }

        // if execution_condition_id is specified, we need to add the id in build_option::outputs
        if (!get_condition_id().empty()) {
            output_names.insert(get_condition_id());
        }

        std::vector<primitive_id> output_names_vec(output_names.begin(), output_names.end());
        auto config = get_program().get_config();
        config.set_property(ov::intel_gpu::custom_outputs(output_names_vec));
        body_program = program::build_program(get_program().get_engine(), body, config, false, false, true);
    }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->initial_execution_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->current_iteration_id; }
    const primitive_id& get_condition_id() const { return get_primitive()->condition_id; }
    const primitive_id& get_num_iteration_id() const { return get_primitive()->num_iteration_id; }
    const topology& get_body_topology() const { return get_primitive()->body; }
};

using loop_node = typed_program_node<loop>;

template <>
class typed_primitive_inst<loop> : public typed_primitive_inst_base<loop> {
    using parent = typed_primitive_inst_base<loop>;
    using parent::parent;

public:
    struct backedge_memory_mapping {
        enum backedge_type {
            // output memory(from_primitive) of body network needs to be concatenated
            CONCAT_OUTPUT,
            // output memory(from_primitive) of body network does not need to be concateneated
            // input memory is shared by output memory
            SINGLE_SHARED,
            // output memory(from_primitive) of body network does not need to be concateneated
            // input memory is not shared by output memroy
            // each iteration input memory and output memory are swapped
            SINGLE,
        };
        std::shared_ptr<primitive_inst> from_primitive;
        std::shared_ptr<primitive_inst> to_primitive;
        std::vector<memory::ptr> from_mems;
        memory::ptr initial_mem;
        cldnn::stream& stream;
        backedge_type type;
        size_t total_bytes;

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            std::vector<memory::ptr> _from_mems, memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = CONCAT_OUTPUT):
            from_primitive(_from_primitive),
            to_primitive(_to_primitive),
            from_mems(_from_mems),
            initial_mem(_initial_mem),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _from_mem, memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE_SHARED):
            from_primitive(_from_primitive),
            to_primitive(_to_primitive),
            from_mems{_from_mem},
            initial_mem(_initial_mem),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> _from_primitive, std::shared_ptr<primitive_inst> _to_primitive,
            memory::ptr _initial_mem, cldnn::stream& _stream, backedge_type _type = SINGLE):
            from_primitive(_from_primitive),
            to_primitive(_to_primitive),
            initial_mem(_initial_mem),
            stream(_stream),
            type(_type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        void setup_iteration(int64_t iter) const {
            if (type == CONCAT_OUTPUT) {
                if (iter == 0) {
                    to_primitive->set_output_memory(initial_mem);
                } else if (iter > 0) {
                    to_primitive->set_output_memory(from_mems.at(iter - 1));
                } else {
                    throw std::runtime_error("Invalid iteraton count" + std::to_string(iter));
                }
            } else if (type == SINGLE_SHARED && iter == 0) {
                from_mems.front()->copy_from(stream, *initial_mem);
            } else if (type == SINGLE) {
                memory::ptr mem1 = to_primitive->output_memory_ptr();
                if (iter == 0) {
                    mem1->copy_from(stream, *initial_mem);
                } else {
                    memory::ptr mem2 = from_primitive->output_memory_ptr();
                    to_primitive->set_output_memory(mem2);
                    from_primitive->set_output_memory(mem1);
                }
            }
        }

private:
        void validate_backedge_memory() {
            for (const auto& from_mem : from_mems) {
                const size_t from_mem_bytes = from_mem->get_layout().bytes_count();
                if (from_mem_bytes != total_bytes) {
                    throw std::runtime_error("Invalid backedge memory layout: "
                        "size not matched with that of initial_mem");
                }
            }
        }
    };

    struct concatenated_memory_mapping {
        concatenated_memory_mapping(int64_t axis,
                                    memory::ptr concatenated_mem,
                                    std::vector<memory::ptr> sliced_mems,
                                    stream& stream,
                                    int64_t iteration_elements = 0,
                                    int64_t stride = 0,
                                    int64_t initial_offset = 0) :
            axis(axis),
            concatenated_mem(concatenated_mem),
            sliced_mems(sliced_mems),
            stream(stream),
            bytes_per_element(data_type_traits::size_of(concatenated_mem->get_layout().data_type)),
            batch_size(get_batch_size(concatenated_mem->get_layout(), axis)),
            bytes_batch_stride((static_cast<int64_t>(concatenated_mem->get_layout().count()) / batch_size) * bytes_per_element),
            bytes_iteration(iteration_elements * bytes_per_element),
            bytes_iteration_stride(stride * bytes_iteration),
            bytes_iteration_initial_offset(initial_offset * bytes_iteration) {}

        static int64_t get_batch_size(layout mem_layout, int64_t axis) {
            if (axis < 0) {
                throw std::runtime_error("axis should be positive integer or zero");
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

        void restore_concatenated_mem() const {
            mem_lock<uint8_t> concat_mem_lock{ concatenated_mem, stream };
            int64_t iteration_offset = bytes_iteration_initial_offset;
            for (const auto& sliced_mem : sliced_mems) {
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

        void setup_concatenated_output_memory(uint64_t iteration) const {
            const auto& sliced_output_mem = sliced_mems.at(iteration);
            concat_data_prim->set_output_memory(sliced_output_mem);
        }

        memory::ptr get_sliced_mem(int64_t iteration) const {
            mem_lock<uint8_t, mem_lock_type::read> from_lock{ concatenated_mem, stream };
            int64_t batch_offset = 0;
            const int64_t iteration_offset = bytes_iteration_initial_offset +
                bytes_iteration_stride * iteration;
            for (int64_t batch = 0; batch < batch_size; ++batch) {
                const int64_t src_offset = batch_offset + iteration_offset;
                const int64_t dst_offset = batch * bytes_iteration;
                mem_lock<uint8_t> to_lock{ sliced_mems.at(iteration), stream };
                const auto src = from_lock.begin() + src_offset;
                const auto dst = to_lock.begin() + dst_offset;
                std::copy(src, src + bytes_iteration, dst);
                batch_offset += bytes_batch_stride;
            }
            return sliced_mems.at(iteration);
        }

        const int64_t axis;
        std::shared_ptr<primitive_inst> concat_data_prim;
        std::shared_ptr<primitive_inst> sliced_data_prim;
        memory::ptr concatenated_mem;
        std::vector<memory::ptr> sliced_mems;
        cldnn::stream& stream;
        // element size
        const int64_t bytes_per_element;
        // number of higher level of dimension of slicing axis
        const int64_t batch_size;
        // stride of batch in concatanated memory
        const int64_t bytes_batch_stride;
        // byte size of each iteration per batch in a sliced memory
        const int64_t bytes_iteration;
        // byte size of each iteration (bytes_iteration * batch_size) in a sliced memory
        const int64_t bytes_iteration_stride;
        // byte offset of 1st iteration in a batch in a sliced memory
        const int64_t bytes_iteration_initial_offset;
    };

    static layout calc_output_layout(const loop_node& node, kernel_impl_params const& impl_param);
    bool preproc_memories_done;
    std::vector<backedge_memory_mapping> backedge_memory_mappings;
    std::vector<concatenated_memory_mapping> concatenated_input_mem_mappings;
    std::vector<concatenated_memory_mapping> concatenated_output_mem_mappings;

    static std::string to_string(const loop_node& node);
    size_t current_iteratoin_backedge_mapping_idx = 0;

public:
    typed_primitive_inst(network& network, const loop_node& node);
    network::ptr get_body_network() const { return body_network; }
    void preprocess_input_memory();
    void preprocess_output_memory();
    void preprocess_backedge_memory();
    void update_mapped_memory();
    void set_output_memory(memory::ptr mem, bool check = true, size_t idx = 0) override;
    const backedge_memory_mapping& get_current_iteration_backedge_mapping() const {
        OPENVINO_ASSERT(node->is_current_iteration_used(), "[GPU] No backedge mapping for current_iteration for primitive ", node->id());
        return backedge_memory_mappings.at(current_iteratoin_backedge_mapping_idx);
    }
    void save(BinaryOutputBuffer& ob) const override;
    void load(BinaryInputBuffer& ib) override;

private:
    network::ptr body_network;
    memory::ptr get_external_memory(const primitive_id& external_id) const;
    std::vector<memory::ptr> get_sliced_mem(const primitive_id& internal_id) const;
    std::vector<loop::io_primitive_map> _input_primitive_maps;
    std::vector<loop::io_primitive_map> _output_primitive_maps;
    std::vector<loop::backedge_mapping> _back_edges;
    primitive_id _trip_count_id;
    primitive_id _initial_execution_id;
    primitive_id _current_iteration_id;
    primitive_id _condition_id;
    primitive_id _num_iteration_id;
    int64_t _max_iteration;
};

using loop_inst = typed_primitive_inst<loop>;
}  // namespace cldnn
