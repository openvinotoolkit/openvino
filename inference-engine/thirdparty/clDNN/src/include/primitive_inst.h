// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "cldnn/primitives/primitive.hpp"
#include "cldnn/primitives/concatenation.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/memory.hpp"
#include "kernel_selector_helper.h"
#include "meta_utils.h"
#include "program_node.h"
#include "primitive_type.h"

#include <memory>
#include <vector>
#include <string>

namespace cldnn {

struct network_impl;
class primitive_inst;

template <class PType>
class typed_primitive_inst;

/*
    Base class for all implementations.
*/
struct primitive_impl {
    // NOTE: This constuctor in necessary since the spec says:
    //   A defaulted default constructor for class X is defined as deleted if: [...] any non-variant non-static data
    //   member of const-qualified type (or array thereof) with no brace-orequal-initializer does not have a
    //   user-provided default constructor.
    // and the classes with only declared brace-orequal-initializer on members are not considered to have user-provided
    // default constructor:
    //   A special member function is user-provided if it is user-declared and not explicitly defaulted or deleted
    //   on its first declaration.
    primitive_impl() : _weights_reorder_params() {}
    explicit primitive_impl(const kernel_selector::weights_reorder_params& params, std::string kernel_name = "")
        : _weights_reorder_params(params), _kernel_name(kernel_name) {}
    virtual ~primitive_impl() = default;

    virtual void set_arguments(primitive_inst& instance) = 0;
    virtual event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) = 0;
    virtual bool validate(const primitive_inst& instance) const = 0;
    std::string get_kernel_name() const { return _kernel_name; }
    // TODO: added a derived class for weights reordering (maybe for all static data reordering)
    kernel_selector::weights_reorder_params _weights_reorder_params;
    // class typed_primitive_gpu_impl override this with return false;
    virtual bool is_cpu() const { return true; }
    virtual void init_kernels() = 0;
    virtual std::unique_ptr<primitive_impl> clone() const = 0;

protected:
    std::string _kernel_name;
};

/*
    Base class for all primitive instances.
    It's main responsibility is to allocate memory required to run single, specified in ctor,
    program_node. It also contains informations about it's predecessor in network graph and checks (<<-- TODO)
    if output should be recalculated between network runs.
*/
class primitive_inst {
    template <class PType>
    friend class typed_primitive_inst;

public:
    virtual ~primitive_inst() = default;

    const std::vector<std::shared_ptr<const primitive_inst>>& dependencies() const {
        return reinterpret_cast<std::vector<std::shared_ptr<const primitive_inst>> const&>(_deps);
    }

    memory& dep_memory(size_t index) const { return dependencies().at(index)->output_memory(); }
    memory::ptr dep_memory_ptr(size_t index) const { return dependencies().at(index)->output_memory_ptr(); }
    memory& output_memory() const { return *_output; }
    memory::ptr output_memory_ptr() const { return _output; }
    size_t inputs_memory_count() const { return _node.get_primitive()->input_size(); }
    primitive_type_id type() const { return _node.type(); }
    primitive_id id() const { return _node.id(); }
    primitive_id org_id() const { return _node.get_org_primitive_id(); }
    bool can_be_optimized() const { return _node.can_be_optimized(); }
    std::shared_ptr<const primitive> desc() const { return _node.get_primitive(); }
    program_node const& get_node() const { return _node; }
    network_impl& get_network() const { return _network; }
    uint32_t get_network_id() const;
    void set_output_memory(memory::ptr mem);
    void check_memory_to_set(const memory& mem, const layout& layout) const;
    const std::list<const cldnn::program_node *>& get_users() const { return _node.get_users(); }

    // return pointer to const to prevent arbitrary 'execute' call -> use primitive_inst.execute() instead
    const primitive_impl* get_impl() const { return _impl.get(); }

    memory& input_memory(size_t index = 0) const {
        if (index >= inputs_memory_count())
            throw std::range_error("input offset too big");
        return dep_memory(index);
    }

    memory::ptr input_memory_ptr(size_t index = 0) const {
        if (index >= inputs_memory_count())
            throw std::range_error("input offset too big");
        return dep_memory_ptr(index);
    }

    event::ptr execute(const std::vector<event::ptr>& events);
    void init_kernels();
    void set_arguments();
    bool validate() const {
        if (_impl == nullptr)
            throw std::invalid_argument("[Internal cldnn error].  Validation method for nullptr impl is not allowed.");
        return _impl->validate(*this);
    }
    bool output_changed() const { return _output_changed; }
    void reset_output_change() { _output_changed = false; }

    void build_deps();

    memory::ptr fused_memory(size_t dep_id) const {
        return dep_memory_ptr(get_fused_mem_offset() + dep_id);
    }

    bool has_fused_primitives() const { return !_node.get_fused_primitives().empty(); }
    size_t get_fused_mem_count() const { return _node.get_fused_inputs_count(); }
    size_t get_fused_mem_offset() const { return _node.get_fused_primitives()[0].dep_start_idx; }

    bool has_mutable_input() const {
        return _has_mutable_input;
    }

    void set_mutable_input(bool val) {
        _has_mutable_input = val;
    }

    bool is_output() const {
        return _node.is_output();
    }

protected:
    primitive_inst(network_impl& network, program_node const& node, bool allocate_memory);

    network_impl& _network;
    program_node const& _node;

    std::unique_ptr<primitive_impl> _impl;

    // this is a set of dependencies in terms of memory, if execution of this primitive requires data from another one,
    // it should be added to this set
    std::vector<std::shared_ptr<primitive_inst>> _deps;

    // this is a set of dependencies in terms of execution
    // execution of all primitives from this set should be enough to guarantee that all memory deps (see _deps)
    // will be valid when executing this primitive. Most of the time this set will be equal to the _deps minus all
    // cldnn::data (which don't need to be execued) -- this is default, but it is also possible to have, for example,
    // only one fused primitive which will calculate multiple outputs (for example device enqueue can work in such
    // manner) in general - this member is introduced to relax logical connection between primitives which have to be
    // executed and memories which are used by this primitive
    std::vector<std::shared_ptr<primitive_inst>> _exec_deps;

    // _output is optional because its initialization might be postponed (reshape_inst may either allocate it's own
    // buffer or attach input as output
    // depending on reshape_node.is_in_place())
    memory::ptr _output;

    bool _output_changed;  // todo: implement output reuse if neither of inputs has changed
    bool _has_valid_input =
        true;  // by default all primitives has valid inputs, exception is input_layout (see input_layout_inst)
    bool _has_mutable_input = false;

    memory::ptr allocate_output();
    static std::vector<std::shared_ptr<primitive_inst>> build_exec_deps(
        std::vector<std::shared_ptr<primitive_inst>> const& mem_deps);

    // event function called by primitive_inst::execute after checking if primitive should rerun and before calling
    // _impl->execute() mainly for reshape (to update output memory if reshape_node.is_in_place() == true)
    virtual void on_execute() {}

    static std::string generic_to_string(program_node const& node, const char* type_name);
};

/*
Base class for all implementation of specified primitive type.
For example, all cpu convolution implementations should derive directly from typed_primitive_impl<convolution>.
GPU implementations should derive from typed_primitive_gpu_impl<convolution>;
*/
template <class PType>
struct typed_primitive_impl : public primitive_impl {
    static_assert(meta::is_primitive<PType>::value,
                  "PType should be a non-const, non-volatile class derived from primitive");

    using primitive_impl::primitive_impl;

private:
    event::ptr execute(const std::vector<event::ptr>& event,
                            primitive_inst& instance) override {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to execute primitive implementation with mismatching primitive instance");

        return execute_impl(event, reinterpret_cast<typed_primitive_inst<PType>&>(instance));
    }

    void set_arguments(primitive_inst& instance) override {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to set_arguments for primitive implementation with mismatching primitive instance");

        return set_arguments_impl(reinterpret_cast<typed_primitive_inst<PType>&>(instance));
    }


    virtual void set_arguments_impl(typed_primitive_inst<PType>& /*instance*/) {}
    virtual event::ptr execute_impl(const std::vector<event::ptr>& event,
                                         typed_primitive_inst<PType>& instance) = 0;

    bool validate(const primitive_inst& instance) const override {
        if (instance.type() != PType::type_id())
            throw std::invalid_argument("Implementation type does not match primitive type");
        if (instance.get_impl() != this)
            throw std::invalid_argument(
                "Trying to validate primitive implementation with mismatching primitive instance");

        return validate_impl(reinterpret_cast<const typed_primitive_inst<PType>&>(instance));
    }
    virtual bool validate_impl(const typed_primitive_inst<PType>&) const { return true; }
};

template <class PType>
class typed_primitive_inst_base : public primitive_inst {
public:
    using typed_node = typed_program_node<PType>;
    using typed_impl = typed_primitive_impl<PType>;

    const typed_node& node;
    const PType& argument;

    typed_primitive_inst_base(network_impl& network, typed_node const& node)
        : typed_primitive_inst_base(network, node, do_allocate_memory(node)) {}

protected:
    typed_primitive_inst_base(network_impl& network, typed_node const& node, bool allocate_memory)
        : primitive_inst(network, node, allocate_memory), node(_node), argument(*node.get_primitive()) {}

    typed_primitive_inst_base(network_impl& network, typed_node const& node, memory::ptr buffer)
        : typed_primitive_inst_base(network, node, false) {
        _output = buffer;
    }

private:
    bool do_allocate_memory(typed_node const& typ_node) {
        if (typ_node.template have_user_with_type<concatenation>() && typ_node.get_users().size() == 1 &&
            typ_node.get_users().front()->can_be_optimized()) {  // check if the only user is concat
            return false;
        }
        return true;
    }
};

/*
    Template class which represents instance of primitive 'PType'.
    Each new primitive should explicitly specialize this class.
    The pattern is as follows:
        struct new_primitive {}; // C++ API layer
        template <>
        class typed_primitive_inst<new_primitive> : public typed_primitive_inst_base<new_primitive> {}; // network
   instance specialization using new_primitive_inst = typed_primitive_inst<new_primitive>; //to simplify usage

    Using template specialization instead of dedicated classes for each primitive comes in hand
    when writing other template methods/classes which would like to use primitive_inst.
    As alternative to this, one could use some kind of type traits to translate primitive type
    to related primitive_inst implementation but this approach does the same with less code/classes.
*/
template <class PType>
class typed_primitive_inst : public typed_primitive_inst_base<PType> {
    static_assert(meta::always_false<PType>::value, "Missing typed_primitive_inst specialization");
};

}  // namespace cldnn
