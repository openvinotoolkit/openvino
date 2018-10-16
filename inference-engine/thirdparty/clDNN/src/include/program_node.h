/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include <set>

#include "api/CPP/primitive.hpp"
#include "internal_primitive.h"

#include "to_string_utils.h"
#include "json_object.h"
#include "xml_object.h"
#include "meta_utils.h"

namespace cldnn
{

struct program_impl;

template <class T>
struct typed_program_node;

template <class PType>
struct internal_primitive_type_base;

/*
    Base class for all primitives which wraps API class and extends it to be used
    in graph context.

    Besides primitive description provided by user, this class includes functionality to
    ask for direct predecessors and succesors as well as takes care of changes to primitive
    which would affect other graph's nodes (the most commont case is probably calculating output layout).

    At graph level, all connections between nodes are directly stored inside program_nodes - in oposite
    to API level where all primitives store only ids of related ones.
*/
struct program_node
{
    friend struct program_impl;
    friend class constants_propagator;

    template <class PType>
    friend struct typed_program_node;

    program_node(std::shared_ptr<primitive> prim, program_impl& prog);

    program_node(program_node const&) = delete;

public:
    virtual const primitive_id& id() const { return desc->id; }
    virtual primitive_type_id type() const { return desc->type; }

    template <class PType>
    bool is_type() const
    {
        static_assert(meta::is_primitive_v<PType>, "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return type() == PType::type_id();
    }

    auto& get_program() { return myprog; }
    auto const& get_program() const { return myprog; }

    auto get_selected_impl() const { return selected_impl; }

    auto const& get_dependencies() const { return dependencies; }
    auto& get_dependency(size_t idx) const { return *dependencies.at(idx); }

    //replaces idx-th dependency of 'this' with 'new_dep', calls program::remove_if_dangling(old_dep, detach_whole_branch)
    void replace_dependency(size_t idx, program_node& new_dep, bool detach_whole_branch = false);
    //searches for 'old_dep' in dependencies list of 'this' and replaces it with 'new_dep', calls program::remove_if_dangling(old_dep, detach_whole_branch)
    void replace_dependency(program_node const& old_dep, program_node& new_dep, bool detach_whole_branch = false);

    std::vector<primitive_id> get_dependencies_ids() const;

    void remove_dependency(size_t idx);
    void remove_dependency(program_node& node);

    std::set<primitive_id> get_memory_dependencies() const;
    void add_memory_dependency(primitive_id);
    void add_memory_dependency(std::vector<primitive_id>);

    template<class PType>
    bool have_user_with_type() const
    {
        for (auto const& usr : users)
        {
            if (usr->is_type<PType>()) return true;
        }
        return false;
    }

    bool is_detached(bool whole_branch = false);

    auto const& get_users() { return users; }
    // for const method, add const to stored successors/predecessors
    auto const& get_users() const { return reinterpret_cast<const std::list<const program_node*>&>(users); }

    bool has_next() const;
    program_node* get_next() { auto itr = processing_itr; return (*++itr); }
    const program_node* get_next() const { auto itr = processing_itr; return (*++itr); }

    json_composite desc_to_json() const;
	xml_composite desc_to_xml() const;
    //do not modify primitive directly to keep synchronisation wit graph
    std::shared_ptr<const primitive> get_primitive() const { return desc; }
    //primitive modification functions
    void set_output_padding(padding const& padd)
    {
        //changing output padding shouldn't cause any changes to other primitives
        //so just change it
        output_layout.data_padding = padd;
    }

    void merge_output_padding(padding const& padd)
    {
        set_output_padding(padding::max(padd, output_layout.data_padding));
    }

    //only calculated output layout (for external usage), does not modify/use cached output layout nor invalidate users
    layout calc_output_layout() const;

    //uses cached output layout if vlid, if not calls 'calc_output_layout' and stores its result + invalidate all users if layout has changed and @p invalidate_users_if_changed is set to true
    layout get_output_layout(bool invalidate_users_if_changed = true);
    //returns cached output layout if valid, otherwise throws an exception
    layout get_output_layout() const;
    //returns result of get_output_layout without padding
    layout get_non_padded_output_layout(bool invalidate_users_if_changed = true);

    //sets cached output layout to an arbitrary value, invalidates users if new layout differs from previous one and @p invalidate_users_if_changed is set to true
    //returns whether output layout has changed
    bool set_output_layout(layout new_layout, bool invalidate_users_if_changed = true);

    //forces recalculation of cached output layout, invalidates users if new layout is different than previous one and @p invalidate_users_if_changed is set to true
    //returns whether output layout has changed
    bool recalc_output_layout(bool invalidate_users_if_changed = true);

    bool is_padded() { return static_cast<bool>(get_output_layout().data_padding); }
    bool is_padded() const { return static_cast<bool>(get_output_layout().data_padding); }

    bool has_padded_dependency();
    bool has_padded_dependency() const;

    auto is_input() const { return dependencies.empty(); }
    auto is_endpoint() const { return users.empty(); }
    auto set_output(bool out) { output = out; }
    auto is_output() const { return output; }

    auto is_valid_output_layout() const { return valid_output_layout; }
    auto get_processing_num() const { return processing_num; }

    uint8_t mark(uint8_t val = 1) { uint8_t ret = user_mark; user_mark = val; return ret; }
    void unmark() { user_mark = 0; }
    auto is_marked() const { return user_mark != 0; }
    auto is_marked(uint8_t val) const { return user_mark == val; }
    uint8_t get_user_mark() const { return user_mark; }

    void set_fused_activation(cldnn_activation_func activation_func, cldnn_activation_additional_params additional_params)
    {
        fused_activation.activation_func = activation_func;
        fused_activation.additional_params = additional_params;
    }

    cldnn_activation_func get_fused_activation_func() const
    {
        return fused_activation.activation_func;
    }

    cldnn_activation_additional_params get_fused_activation_params() const
    {
        return fused_activation.additional_params;
    }

    auto can_be_optimized() const { return optimized; }
    void can_be_optimized(bool opt) { optimized = opt; }

    primitive_id get_org_primitive_id() const { return org_id; }
    void set_org_primitive_id(primitive_id org_prim_id) 
    {
        org_id = org_prim_id;
    }

    // returns immidiate dominator of this node if it's not its direct predecessor, otherwise returns nullptr
    program_node* get_dominator() { return dominator; }
    const program_node* get_dominator() const { return dominator; }

    //returns joint point associated with this node,
    //if the node is not a split point (i.e. it has exactly one user) this function returns nullptr,
    //otherwise returns pointer to a node which immidiately post-dominates this
    program_node* get_joint() { return joint; }
    const program_node* get_joint() const { return joint; }

    bool is_joint_point() const { return dominator != nullptr; }
    bool is_split_point() const { return joint != nullptr; }

    bool is_constant() const { return constant; }
    bool has_non_const_user() const { return (!constant || constant_frontier); }

    //returns true if all paths from network's source to sink must come through this node
    //(i.e. if this is a dominator of all the outputs)
    //a source, in this context, is defined as an input which lies within a data flow (see is_in_data_flow)
    bool is_in_main_branch() const { return main_branch; }

    //returns true if this node is within main data flow of the network (i.e. it does not describe helper data like convolution's weights etc.)
    bool is_in_data_flow() const { return data_flow; }

    //conversion from generic to specific
    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To>& as()
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To>&>(*this);
    }

    template <class To, class..., class = std::enable_if_t<!std::is_same<To, primitive>::value>>
    typed_program_node<To> const& as() const
    {
        if (type() != To::type_id())
            throw std::invalid_argument("program_node: mismatching primitive's type");

        return reinterpret_cast<typed_program_node<To> const&>(*this);
    }

    template <class To>
    operator typed_program_node<To>& ()
    {
        return as<To>();
    }

    template <class To>
    operator typed_program_node<To> const& () const
    {
        return as<To>();
    }

    void set_reused_memory_color(uint32_t color) const
    {
        has_reused_memory = true;
        reused_memory_color = color;
    }

    bool is_reusing_memory() { return has_reused_memory; };
    uint32_t get_reused_memory_color() { return reused_memory_color; ; }

protected:
    std::shared_ptr<primitive> desc;
    program_impl& myprog;

    std::shared_ptr<primitive_impl> selected_impl;

    bool valid_output_layout = false;
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::vector<program_node*> dependencies;
    std::list<program_node*> users;

    std::list<program_node*>::const_iterator processing_itr;
    uint32_t processing_num = 0;

    // list of primitives that can reuse same memory buffers due to execution order conflicts
    std::set<primitive_id> memory_dependencies;  

    program_node* dominator = nullptr;
    program_node* joint = nullptr;
    bool constant = false;
    bool constant_frontier = false;

    bool main_branch = true;
    bool data_flow = false;

    bool output = false;
    uint8_t user_mark = 0;

    bool optimized = false;

    mutable bool has_reused_memory = false;
    mutable uint32_t reused_memory_color = 0;

    primitive_id org_id = "";

    struct fused_activation_params
    {
        cldnn_activation_func activation_func = activation_none;
        cldnn_activation_additional_params additional_params = { 0.0f, 0.0f };
    };

    fused_activation_params fused_activation;

    void invalidate_users() const;
};

namespace details
{
    template <class PType>
    struct api_typed_program_node_base : public program_node
    {
        static_assert(meta::is_api_primitive_v<PType>, "PType should name a non-const, non-volatile type derived from cldnn::primitive but not from cldnn::internal_primitive");
        friend struct cldnn::program_impl;

    public:
        using program_node::program_node;

        std::shared_ptr<const PType> get_primitive() const { return std::static_pointer_cast<const PType>(program_node::get_primitive()); }

    protected:
        std::shared_ptr<PType> typed_desc() const { return std::static_pointer_cast<PType>(desc); }
    };

    struct internal_program_node_base : public program_node
    {
        friend struct cldnn::program_impl;

        internal_program_node_base(program_impl& prog);

        const primitive_id& id() const override { return internal_id; }

        void set_implementation(std::unique_ptr<primitive_impl>&& impl);

    private:
        primitive_id internal_id;

        static primitive_id get_next_internal_id();
    };

    template <class PType>
    struct internal_typed_program_node_base : public internal_program_node_base
    {
        static_assert(meta::is_internal_primitive_v<PType>, "PType should name a non-const, non-volatile type derived from cldnn::internal_primitive");

    public:
        using internal_program_node_base::internal_program_node_base;

        primitive_type_id type() const override { return PType::type_id(); }

        template <class... Guard>
        [[noreturn]]
        void get_primitive(Guard&&...)
        {
            static_assert(meta::always_false_v<meta::pack<Guard...>>, "Trying to get primitive from internal node");
        }


    protected:
        template <class... Guard>
        [[noreturn]]
        void typed_desc(Guard&&...)
        {
            static_assert(meta::always_false_v<meta::pack<Guard...>>, "Trying to get primitive from internal node");
        }
    };
}

/*
Template class used to indicate that usage context requires 'program_node' to wrap primitive
of type 'PType'. Successful conversion from 'program_node' to 'typed_program_node<PType>' means
that this restriction in fact holds and functions/method/etc. may saftly use uderlaying primitive.

This class shadows 'get_primitive' method from base class which now returns pointer to more specific
type.
*/
template <class PType>
using typed_program_node_base = std::conditional_t<meta::is_api_primitive_v<PType>, details::api_typed_program_node_base<PType>, details::internal_typed_program_node_base<PType>>;

/*
    Actual template class used in context which requires 'program_node' to wrap
    primitive of type 'PType'. This class is introduced to provide possibility of explicit specialization.
    In most cases such specializations would add accessors to make access to PType-specific fields easier.

    It's not required to specialize this class for new primitives types.
*/
template <class PType>
struct typed_program_node : public typed_program_node_base<PType>
{
    using typed_program_node_base<PType>::typed_program_node_base;

    decltype(auto) input() const { return program_node::get_dependency(0); }
};

}