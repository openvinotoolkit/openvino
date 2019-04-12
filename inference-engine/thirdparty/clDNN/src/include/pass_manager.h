/*
// Copyright (c) 2018 Intel Corporation
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

#include "program_impl.h"
#include "layout_optimizer.h"

namespace cldnn
{
    class base_pass
    {
        friend class pass_manager;
    public:
        base_pass(const std::string& pass_name) : name(pass_name) {}
        virtual void run(program_impl& p) = 0;
        std::string get_name() { return name; }
        void clean_marks(program_impl& p) {
            for (auto& node : p.get_processing_order())
            {
                node->unmark();
            }
        }
    private:
        const std::string name;
    };

    class pass_manager
    {
    public:
        pass_manager()
        {
            pass_count = 0;
        }
        void run(program_impl& p, base_pass& pass)
        {
            pass.run(p);
            std::string dump_file_name;
            if (pass_count < 10)
                dump_file_name += "0";
            dump_file_name += std::to_string(pass_count) + "_" + pass.get_name();
            p.dump_program(dump_file_name.c_str(), true);
            pass.clean_marks(p);
            pass_count++;
        }
        uint32_t get_pass_count() { return pass_count; }
        uint32_t inc_pass_count() { return ++pass_count; }
        ~pass_manager() {}
    private:
        uint32_t pass_count;
    };

    class add_required_reorders : public base_pass
    {
    public:
        add_required_reorders() : base_pass("add_required_reorders") {}
    private:
        virtual void run(program_impl& p) override;
        void add_reorder(program_impl& p, program_node* node, program_node* usr, layout reorder_layout);
    };

    class add_reshape_to_primitives : public base_pass
    {
    public:
        add_reshape_to_primitives() : base_pass("add_reshape_to_primitives_pass") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class calculate_prior_boxes : public base_pass
    {
    public: 
        calculate_prior_boxes() : base_pass("calculated_prior_boxes") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class compile_graph: public base_pass
    {
    public:
        compile_graph() : base_pass("compile_graph") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class eltwise_shrinking : public base_pass
    {
    public:
        eltwise_shrinking() : base_pass("eltwise_shrinking") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class eltwise_remove_stride : public base_pass
    {
    public:
        eltwise_remove_stride() : base_pass("eltwise_remove_stride") {}
    private:
        virtual void run(program_impl& p) override;
        void conv_stride_extend(program_impl & p, program_node & node, cldnn::tensor & tensor);
    };

    class graph_initializations : public base_pass 
    {
    public:
        graph_initializations() : base_pass("init") {}
    private:
        virtual void run(program_impl& p) override;
        void replace_nodes(program_impl& p);
        void handle_detection_output(program_impl& p);
        void handle_lstm(program_impl& p);
        void set_outputs(program_impl& p);  
    };

    class handle_input_padding : public base_pass
    {
    public:
        handle_input_padding() : base_pass("handle_input_padding") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class mark_nodes : public base_pass
    {
    public:
        mark_nodes() : base_pass("analyzed_graph") {}
    private:
        virtual void run(program_impl& p) override;
        void mark_constants(program_impl& p);
        void mark_data_flow(program_impl& p);
    };

    class prepare_buffer_fusing : public base_pass
    {
    public:
        prepare_buffer_fusing() : base_pass("prepare_buffer_fusing") {}
    private:
        virtual void run(program_impl& p) override;
    };

    class prepare_conv_eltw_fusing : public base_pass
    {
    public:
        prepare_conv_eltw_fusing() : base_pass("prepare_conv_eltw_fusing") {}
    private:
        virtual void run(program_impl& p) override;
        void fuse_conv_eltwise(program_impl& p, program_node* node);
    };

    class prepare_conv_eltw_read_write_opt : public base_pass
    {
    public:
        prepare_conv_eltw_read_write_opt() : base_pass("prepare_conv_eltw_read_write_opt") {}
    private:
        virtual void run(program_impl& p) override;
        void conv_eltwise_read_write_opt(program_impl& p, program_node* node);
    };

    class prepare_depthwise_sep_opt : public base_pass
    {
    public:
        prepare_depthwise_sep_opt() : base_pass("prepare_depthwise_sep_opt") {}
    private:
        virtual void run(program_impl& p) override;
        template <typename T> void optimize_depthwise_sep_pre(T& node);
    };

    class prep_opt_depthwise_sep_post : public base_pass
    {
    public:
        prep_opt_depthwise_sep_post() : base_pass("prep_opt_depthwise_sep_post") {}
    private:
        virtual void run(program_impl& p) override;
        template <typename T> void optimize_depthwise_sep_pre(program_impl& p, T& node);
    };

    class prepare_primitive_fusing : public base_pass
    {
    public:
        prepare_primitive_fusing() : base_pass("prepare_primitive_fusing") {}
    private:
        virtual void run(program_impl& p) override;
        void fuse_skip_layers(program_impl& p, program_node* node);
        void fuse_conv_bn_scale(program_impl& p, program_node* node);
    };

    class pre_optimize_bias : public base_pass
    {
    public:
        pre_optimize_bias(layout_optimizer& lo_ref);
    private:
        virtual void run(program_impl& p) override;
        virtual void run(program_impl& p, layout_optimizer& lo);
        template <typename T>
        void optimize_bias(T& node, layout_optimizer& lo, program_impl& p);
        layout_optimizer& _lo;
    };

    class prepare_padding : public base_pass
    {
    public:
        prepare_padding(bool output_size_handling_enabled_switch) : base_pass("prepare_padding"),
            output_size_handling_enabled(output_size_handling_enabled_switch) {}
    private:
        virtual void run(program_impl& p) override;
        bool output_size_handling_enabled;
    };

    class post_optimize_weights : public base_pass
    {
    public:
        post_optimize_weights(layout_optimizer& lo_ref);
    private:
        virtual void run(program_impl& p) override;
        virtual void run(program_impl& p, layout_optimizer& lo);
        template <typename T>
        void optimize_weights(T& node, layout_optimizer& lo, program_impl& p);
        layout_optimizer& _lo;
    };

    class propagate_constants : public base_pass
    {
    public:
        propagate_constants() : base_pass("propagate_constants") {}
    private:
        virtual void run(program_impl& p) override;
        std::list<std::pair<primitive_id, memory_impl::ptr>> calculate(engine_impl &engine);
        bool has_non_const_user(program_node& node) const;
        void handle_constant(program_impl& prog, program_node& node);
        void add_constant(program_impl& prog, program_node& node);
        void add_deps_to_tpl(program_impl& prog, const std::vector<program_node*>& node);

        bool has_non_trivial_constants = false;
        std::list<typed_program_node<data>*> const_inputs;
        std::vector<primitive_id> const_outputs;
        std::set<std::shared_ptr<program_node>> nodes;
    };

    class remove_redundant_reorders : public base_pass
    {
    public:
        remove_redundant_reorders() : base_pass("remove_redundant_reorders") {}
        virtual void run(program_impl& p) override;
    };

    class reorder_inputs : public base_pass
    {
    public:
        reorder_inputs(layout_optimizer& lo_ref);
    private:
        virtual void run(program_impl& p) override;
        virtual void run(program_impl& p, layout_optimizer& lo);
        layout_optimizer& _lo;
    };

    class trim_to_outputs : public base_pass
    {
    public:
        trim_to_outputs() : base_pass("trimmed") {}
    private:
        virtual void run(program_impl& p) override;
    };
}