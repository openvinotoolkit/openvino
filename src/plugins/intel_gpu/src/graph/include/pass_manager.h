// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/program.hpp"
#include "layout_optimizer.h"
#include "split_inst.h"
#include "lstm_inst.h"
#include "lstm_dynamic_inst.h"
#include "quantize_inst.h"
#include "eltwise_inst.h"
#include "convolution_inst.h"
#include "permute_inst.h"
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <utility>
#include <set>
#include <functional>

#include <fstream>

#define GPU_DEBUG_LOG_PASS    GPU_DEBUG_LOG << "[" << get_name() << "] "

namespace cldnn {
class base_pass {
    friend class pass_manager;

public:
    explicit base_pass(const std::string& pass_name) : name(pass_name) {}
    virtual void run(program& p) = 0;
    std::string get_name() { return name; }

private:
    const std::string name;
};

class pass_manager {
public:
    explicit pass_manager(program& p);
    void run(program& p, base_pass& pass);
    uint32_t get_pass_count() { return pass_count; }
    uint32_t inc_pass_count() { return ++pass_count; }
    ~pass_manager() {}

private:
    uint32_t pass_count;
    std::ofstream graph_opt_log;
};

class add_required_reorders : public base_pass {
public:
    add_required_reorders() : base_pass("add_required_reorders") {}

private:
    void run(program& p) override;
    void add_reorder(program& p, program_node* node, program_node* usr, bool keep_original_dt = false);
};

class add_reshape_to_primitives : public base_pass {
public:
    add_reshape_to_primitives() : base_pass("add_reshape_to_primitives_pass") {}

private:
    void run(program& p) override;
};

class compile_graph : public base_pass {
public:
    compile_graph() : base_pass("compile_graph") {}

private:
    void run(program& p) override;
};

class graph_initializations : public base_pass {
public:
    graph_initializations() : base_pass("init") {}

private:
    void run(program& p) override;
    void handle_split_node(program& p, split_node& node);
    void handle_lstm_node(program& p, lstm_node& node);
    void handle_dynamic_lstm_node(program& p, lstm_dynamic_node& node);
    void set_outputs(program& p);
};

class handle_reshape : public base_pass {
public:
    handle_reshape() : base_pass("handle_reshape") {}

private:
    void run(program& p) override;
};

class mark_nodes : public base_pass {
public:
    mark_nodes() : base_pass("analyzed_graph") {}

private:
    void run(program& p) override;
};

class clamp_fp16_output : public base_pass {
public:
    clamp_fp16_output() : base_pass("clamp_fp16_output") {}

private:
    void run(program& p) override;
};

class mark_shape_of_subgraphs : public base_pass {
    // This optimization pass aggregates nodes into shape_of subgraphs for further optimizations.
    // There are few key requirements to decide if node belongs to shape_of subgraph or not:
    // - Node type is shape_of OR
    // - All node's dependencies are marked as members of shape_of subgraphs OR
    // - Node is a shape infer dependency of any user
    // Also, there is some additional requirement:
    // - Primitive must have CPU implementation (this requirement is ignored for reshape
    //   primitives, since currently ocl optimized_out implementation is used for reshape execution in such subgraphs)
public:
    mark_shape_of_subgraphs(bool update_impls = false) :
        base_pass("mark_shape_of_subgraphs"), _update_impls(update_impls) {}

private:
    void run(program& p) override;
    void look_for_shape_of_subgraph(program_node& node);
    bool can_mark_node(const program_node& node);
    void mark_node(program_node& node);

    bool _update_impls;
};

class prepare_buffer_fusing : public base_pass {
public:
    prepare_buffer_fusing() : base_pass("prepare_buffer_fusing") {}

private:
    void run(program& p) override;
};

class prepare_quantization : public base_pass {
public:
    prepare_quantization() : base_pass("prepare_quantization") {}

private:
    void run(program& p) override;
    void handle_quantize_node(program& p, quantize_node& quantize_node);
    void prepare_dequantize_merge(program& p, eltwise_node& eltwise_node);
    void remove_fake_reorders(program& p, reorder_node& reorder_node);
    void prepare_asymmetric_quantization(program& p, convolution_node& convolution_node);
    void prepare_scale_shift_opt(program &p, quantize_node& quantize_node);
    bool optimize_quantize(program &p, quantize_node& quantize_node);
};

class prepare_conv_eltw_fusing : public base_pass {
public:
    explicit prepare_conv_eltw_fusing(layout_optimizer& lo_ref, bool b_fs_yx_fsv16_opt = false) :
        base_pass("prepare_conv_eltw_fusing"), _lo(lo_ref), b_fs_yx_fsv16_opt(b_fs_yx_fsv16_opt) {}

private:
    void run(program& p) override;
    void fuse_conv_eltwise(program& p, program_node* node);
    void fuse_conv_depth_to_space(program& p, program_node* node);
    layout_optimizer& _lo;
    bool b_fs_yx_fsv16_opt;
};

class prepare_conv_eltw_read_write_opt : public base_pass {
public:
    prepare_conv_eltw_read_write_opt() : base_pass("prepare_conv_eltw_read_write_opt") {}

private:
    void run(program& p) override;
    void conv_eltwise_read_write_opt(program& p, program_node* node);
};

class prepare_primitive_fusing_through : public base_pass {
public:
    prepare_primitive_fusing_through() : base_pass("prepare_primitive_fusing_through") {}
    void run(program& p) override;
};

class prepare_primitive_fusing : public base_pass {
public:
    explicit prepare_primitive_fusing(layout_optimizer& lo_ref) :
        base_pass("prepare_primitive_fusing"), _lo(lo_ref) {}

private:
    void run(program& p) override;
    void fuse_sigmoid_mul_to_swish(program &p);
    void fuse_bias(program &p);
    void fuse_reorders(program& p);
    void fuse_simple_primitives(program &p);
    void fuse_constant_transposes(program &p);
    void optimize_fused_ops(program &p);
    void remove_redundant_reshape(program &p);
    layout_optimizer& _lo;
};

class pre_replace_deconv : public base_pass {
public:
    explicit pre_replace_deconv(layout_optimizer& lo_ref) :
        base_pass("pre_replace_deconv"), _lo(lo_ref) {}

private:
    void run(program& p) override;
    layout_optimizer& _lo;
};

class prepare_padding : public base_pass {
public:
    explicit prepare_padding(bool output_size_handling_enabled_switch)
        : base_pass("prepare_padding"), output_size_handling_enabled(output_size_handling_enabled_switch) {}

private:
    void run(program& p) override;
    bool output_size_handling_enabled;
};

class post_input_reorder : public base_pass {
public:
    post_input_reorder() : base_pass("post_input_reorder") {}

private:
    void run(program& p) override;
    program_node& add_reorder(program& p, program_node* node, program_node* usr, const layout& reorder_layout);
};

class post_optimize_weights : public base_pass {
public:
    explicit post_optimize_weights(reorder_factory& rf_ref);

private:
    struct weights_bias_offset {
        size_t weights_offset;
        size_t bias_offset;

        // When using this ctor weights offset is added to the bias_offset
        weights_bias_offset(const size_t w_offset, const size_t b_offset)
            : weights_offset(w_offset)
            , bias_offset(weights_offset + b_offset)
        {}
    };

    void run(program& p) override;
    template<typename T>
    weights_bias_offset get_weights_bias_offset(const T& node);
    template<typename T>
    void optimize_weights(T& node, program& p);
    reorder_factory& _rf;
};

class propagate_constants : public base_pass {
public:
    propagate_constants() : base_pass("propagate_constants") {}

private:
    void run(program& p) override;
    std::list<std::pair<primitive_id, memory::ptr>> calculate(engine& engine,
                                                              const ExecutionConfig& config,
                                                              std::shared_ptr<ov::threading::IStreamsExecutor> task_executor);
    bool has_non_const_user(program_node& node) const;
    void handle_constant(program& prog, program_node& node);
    void add_constant(program& prog, program_node& node);
    void add_deps_to_tpl(program& prog, const std::vector<std::pair<program_node*, int32_t>>& node);

    bool has_non_trivial_constants = false;
    std::list<typed_program_node<data>*> const_inputs;
    std::vector<primitive_id> const_outputs;
    std::set<std::shared_ptr<program_node>> nodes;
};

class remove_redundant_reorders : public base_pass {
public:
    explicit remove_redundant_reorders(layout_optimizer& lo_ref, bool enable_reorder_fusing = false, bool update_implementations = false,
        bool remove_output_reorders = false);
    void run(program& p) override;

private:
    layout_optimizer& lo;
    bool enable_reorder_fusing;
    bool update_implementations;
    bool remove_output_reorders;
};

class reorder_inputs : public base_pass {
public:
    reorder_inputs(layout_optimizer& lo_ref, reorder_factory& rf_ref);

private:
    void run(program& p) override;
    virtual void run(program& p, layout_optimizer& lo, reorder_factory& rf);
    layout_optimizer& _lo;
    reorder_factory& _rf;
};

class select_preferred_formats : public base_pass {
public:
    explicit select_preferred_formats(layout_optimizer& lo_ref) :
        base_pass("select_preferred_formats"), _lo(lo_ref) {}

private:
    void run(program& p) override;
    layout_optimizer& _lo;
};

class trim_to_outputs : public base_pass {
public:
    trim_to_outputs() : base_pass("trimmed") {}

private:
    void run(program& p) override;
};

class strided_slice_optimize : public base_pass {
public:
    strided_slice_optimize() : base_pass("strided_slice_optimize") {}
    void run(program& p) override;
};

class reverse_optional_nodes_outputs : public base_pass {
public:
    reverse_optional_nodes_outputs() : base_pass("reverse_optional_nodes_outputs") {}
    void run(program& p) override;
};

class concat_input_order : public base_pass {
    // This optimization changes order of inputs for concatenation to provide
    // better alignment for execution and allow for optimizing out in some cases.
    // For example concatenation along features with inputs [13, 1024] in format fsv16
    // has only first input aligned to feature blocks, blocking performant implementation
    // for second one.
    // This can be fixed by chaning order to [1024, 13] and fusing reshuffling of those features
    // into following layers, such as convolution or fully connected, where it can be
    // implemented as compile-time weights shuffling.
    //
    // Requirements - may work incorrectly if not fullfiled:
    // - formats are selected
    // - implementations aren't selected
    //
    // Soft requirements - reduce applicability if not fullfiled:
    // - constant primitives are reduced to data nodes
    // - no fused primitives
public:
    concat_input_order() : base_pass("concat_input_order") {}
    void run(program& p) override;
};

class memory_dependency_pass : public base_pass {
public:
    explicit memory_dependency_pass(const std::string& pass_name) : base_pass(pass_name) {}
    void add_memory_dependency(program_node* node, program_node* dep) {
        if (node->can_be_optimized() || !dep->can_be_optimized()) {
            node->add_memory_dependency(dep->id());
        } else {
            if (node->id() == dep->id()) {
                return;
            }
            for (const auto& subdep : dep->get_dependencies()) {
                add_memory_dependency(node, subdep.first);
                add_memory_dependency(subdep.first, node);
            }
        }
    }
};

class basic_memory_dependencies : public memory_dependency_pass {
public:
    basic_memory_dependencies() : memory_dependency_pass("basic_memory_dependencies") {}
    void run(program& p) override;
};

class skipped_branch_memory_dependencies : public memory_dependency_pass {
public:
    skipped_branch_memory_dependencies() : memory_dependency_pass("skipped_branch_memory_dependencies") {}
    void run(program& p) override;
};

class oooq_memory_dependencies : public memory_dependency_pass {
public:
    oooq_memory_dependencies() : memory_dependency_pass("oooq_memory_dependencies") {}
    void run(program& p) override;
};

class update_inner_program_io_map : public base_pass {
public:
    update_inner_program_io_map() : base_pass("update_inner_program_io_map") {}

private:
    void run(program& p) override;
};

class add_onednn_optimization_attributes : public base_pass {
public:
    add_onednn_optimization_attributes() : base_pass("add_onednn_optimization_attributes") {}
    void run(program& p) override;
};

class build_implementations : public base_pass {
public:
    build_implementations() : base_pass("build_implementations") {}
    void run(program& p) override;
};

class reorder_transfer : public base_pass {
public:
    reorder_transfer() : base_pass("reorder_transfer") {}

private:
    void run(program& p) override;
};

}  // namespace cldnn
