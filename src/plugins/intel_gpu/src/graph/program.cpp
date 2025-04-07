// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/util/weights_path.hpp"

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/graph/program.hpp"

#include "layout_optimizer.h"
#include "pass_manager.h"
#include "primitive_type.h"
#include "program_dump_graph.h"
#include "program_node.h"
#include "sliding_window_utils.hpp"
#include "program_helpers.h"

#include "matrix_nms_inst.h"
#include "roi_pooling_inst.h"
#include "reorg_yolo_inst.h"
#include "eltwise_inst.h"
#include "non_zero_inst.h"
#include "softmax_inst.h"
#include "permute_inst.h"
#include "custom_gpu_primitive_inst.h"
#include "resample_inst.h"
#include "reshape_inst.h"
#include "ctc_loss_inst.hpp"
#include "group_normalization_inst.h"
#include "quantize_inst.h"
#include "activation_inst.h"
#include "depth_to_space_inst.h"
#include "convolution_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "deconvolution_inst.h"
#include "detection_output_inst.h"
#include "generate_proposals_inst.h"
#include "experimental_detectron_generate_proposals_single_image_inst.hpp"
#include "input_layout_inst.h"
#include "shuffle_channels_inst.h"
#include "arg_max_min_inst.h"
#include "dft_inst.h"
#include "multiclass_nms_inst.h"
#include "mutable_data_inst.h"
#include "pooling_inst.h"
#include "border_inst.h"
#include "primitive_inst.h"
#include "prior_box_inst.h"
#include "scatter_elements_update_inst.h"
#include "proposal_inst.h"
#include "reorder_inst.h"
#include "mvn_inst.h"
#include "gemm_inst.h"
#include "adaptive_pooling_inst.h"
#include "reduce_inst.h"
#include "region_yolo_inst.h"
#include "strided_slice_inst.h"
#include "loop_inst.h"
#include "reverse_inst.h"
#include "unique_inst.hpp"
#include "condition_inst.h"
#include "scaled_dot_product_attention_inst.h"
#include "to_string_utils.h"
#include "intel_gpu/graph/serialization/map_serializer.hpp"

#include "intel_gpu/primitives/rnn.hpp"

// TODO: Remove once we have interface for kernels cache
#include "impls/ocl/kernels_cache.hpp"

// TODO: implement self-registration for impls
#include "impls/ocl/register.hpp"
#include "impls/cpu/register.hpp"
#include "impls/common/register.hpp"

#include "kernel_base.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <unordered_set>

#ifdef __unix__
#include <sys/resource.h>
#endif

using namespace cldnn;
using namespace ov::intel_gpu;

static ov::threading::IStreamsExecutor::Config make_task_executor_config(const ExecutionConfig& config, std::string tags, int num_streams = 0) {
    int streams = (num_streams > 0) ? num_streams : config.get_compilation_num_threads();
    auto priority = config.get_host_task_priority();
    auto core_type = ov::hint::SchedulingCoreType::ANY_CORE;
    switch (priority) {
        case ov::hint::Priority::LOW: core_type = ov::hint::SchedulingCoreType::ECORE_ONLY; break;
        case ov::hint::Priority::MEDIUM: core_type = ov::hint::SchedulingCoreType::ANY_CORE; break;
        case ov::hint::Priority::HIGH: core_type = ov::hint::SchedulingCoreType::PCORE_ONLY; break;
        default: OPENVINO_ASSERT(false, "[GPU] Can't create task executor: invalid host task priority value: ", priority);
    }
    bool enable_cpu_pinning = config.get_enable_cpu_pinning();

    ov::threading::IStreamsExecutor::Config task_executor_config(tags,
                                                                 streams,
                                                                 1,
                                                                 core_type,
                                                                 false,
                                                                 enable_cpu_pinning);

    return task_executor_config;
}

std::shared_ptr<ov::threading::IStreamsExecutor> program::make_task_executor(const ExecutionConfig& config) {
    ov::threading::IStreamsExecutor::Config task_executor_config = make_task_executor_config(config, "CPU Tasks executor for GPU plugin");
    return std::make_shared<ov::threading::CPUStreamsExecutor>(task_executor_config);
}

std::shared_ptr<ICompilationContext> program::make_compilation_context(const ExecutionConfig& config) {
    const int _num_async_build_threads = 1;
    return ICompilationContext::create(make_task_executor_config(config,
                                                                 "Task executor config for CompilationContext in GPU plugin", _num_async_build_threads));
}

program::program(engine& engine_ref,
                 topology const& topology,
                 const ExecutionConfig& config,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                 std::shared_ptr<ICompilationContext> compilation_context,
                 bool is_internal,
                 bool no_optimizations,
                 bool is_body_program)
    : _engine(engine_ref),
      _stream(_engine.create_stream(config)),
      _config(config),
      _task_executor(std::move(task_executor)),
      processing_order(),
      is_internal(is_internal),
      _is_body_program(is_body_program),
      _compilation_context(compilation_context) {
    init_primitives();
    _config.finalize(_engine);
    GPU_DEBUG_INFO << "Program config\n" << _config.to_string();
    init_program();
    prepare_nodes(topology);
    program_node::reset_unique_id();
    if (no_optimizations) {
        init_graph();
    } else {
        build_program(is_internal);
        if (_is_body_program) {
            // To skip empty if (condition) subgraph
            bool can_be_optimized = true;
            for (auto& node : processing_order) {
                if (node->is_type<input_layout>()) {
                    continue;
                } else if (node->is_type<data>()) {
                    continue;
                } else if (node->is_output() && node->is_type<reorder>() && !node->has_fused_primitives() &&
                      node->get_input_layout(0).data_type == node->get_output_layouts(false)[0].data_type &&
                      node->get_input_layout(0).format == node->get_output_layouts(false)[0].format &&
                      node->get_input_layout(0).get_partial_shape().size() == node->get_output_layouts(false)[0].get_partial_shape().size()) {
                    continue;
                }
                can_be_optimized = false;
                break;
            }
            this->_can_be_optimized = can_be_optimized;
        }
    }
}

program::program(engine& engine_ref,
                 std::set<std::shared_ptr<program_node>> const& nodes,
                 const ExecutionConfig& config,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                 bool is_internal)
    : _engine(engine_ref),
      _stream(_engine.create_stream(config)),
      _config(config),
      _task_executor(std::move(task_executor)),
      processing_order(),
      is_internal(is_internal) {
    _config.finalize(_engine);
    init_primitives();
    init_program();
    prepare_nodes(nodes);
    build_program(is_internal);
}

program::program(engine& engine, const ExecutionConfig& config)
    : _engine(engine),
      _stream(_engine.create_stream({})),
      _config(config),
      processing_order() {
    init_primitives();
    _config.finalize(_engine);
    new_shape_infer = _config.get_allow_new_shape_infer();
    _layout_optimizer = std::make_unique<layout_optimizer>();
}

program::~program() {
}

void program::init_program() {
    set_options();

    pm = std::unique_ptr<pass_manager>(new pass_manager(*this));
    new_shape_infer = _config.get_allow_new_shape_infer();

    if (_task_executor == nullptr)
        _task_executor = program::make_task_executor(_config);
    _kernels_cache = std::unique_ptr<kernels_cache>(new kernels_cache(_engine, _config, prog_id, _task_executor,
                                                                      kernel_selector::KernelBase::get_db().get_batch_headers()));

    _kernels_cache->set_kernels_reuse(_config.get_enable_kernels_reuse());

    if (!_compilation_context)
        _compilation_context = program::make_compilation_context(_config);


    _layout_optimizer = std::make_unique<layout_optimizer>();
    _impls_cache = std::make_unique<ImplementationsCache>(get_config().get_impls_cache_capacity());
    // Remove items of compilation context's internal queue when some impl is popped in kernels_cache
    // compilation context's queue check duplication of inserted task
    _impls_cache->set_remove_item_callback([this](ImplementationsCache::ItemType& item) {
        get_compilation_context().remove_keys({item.first});
    });
}

void program::init_primitives() {
    // Register implementations in order of their selection priority: common, OCL, oneDNN, CPU
    // We register OCL implementation before oneDNN, because oneDNN is not always preferable (in case of iGPU)
    // This order will only apply to primitives with preferrable implementation type equal to impl_types::any

    static bool is_initialized = false;
    if (!is_initialized) {
        common::register_implementations();
        ocl::register_implementations();
        cpu::register_implementations();
        is_initialized = true;
    }
}

kernels_cache& program::get_kernels_cache() const {
    return *_kernels_cache;
}

program::ptr program::build_program(engine& engine,
                                    const topology& topology,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal,
                                    bool no_optimizations,
                                    bool is_body_program) {
    return std::make_shared<program>(engine, topology, config, task_executor, nullptr, is_internal, no_optimizations, is_body_program);
}

program::ptr program::build_program(engine& engine,
                                    const topology& topology,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    std::shared_ptr<ICompilationContext> compilation_context,
                                    bool is_internal,
                                    bool no_optimizations,
                                    bool is_body_program) {
    return std::make_shared<program>(engine, topology, config, task_executor, compilation_context, is_internal, no_optimizations, is_body_program);
}

program::ptr program::build_program(engine& engine,
                                    const topology& topology,
                                    const ExecutionConfig& config,
                                    bool is_internal,
                                    bool no_optimizations,
                                    bool is_body_program) {
    return std::make_shared<program>(engine, topology, config, nullptr, nullptr, is_internal, no_optimizations, is_body_program);
}

program::ptr program::build_program(engine& engine,
                                    const std::set<std::shared_ptr<program_node>>& nodes,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal) {
    return std::make_shared<program>(engine, nodes, config, task_executor, is_internal);
}

program_node& program::get_node(primitive_id const& id) {
    try {
        return *nodes_map.at(id);
    } catch (...) {
        throw std::runtime_error("Program doesn't contain primitive node: " + id);
    }
}

program_node const& program::get_node(primitive_id const& id) const {
    try {
        return *nodes_map.at(id);
    } catch (...) {
        throw std::runtime_error("Program doesn't contain primitive node: " + id);
    }
}

// TODO: Remove once we will get full support for input/output padding in all primitive implementations.
bool program::analyze_output_size_handling_need() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("analyze_output_size_handling_need");
    bool handling_needed = false;

    // Calculate output size and compare with specified.
    for (const auto& node : processing_order) {
        if (node->is_type<deconvolution>()) {
            auto& prim_node = node->as<deconvolution>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            auto filter_size = prim_node.weights().get_output_layout().get_tensor();

            auto primInputSize = prim_node.get_input_layout().get_tensor();
            auto calc_output_range = calc_sliding_window_needed_input_range(primInputSize,
                                                                            filter_size,
                                                                            prim->pad,
                                                                            prim->stride,
                                                                            ov::Strides(prim->stride.size(), 1),
                                                                            true,
                                                                            1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        } else if (node->is_type<pooling>()) {
            auto& prim_node = node->as<pooling>();
            const auto& prim = prim_node.get_primitive();

            if (!prim->with_output_size)
                continue;

            tensor specified_output_range(
                {0, 0, prim->output_size.spatial[0], prim->output_size.spatial[1], prim->output_size.spatial[2]},
                1);

            tensor size(1);
            for (size_t i = 0; i < prim->size.size(); i++) {
                size.spatial[i] = static_cast<tensor::value_type>(prim->size[prim->size.size() - i - 1]);
            }
            // TODO: Check compatibility of output size calculation (with caffe).
            auto primInputSize = prim_node.get_input_layout().get_tensor();
            auto calc_output_range = calc_sliding_window_output_range<swor_mode::exceed_once_data>(
                primInputSize,
                size,
                ov::CoordinateDiff(prim->pads_begin.begin(), prim->pads_begin.end()),
                prim->stride,
                ov::Strides(prim->stride.size(), 1),
                true,
                1);

            if (specified_output_range != calc_output_range)
                handling_needed = true;
        }
    }

    return handling_needed;
}

// create new nodes for a program based on the set of nodes
// method created to be used by propagate_constants to build sub program from constant nodes
void program::prepare_nodes(std::set<std::shared_ptr<program_node>> const& nodes) {
    for (const auto& itr : nodes) {
        if (itr.get()->is_type<data>()) {
            get_or_create(std::make_shared<input_layout>(itr.get()->id(),
                                                         itr.get()->as<data>().get_primitive()->mem->get_layout()));
        } else {
            get_or_create(itr->desc);
        }
    }
    for (const auto& node : nodes_map) {
        auto node_ptr = node.second;
        if (node_ptr == nullptr)
            throw std::runtime_error("NULL pointer in nodes_map.");
        // ToDo: avoid O(n^2) run time here (pass map instead of set?)
        bool found = false;
        for (const auto& src_node : nodes) {
            if (src_node == nullptr)
                throw std::runtime_error("NULL pointer in nodes_map.");
            if (node.first == src_node->get_primitive()->id) {
                copy_node_dependencies(node_ptr.get(), src_node.get());
                found = true;
                break;
            }
        }
        if (!found) {
            add_node_dependencies(node_ptr.get());
        }
        if (node_ptr->dependencies.size() == 0)
            inputs.push_back(node_ptr.get());
    }
}

// create all nodes from topology primitives, add dependencies among them and create inputs list
void program::prepare_nodes(topology const& topology) {
    auto const& topo_map = topology.get_primitives();
    for (const auto& prim : topo_map) {
        get_or_create(prim.second);
    }
    for (const auto& node : nodes_map) {
        auto node_ptr = node.second.get();
        if (node_ptr == nullptr)
            throw std::runtime_error("NULL pointer in nodes_map.");
        add_node_dependencies(node_ptr);
        if (node_ptr->dependencies.size() == 0) {
            inputs.push_back(node_ptr);
        }
    }
}

// add node's dependencies from its primitive dependencies
void program::add_node_dependencies(program_node* node) {
    auto deps = node->get_primitive()->dependencies();
    // add pointers to node's dependencies
    for (auto& dep : deps) {
        try {
            auto dep_node = nodes_map.at(dep.pid);
            node->dependencies.push_back({dep_node.get(), dep.idx});
            dep_node->users.push_back(node);
        } catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + dep.pid +
                                     " that is input to: " + node->get_primitive()->id);
        }
    }
}

/* helper method for program constructor from list of nodes which
   copies src_node dependencies to the destination node dest_node dependencies.
   But only to those which appaer in this program implementation nodes_map */
void program::copy_node_dependencies(program_node* dest_node, program_node* src_node) {
    if (dest_node->get_primitive()->id != src_node->get_primitive()->id) {
        throw std::runtime_error("Node " + src_node->get_primitive()->id + " and its copy " +
                                 dest_node->get_primitive()->id + " do not match.");
    }
    auto src_deps = src_node->get_dependencies();
    // add pointers to node's dependencies
    for (auto& src_dep : src_deps) {
        // do not copy dependencies to nodes which does not belong to the new (subgraph) topology
        if (nodes_map.find(src_dep.first->get_primitive()->id) == nodes_map.end())
            continue;

        try {
            auto dest_dep = nodes_map.at(src_dep.first->get_primitive()->id);
            dest_node->dependencies.push_back({dest_dep.get(), src_dep.second});
            dest_dep->users.push_back(dest_node);
        } catch (...) {
            throw std::runtime_error("Program doesn't contain primitive: " + src_dep.first->get_primitive()->id +
                                     " that is input to: " + src_node->get_primitive()->id);
        }
    }
}

void program::set_options() {
    static std::atomic<uint32_t> id_gen{0};
    prog_id = ++id_gen;
    assert(prog_id != 0);
}

void program::build_program(bool is_internal) {
    init_graph();
    _config.finalize(_engine);
    { pre_optimize_graph(is_internal); }
    run_graph_compilation();
    { post_optimize_graph(is_internal); }

#ifdef GPU_DEBUG_CONFIG
    if (get_config().get_dry_run_path().empty() || is_internal) {
#else
    {
#endif
        prepare_memory_dependencies();
        apply_opt_pass<build_implementations>();
    }

    if (!is_internal) {
        prim_info = get_current_stage_info();
        if (get_engine().get_device_info().has_separate_cache)
            transfer_memory_to_device();
    }
}

void program::init_graph() {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "Program::init_graph");
    apply_opt_pass<graph_initializations>();

    apply_opt_pass<mark_nodes>();
    for (auto& node : processing_order) {
        if (!node->is_type<data>())
            node->get_output_layouts();
    }
    // Perform initial shape_of subgraphs markup
    apply_opt_pass<mark_shape_of_subgraphs>();
}

void program::run_graph_compilation() { apply_opt_pass<compile_graph>(); }

void program::pre_optimize_graph(bool is_internal) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "Program::pre_optimize_graph");

    // trim to outputs
    apply_opt_pass<trim_to_outputs>();  // ToDo remove hidden dependencies from trimm pass

    processing_order.calculate_BFS_processing_order();  // this method makes sense only for OOOQ (out of order execution queue)

    bool output_size_handling_enabled = analyze_output_size_handling_need();

    bool optimize_data = _config.get_optimize_data();
    if (optimize_data) {
        apply_opt_pass<prepare_quantization>();
    }

    _layout_optimizer = std::make_unique<layout_optimizer>(output_size_handling_enabled);
    set_layout_optimizer_attributes(*_layout_optimizer);

    reorder_factory rf;
    if (optimize_data) {
        apply_opt_pass<prepare_primitive_fusing_through>();

        apply_opt_pass<pre_replace_deconv>();

        apply_opt_pass<reorder_transfer>();

        apply_opt_pass<prepare_primitive_fusing>();

        apply_opt_pass<select_preferred_formats>();

        apply_opt_pass<reorder_inputs>(rf);
        // Ideally this should be done before fusing to simplify logic and make the pass more powerful,
        // but after format selection to select correct alignment.
        // Unfortunately those passes currently happen in reverse order.
        apply_opt_pass<concat_input_order>();
    }

    apply_opt_pass<handle_reshape>();

    apply_opt_pass<prepare_padding>(output_size_handling_enabled);

    apply_opt_pass<remove_redundant_reorders>(optimize_data);

    // try to fuse buffers (i.e. depth_concat in bfyx format) after padding calculations
    if (optimize_data) {
        apply_opt_pass<prepare_buffer_fusing>();
    }

    // check if there exists some layout incompatibilities and add an reorder node if required
    apply_opt_pass<add_required_reorders>();

    // Check fusing primitives based on preferred format or layout optimization
    if (optimize_data) {
        apply_opt_pass<fuse_primitives_with_layout>();
    }

    // add optimization attributes for onednn primitives
    apply_opt_pass<add_onednn_optimization_attributes>();

    // Call shape_of subgraphs markup second time to update newely added nodes after graph
    // optimization passes
    apply_opt_pass<mark_shape_of_subgraphs>();

    // Mark operations that might be skipped at runtime as can_be_optimized.
    apply_opt_pass<mark_runtime_skippable_nodes>();
}

void program::post_optimize_graph(bool is_internal) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "Program::post_optimize_graph");
    // input reorder for fully connected if necessary
    apply_opt_pass<post_input_reorder>();

    reorder_factory rf;

    bool optimize_data = _config.get_optimize_data();

    if (!is_internal) {
        apply_opt_pass<post_optimize_weights>(rf);
    }

    apply_opt_pass<remove_redundant_reorders>(false, true);  // TODO: do we need it at this place also?

    auto partial_build = _config.get_partial_build_program();
#ifdef GPU_DEBUG_CONFIG
    if (!is_internal && (!partial_build || !_config.get_dry_run_path().empty())) {
#else
    if (!is_internal && !partial_build) {
#endif
        // ToDo remove hidden dependencies from propagate_constants pass
        apply_opt_pass<propagate_constants>();
    }

    if (optimize_data)
        apply_opt_pass<remove_redundant_reorders>(false, true, true); // pass to remove output reorders while all others graph optimizations were done

    // update inner program input/output primitive mappings
    apply_opt_pass<update_inner_program_io_map>();

    // Recalculate processing order after all graph transformation to keep optimal primitives ordering
    // for OOO queue
    if (_config.get_queue_type() == QueueTypes::out_of_order)
        get_processing_order().calculate_BFS_processing_order();

    apply_opt_pass<mark_state_init_subgraphs>();
}

// mark if the node is constant assuming that all dependencies are marked properly
void program::mark_if_constant(program_node& node) {
    if (node.get_dependencies().empty() || node.is_type<assign>() || node.is_type<read_value>() || node.is_type<gather_nonzero>()) {
        return;
    }
    node.constant = true;
    for (auto& dep : node.get_dependencies()) {
        if (!dep.first->is_constant()) {
            node.constant = false;
            return;
        }
    }
}

// mark if the node is in data flow assuming that all dependencies are marked properly
void program::mark_if_data_flow(program_node& node) {
    if (node.is_type<mutable_data>() || node.is_type<input_layout>() || node.is_type<read_value>()) {
        node.data_flow = true;
    } else {
        node.data_flow = false;
        size_t inputs_count = node.get_dependencies().size();
        if (node.is_type<detection_output>() || node.is_type<proposal>())
            inputs_count = 2;  // ignore third input as it is related to prior boxes (i.e. concat of prior-boxes)
        for (size_t idx = 0; idx < inputs_count; idx++) {
            if (node.get_dependency(idx).is_in_data_flow()) {
                node.data_flow = true;
                return;
            }
        }
    }
}

void program::transfer_memory_to_device() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("transfer_memory_to_device");
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "Program::transfer_memory_to_device");
    if (!get_engine().supports_allocation(allocation_type::usm_device))
        return;

    for (auto& node : processing_order) {
        if (node->is_shape_infer_dep()) {
            continue;
        }
        if (node->is_type<data>() && !node->need_lockable_memory()) {
            auto& data_node = node->as<data>();
            auto data_node_layout = data_node.get_output_layout();
            auto& mem = data_node.get_attached_memory();
            auto mem_layout = mem.get_layout();
            auto alloc_type = mem.get_allocation_type();

            if (mem_layout.count() == 0)
                continue;

            if (!mem_layout.compatible(data_node_layout)) {
                std::string err_str("Node and memory layouts are incompatible, error occurred for " + node->id() + " node");
                throw std::invalid_argument(err_str);
            }

            if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
                GPU_DEBUG_LOG << "[" << data_node.id() << ": constant]" << std::endl;
                // Allocate and transfer memory
                auto device_mem = mem.get_engine()->allocate_memory(data_node_layout, allocation_type::usm_device, false);
                device_mem->copy_from(get_stream(), mem);
                data_node.attach_memory(device_mem);
                GPU_DEBUG_LOG << "[" << data_node.id() << ": constant]" << std::endl;
                const_cast<memory::ptr&>(data_node.get_primitive()->mem).reset();
                // TODO: Do we need finish call here? Maybe call it in network::execute() ?
                get_stream().finish();
            }
        }
    }
}

program::nodes_ordering& program::get_processing_order() { return processing_order; }

const program::nodes_ordering& program::get_processing_order() const { return processing_order; }

const std::vector<primitive_id>& program::get_allocating_order(bool forced_update) {
    if (!forced_update && allocating_order.size() > 0)
        return allocating_order;

    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    auto& po = get_processing_order();
    for (auto node : po) {
        nodes_to_allocate.push_back(get_node_ptr(node->id()));
    }

    std::sort(nodes_to_allocate.begin(),
            nodes_to_allocate.end(),
            [&po](std::shared_ptr<program_node> const& lhs, std::shared_ptr<program_node> const& rhs) {
                    auto lhs_layout = lhs->get_output_layout();
                    auto rhs_layout = rhs->get_output_layout();
                    if (lhs_layout.is_dynamic() && lhs_layout.has_upper_bound()) {
                        lhs_layout.set_tensor(lhs_layout.get_tensor());
                    }
                    if (rhs_layout.is_dynamic() && rhs_layout.has_upper_bound()) {
                        rhs_layout.set_tensor(rhs_layout.get_tensor());
                    }

                    if (rhs_layout.is_dynamic() && !rhs_layout.has_upper_bound() && lhs_layout.is_dynamic() && !lhs_layout.has_upper_bound()) {
                        return po.get_processing_number(lhs.get()) < po.get_processing_number(rhs.get());
                    }

                    if (rhs_layout.is_dynamic())
                        return true;
                    if (lhs_layout.is_dynamic())
                        return false;

                    return (lhs_layout.bytes_count() > rhs_layout.bytes_count());
            });

    for (auto const& node : nodes_to_allocate) {
        allocating_order.emplace_back(node->id());
    }

    return allocating_order;
}

void program::prepare_memory_dependencies() {
    if (!_config.get_enable_memory_pool())
        return;
    for (auto& node : get_processing_order()) {
        node->add_memory_dependency(node->get_unique_id());
    }
    apply_opt_pass<basic_memory_dependencies>();
    apply_opt_pass<skipped_branch_memory_dependencies>();
    apply_opt_pass<oooq_memory_dependencies>();
}

std::string program::get_memory_dependencies_string() const {
    std::string mem_dep = "Memory dependencies/restrictions:\n";
    auto itr = processing_order.begin();
    while (itr != processing_order.end()) {
        auto& node = *itr;
        itr++;
        mem_dep = mem_dep.append("primitive: ")
                         .append(node->id())
                         .append("(unique_id:")
                         .append(std::to_string(node->get_unique_id()))
                         .append(") restricted list: ");
        for (auto it : node->get_memory_dependencies())
            mem_dep = mem_dep.append(std::to_string(it)).append(",");
        mem_dep = mem_dep.append("\n");
    }
    return mem_dep;
}

void program::apply_needed_padding(program_node& node, program_node& prev_node, const padding& needed_padding) {
    auto target_layout = prev_node.get_output_layout();

    // Short circuit if padding did not change.
    if (target_layout.data_padding == needed_padding)
        return;

    // Special handling for input nodes.
    if (prev_node.is_type<input_layout>() || prev_node.is_type<mutable_data>()) {
        target_layout.data_padding = needed_padding;

        auto r_prim = std::make_shared<reorder>("reorder_input_" + node.id(), prev_node.id(), target_layout);
        add_intermediate(r_prim, node, 0);
        get_or_create(r_prim).recalc_output_layouts(false);
        return;
    }

    prev_node.merge_output_padding(needed_padding);
}

void program::reverse_connection(program_node& dep_node, program_node& user_node) {
    if (std::find(dep_node.users.begin(), dep_node.users.end(), &user_node) != dep_node.users.end()) {
        remove_connection(dep_node, user_node);
        add_connection(user_node, dep_node);
    } else {
        throw std::runtime_error("Trying to reverse connection, but nodes are wrongly or not connected.");
    }
}

void program::set_state_initializers(const std::string& variable_id, const primitive_id& id) {
    state_initializers[variable_id].push_back(id);
}

bool program::has_state_initializers(const std::string& variable_id, const primitive_id& id) {
    auto it = state_initializers.find(variable_id);
    if (it != state_initializers.end()) {
        const auto& initializers = it->second;
        return std::find(initializers.begin(), initializers.end(), id) != initializers.end();
    }
    return false;
}

bool program::contains_state(const std::string& variable_id) {
    auto it = state_initializers.find(variable_id);
    if (it != state_initializers.end())
        return true;
    else
        return false;
}

program_node& program::get_or_create(std::shared_ptr<primitive> prim) {
    auto itr = nodes_map.lower_bound(prim->id);
    if (itr != nodes_map.end() && itr->first == prim->id)
        return *itr->second;

    auto new_node = prim->type->create_node(*this, prim);
    nodes_map.insert(itr, {prim->id, new_node});
    return *new_node;
}

void program::add_intermediate(program_node& node,
                               program_node& next,
                               size_t prev_idx,
                               bool connect_int_node_with_old_dep,
                               bool move_usrs_of_prev_to_node) {
    if (connect_int_node_with_old_dep && !node.dependencies.empty())
        throw std::invalid_argument(
            "Node which is about to be added in between two other nodes should not have any existing dependencies");

    auto& prev = next.get_dependency(prev_idx);
    // firstly add connection, later replace dependency, so 'prev' won't become dangling and therefore removed
    if (connect_int_node_with_old_dep) {
        add_connection(prev, node);
        if (processing_order.size() != 0) {
            processing_order.insert_next(&prev, &node);
        }
    }

    if (move_usrs_of_prev_to_node) {
        auto itr = prev.get_users().begin();
        while (itr != prev.get_users().end()) {
            auto usr = *itr;
            itr++;
            if (usr->id() != node.id())
                usr->replace_dependency(prev, node);
        }
        mark_if_constant(prev);
        mark_if_constant(node);
        mark_if_data_flow(prev);
        mark_if_data_flow(node);
    } else {
        next.replace_dependency(prev_idx, node);
        node.constant = prev.constant;
        node.data_flow = prev.data_flow;
    }
}

void program::add_intermediate(std::shared_ptr<primitive> prim,
                               program_node& next,
                               size_t prev_idx,
                               bool connect_int_node_with_old_dep,
                               bool move_usrs_of_prev_to_node) {
    add_intermediate(get_or_create(std::move(prim)), next, prev_idx, connect_int_node_with_old_dep, move_usrs_of_prev_to_node);
}

void program::add_intermediate(program_node& node,
                               program_node& next,
                               program_node& prev,
                               bool connect_int_node_with_old_dep,
                               bool move_usrs_of_prev_to_node) {
    bool node_found = false;
    size_t idx = 0;
    for (size_t i = 0; i < next.get_dependencies().size(); i++) {
        auto& input = next.get_dependency(i);
        if (input.id() == prev.id()) {
            idx = i;
            node_found = true;
            break;
        }
    }
    if (!node_found) {
        throw std::runtime_error("Trying to add intermediate node in between " + next.id() + " and dependecy " + prev.id() +
                        " but they are not connected in this way.");
    }
    add_intermediate(node, next, idx, connect_int_node_with_old_dep, move_usrs_of_prev_to_node);
}

void program::add_connection(program_node& prev, program_node& next, int32_t port_idx) {
    prev.users.push_back(&next);
    // When this function is called from program::replace, we need to keep the port number as it was
    if (port_idx < 0)
        port_idx = next.get_port_from_deps(prev.id());

    next.dependencies.push_back({&prev, port_idx});
}

void program::remove_connection(program_node& prev, program_node& next) {
    prev.users.remove(&next);
    next.dependencies.erase(std::remove_if(next.dependencies.begin(), next.dependencies.end(),
    [&](const std::pair<program_node*, int32_t>& dep) {
        return &prev == dep.first;
    }), next.dependencies.end());
}

void program::remove_all_connections(program_node& node) {
    // since the graph is not topological sorted, we need to remove the node from both dependencies and users
    for (auto& e : node.users) {
        e->dependencies.erase(std::remove_if(e->dependencies.begin(), e->dependencies.end(),
        [&](const std::pair<program_node*, int32_t>& dep) {
            return &node == dep.first;
        }), e->dependencies.end());
    }
    for (auto& e : node.dependencies) {
        e.first->users.remove(&node);
    }
    node.dependencies.clear();
    node.users.clear();
}

void program::rename(program_node& node, primitive_id const& new_id) {
    if (nodes_map.count(new_id))
        throw std::runtime_error("Trying to rename program_node but node with id " + new_id + " already exists");
    if (node.is_output())
        throw std::invalid_argument(
            "Trying to rename an output node. If you intend to do that, please clear 'output' flag manually.");

    auto node_itr = nodes_map.find(node.id());
    if (node_itr == nodes_map.end()) return;

    auto node_ptr = node_itr->second;
    nodes_map.emplace(new_id, node_ptr);
    nodes_map.erase(node.id());

    const_cast<primitive_id&>(node.desc->id) = new_id;
}

void program::swap_names(program_node& node1, program_node& node2) {
    const auto _extract_id = [](program_node& node) -> primitive_id& {
        return const_cast<primitive_id&>(node.desc->id);
    };

    nodes_map.at(node1.id()).swap(nodes_map.at(node2.id()));
    std::swap(_extract_id(node1), _extract_id(node2));
}

void program::replace_all_usages(program_node& old_node, program_node& new_node, bool remove_if_dangling) {
    return replace_all_usages(old_node, std::make_pair(&new_node, 0), remove_if_dangling);
}

void program::replace_all_usages(program_node& old_node, std::pair<program_node*, int32_t> new_node, bool remove_if_dangling) {
    // We need a copy of users of old_node because old_node may be removed when doing replace_dependency()
    const std::list<program_node*> users(old_node.users);
    auto itr = users.begin();
    while (itr != users.end()) {
        auto user = *(itr++);
        user->replace_dependency(old_node, new_node, remove_if_dangling);
    }
}

void program::replace(program_node& old_node, program_node& new_node) {
    if (!new_node.dependencies.empty() || !new_node.users.empty())
        throw std::invalid_argument("Node which is about to replace other node should be detached");

    if (new_node.is_output())
        throw std::invalid_argument(
            "Replacement node shouldn't be marked as an output since it's impossible to rename such node.");

    auto id = old_node.id();
    new_node.output_layouts = old_node.get_output_layouts();
    new_node.valid_output_layouts = old_node.valid_output_layouts;

    // copy old's dependencies
    // First copy them from old node to new node
    for (auto& dependency : old_node.dependencies) {
        add_connection(*dependency.first, new_node, dependency.second);
    }
    // Second delete them from old node
    while (!old_node.dependencies.empty()) {
        auto& dep = old_node.dependencies.front().first;
        remove_connection(*dep, old_node);
    }

    // append users
    for (auto& user : old_node.users) {
        new_node.users.push_back(user);
        for (auto& users_dep : user->dependencies) {
            if (users_dep.first == &old_node) {
                users_dep.first = &new_node;
                break;
            }
        }
    }

    old_node.users.clear();

    bool old_was_output = false;
    // copy node's state
    if (old_node.is_output()) {
        old_was_output = true;
        old_node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &old_node), outputs.end());
    }
    if (new_node.is_input())
        inputs.push_back(&new_node);
    if (old_node.is_input())
        inputs.remove(&old_node);

    new_node.constant = old_node.constant;
    new_node.data_flow = old_node.data_flow;
    new_node.user_mark = old_node.user_mark;
    const_cast<std::string&>(new_node.desc->origin_op_name) = old_node.desc->origin_op_name;
    const_cast<std::string&>(new_node.desc->origin_op_type_name) = old_node.desc->origin_op_type_name;

    processing_order.insert(&old_node, &new_node);
    if (processing_order.get_processing_iterator(old_node) != processing_order.end())
        processing_order.erase(&old_node);
    nodes_map.erase(id);
    rename(new_node, id);

    // mark new node as an output after renaming
    if (old_was_output) {
        new_node.set_output(true);
        outputs.push_back(&new_node);
    }
}

bool program::remove_if_dangling(program_node& node) {
    if (!node.users.empty())
        return false;
    if (!node.dependencies.empty())
        return false;

    if (!node.is_output()) {
        if (node.is_input())
            inputs.remove(&node);

        if (std::find(processing_order.begin(), processing_order.end(), &node) != processing_order.end())
            processing_order.erase(&node);
        optimized_out.push_back(node.id());
        nodes_map.erase(node.id());
    }
    return true;
}

bool program::extract(program_node& node) {
    if (node.get_dependencies().size() != 1)
        return false;

    if (node.is_output()) {
        auto& prev = node.get_dependency(0);
        auto node_id = node.id();

        node.set_output(false);
        outputs.erase(std::remove(outputs.begin(), outputs.end(), &node), outputs.end());

        rename(node, "_cldnn_tmp_" + node_id);
        rename(prev, node_id);

        prev.set_output(true);
        outputs.push_back(&prev);
    }

    auto input_with_port = node.get_dependency_with_port(0);
    auto& input = *input_with_port.first;

    // update primitive_map of loop primitive,
    // if extracted node is input of loop
    for (const auto& user : node.users) {
        if (user->is_type<loop>()) {
            loop_node& loop = *user;
            loop.update_primitive_map(node.id(), input.id());
        } else if (user->is_type<condition>()) {
            condition_node& cond = *user;
            cond.update_primitive_map(node.id(), input.id());
        }

        for (auto& dep : node.dependencies) {
            if (dep.first->is_type<loop>()) {
                loop_node& loop = *dep.first;
                loop.update_primitive_map(node.id(), user->id());
            } else if (dep.first->is_type<condition>()) {
                condition_node& cond = *dep.first;
                cond.update_primitive_map(node.id(), user->id());
            }
        }
    }
    input.users.remove(&node);
    node.dependencies.clear();

    if (!node.is_endpoint())
        replace_all_usages(node, input_with_port, false);

    if (std::find(processing_order.begin(), processing_order.end(), &node) != processing_order.end())
        processing_order.erase(&node);

    return true;
}

bool program::extract_and_remove(program_node& node) {
    if (extract(node)) {
        return remove_if_dangling(node);
    }

    return false;
}

bool program::move_node(program_node& node,
                        program_node& new_prev,
                        program_node& new_next) {
    if (extract(node)) {
        add_intermediate(node, new_next, new_prev);
        return true;
    }

    return false;
}

void program::fuse_nodes(program_node &fused_node,
                         program_node &peer_node,
                         std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>>* fusing_history) {
    auto peer_layout = peer_node.get_output_layout();
    fused_primitive_desc local_desc(peer_node.get_primitive());
    local_desc.f_param = get_node_ptr(peer_node.id())->get_fuse_params();
    local_desc.total_num_deps = peer_node.get_dependencies().size();
    local_desc.input_layout = peer_node.get_input_layout(0);
    local_desc.output_layout = peer_layout;

    if (fused_node.in_shape_of_subgraph && !peer_node.in_shape_of_subgraph) {
        fused_node.in_shape_of_subgraph = false;
    }

    int32_t orig_fused_node_num_deps = static_cast<int32_t>(fused_node.get_dependencies().size());
    auto fused_layout = fused_node.get_output_layout();
    auto fused_padding = fused_layout.data_padding;
    cldnn::padding needed_padding = padding::max(peer_layout.data_padding,
                                                 fused_padding);

    auto history_iter = fusing_history->find(peer_node.id());
    if (history_iter != fusing_history->end()) {
        for (auto& id : history_iter->second) {
            local_desc.fused_deps.emplace(id.first, id.second);
        }
    }
    // Add new dependencies to the fused_node
    size_t deps_idx = 0;
    for (size_t i = 0; i < peer_node.get_dependencies().size(); i++) {
        auto [dep, port] = peer_node.get_dependency_with_port(i);
        if (dep->id() == fused_node.id()) {
            if (fused_node.has_fused_primitives()) {
                local_desc.inputs.emplace_back(FusedInputType::INTERNAL, fused_node.get_fused_primitives().size() - 1, fused_layout.data_type);
            } else {
                local_desc.inputs.emplace_back(FusedInputType::ORIGINAL, 0, fused_layout.data_type);
            }
            deps_idx++;
            continue;
        }

        if (peer_node.is_type<quantize>()) {
            quantize_node& q_node = peer_node.as<quantize>();
            if (q_node.get_scale_shift_opt()) {
                bool can_drop_input = false;
                bool out_range_usage = q_node.get_per_tensor_output_range() && q_node.get_output_lo_val() < q_node.get_output_hi_val();

                // Drop input range if we use output per-tensor range or if clamp is used for input range
                can_drop_input |= (i == 1 || i == 2) && (out_range_usage || (!out_range_usage && !q_node.get_need_clamp()));
                // Drop output range - it's not used in scale-shift-opt quantize kernel
                can_drop_input |= i == 3 || i == 4;
                // Drop tensor with input scale when we have per-tensor parameter
                can_drop_input |= i == 5 && q_node.get_per_tensor_input_scale();
                // Drop tensor with input shift when we have per-tensor parameter or it's not needed at all
                can_drop_input |= i == 6 && (!q_node.get_need_pre_shift() || q_node.get_per_tensor_input_shift());
                // Drop tensor with output scale when we have per-tensor parameter or it's not needed at all
                can_drop_input |= i == 7 && (!q_node.get_need_post_scale() || q_node.get_per_tensor_output_scale());
                // Drop tensor with output shift when we have per-tensor parameter or it's not needed at all
                can_drop_input |= i == 8 && (!q_node.get_need_post_shift() || q_node.get_per_tensor_output_shift());

                if (can_drop_input)
                    continue;
            }
        }

        auto port_idx = fused_node.get_port_from_deps(dep->id());
        fused_node.dependencies.push_back({dep, port_idx});
        local_desc.inputs.emplace_back(FusedInputType::EXTERNAL, fused_node.dependencies.size() - 1, dep->get_output_layout(port).data_type);
        local_desc.deps.emplace_back(dep->id(), deps_idx++);
        dep->users.push_back(&fused_node);
    }
    if (local_desc.deps.size()) {
        local_desc.outer_dep_start_idx = orig_fused_node_num_deps;
    }

    local_desc.total_num_deps = std::min(local_desc.total_num_deps, deps_idx);

    fused_node.add_fused_primitive(local_desc);
    // This shouldn't happen, but who knows...
    if (peer_node.has_fused_primitives()) {
        fused_node.add_fused_primitives(peer_node.get_fused_primitives());
    }
    add_optimized_primitive_info(peer_node.id(), { fused_node.id() });

    for (auto& user : peer_node.users) {
        size_t dep_idx = 0;
        for (auto& dep : user->dependencies) {
            if (dep.first->id() == peer_node.id())
                break;
            dep_idx++;
        }
        (*fusing_history)[user->id()].push_back(std::make_pair(peer_node.id(), dep_idx));
    }

    // Remove all edges connected with peer node
    while (peer_node.get_dependencies().size() > 0) {
        auto& dep = peer_node.get_dependency(peer_node.get_dependencies().size() - 1);
        remove_connection(dep, peer_node);
    }
    replace_all_usages(peer_node, fused_node);

    // Update output layout. Recalculation is not needed.
    fused_node.merge_output_padding(needed_padding);
    fused_node.set_output_layout(peer_layout, false);
    fused_node.recalc_output_layout(true);
}

void program::remove_nodes(std::vector<program_node*>& to_remove) {
    for (auto const& node : to_remove) {
        if (node->is_input()) {
            get_inputs().remove(node);
        } else {
            for (auto& dep : node->dependencies) {
                dep.first->users.remove(node);
            }
        }
        for (auto& user : node->users) {
            user->dependencies.erase(std::remove_if(user->dependencies.begin(), user->dependencies.end(),
            [&](const std::pair<program_node*, int32_t>& dep) {
                return node == dep.first;
            }), user->dependencies.end());
        }
        get_processing_order().erase(node);
        optimized_out.push_back(node->id());
        nodes_map.erase(node->id());
    }
}

// TODO: break this function into number of smaller ones + add per-primitive fields (possibly use
// primitive_inst::to_string?)
void program::dump_program(const char* stage, bool with_full_info) const {
    std::string path = get_dir_path(_config);
    if (path.empty() || !with_full_info) {
        return;
    }

    std::ofstream graph(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".graph");
    dump_graph_init(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".info");
    dump_graph_info(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".order");
    dump_graph_processing_order(graph, *this);

    graph.open(path + "cldnn_program_" + std::to_string(prog_id) + "_" + stage + ".optimized");
    dump_graph_optimized(graph, *this);
}

data_types program::get_inference_precision(const program_node& node) const {
    if (node.is_input()) {
        return node.get_output_layout().data_type;
    }
    std::vector<data_types> input_dts;
    for (auto& dep : node.get_dependencies()) {
        if (dep.first->is_valid_output_layout())
            input_dts.push_back(dep.first->get_output_layout().data_type);
    }

    // Return f32 data_type as default inference precision if any layout is invalid
    if (input_dts.size() != node.get_dependencies().size() || !node.is_valid_output_layout())
        return data_types::f32;

    data_types output_dt = node.get_output_layout().data_type;

    assert(!input_dts.empty());
    if (node.is_type<reorder>()) {
        // If reorder has different input/output types - pick the max one as runtime precision
        return data_type_traits::max_type(input_dts[0], output_dt);
    } else if (node.is_type<quantize>()) {
        if (data_type_traits::is_quantized(output_dt))
            return output_dt;
        return data_type_traits::max_type(input_dts[0], output_dt);
    } else if (node.is_type<eltwise>()) {
        auto max_dt = input_dts[0];
        for (size_t i = 1; i < input_dts.size(); i++) {
            max_dt = data_type_traits::max_type(max_dt, input_dts[i]);
        }
        return max_dt;
    } else if (node.is_type<convolution>() || node.is_type<deconvolution>() || node.is_type<fully_connected>() || node.is_type<gemm>()) {
        if (input_dts.size() < 2) {
            throw std::runtime_error("[clDNN] Invalid inputs count in node " + node.id() + " during stage info collection. Expected >= 2 inputs");
        }
        if (data_type_traits::is_quantized(input_dts[0]) && data_type_traits::is_quantized(input_dts[1])) {
            return input_dts[0];
        } else {
            return data_type_traits::max_type(input_dts[0], input_dts[1]);
        }
    }

    return input_dts[0];
}

std::string program::get_implementation_info(const primitive_id& id) const {
    try {
        const auto& node = get_node(id);
        auto impl = node.get_selected_impl();
        auto kernel_name = impl ? impl->get_kernel_name() : "";
        return !kernel_name.empty() ? (kernel_name + "__" + dt_to_str(get_inference_precision(node))) : "undef";
    } catch (...) { }

    return "undef";
}

program::primitives_info program::get_current_stage_info() const {
    primitives_info info;

    // Get info for actually executed graph nodes
    int exec_id = 0;
    for (auto& p : get_processing_order()) {
        std::vector<primitive_id> users;
        for (auto& user : p->users) {
            users.push_back(user->id());
        }
        std::vector<primitive_id> dependencies;
        for (auto& a : p->dependencies) {
            dependencies.push_back(a.first->id());
        }

        std::vector<primitive_id> fused;
        for (auto& op_prim : optimized) {
            for (auto& fused_to : op_prim.second) {
                if (p->id() == fused_to) {
                    fused.push_back(op_prim.first);
                }
            }
        }

        // Initialize output_layout with dummy values and use them if layout is invalid
        layout output_layout{ cldnn::data_types::f32, cldnn::format::any, {1, 1, 1, 1} };

        if (p->is_valid_output_layout())
            output_layout = p->get_output_layout();

        primitive_info pi(p->id(),
                          type_to_str(p->get_primitive()),
                          dependencies,
                          users,
                          fused,
                          output_layout,
                          fmt_to_str(output_layout.format),
                          get_implementation_info(p->id()),
                          p->is_valid_output_layout() ?
                            get_inference_precision(*p) : cldnn::data_types::f32,
                          p->selected_impl ? p->selected_impl->is_cpu() : false,
                          exec_id++);

        info.push_back(pi);
    }

    return info;
}

void program::save_pass_info(std::string pass_name) {
    GPU_DEBUG_IF(!_config.get_dump_graphs_path().empty())
        optimizer_passes_info.emplace_back(pass_name, get_current_stage_info());
}

void program::add_optimized_primitive_info(primitive_id optimized_primitive_id,
                                           std::vector<primitive_id> replaced_with_ids) {
    for (auto& e : optimized) {
        auto it = std::find_if(e.second.begin(), e.second.end(), [&optimized_primitive_id](const primitive_id& id) {
           return optimized_primitive_id == id;
        });

        if (it != e.second.end()) {
            e.second.erase(it);
            e.second.insert(e.second.end(), replaced_with_ids.begin(), replaced_with_ids.end());
        }
    }
    optimized.emplace_back(optimized_primitive_id, replaced_with_ids);
}

const program::graph_optimizer_info& program::get_optimizer_passes_info() const {
    return optimizer_passes_info;
}

const program::primitives_info& program::get_primitives_info() const { return prim_info; }

void program::apply_opt_pass(base_pass& pass) { pm->run(*this, pass); }

void program::set_layout_optimizer_attributes(layout_optimizer& lo) {
    lo.set_implementation_forcing(_config.get_force_implementations());

    // first pass to set layout optimization_attributes for topology
    bool can_use_fsv16 = true;
    bool can_use_bs_fs_yx_bsv16_fsv16 = true;
    bool is_quantized_int8_model = false;
    size_t total_asym_quantized_conv_layers = 0;
    size_t total_dw_conv_layers = 0;
    size_t total_dw_splitted_conv_layers = 0;
    size_t total_1x1_fm_conv_layers = 0;
    size_t total_grouped_conv_layers = 0;
    size_t opt_deconv_layers_b_fs_zyx_fsv16 = 0;
    size_t opt_deconv_layers_b_fs_yx_fsv16 = 0;
    size_t total_crop_layers = 0;

    for (auto& node : get_processing_order()) {
        auto &prim = *node;
        if (prim.type() == cldnn::convolution::type_id()) {
            auto &conv = prim.as<convolution>();
            if (conv.get_primitive()->groups > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::group_convolution, 1);

            if (!conv.is_dynamic()) {
                // In dynamic shape, conv is fixed as a predefined format b_fs_yx_fsv16
                auto input_size = node->get_input_layout(0).get_tensor();
                auto ifm = static_cast<uint32_t>(input_size.feature[0]);
                if (conv.get_primitive()->groups == ifm && conv.get_primitive()->groups >= 16)
                    total_dw_conv_layers++;
                else if (conv.get_primitive()->groups == ifm && conv.get_primitive()->groups < 16)
                    total_dw_splitted_conv_layers++;  // this counter is needed due to compatibility with b_fs_yx_fsv16
                                                      // heuristics
                else if (conv.get_primitive()->groups > 1)
                    total_grouped_conv_layers++;

                if (input_size.spatial[0] == 1 && input_size.spatial[1] == 1)
                    total_1x1_fm_conv_layers++;
            }
            lo.update_formats_map(conv);

            if (conv.weights_zero_points_term() || conv.activations_zero_points_term())
                total_asym_quantized_conv_layers++;
        }
        if (prim.type() == cldnn::deconvolution::type_id()) {
            if (lo.is_format_optimized(prim.as<deconvolution>(), format::b_fs_zyx_fsv16))
                opt_deconv_layers_b_fs_zyx_fsv16 += 1;
            else if (lo.is_format_supported(prim.as<deconvolution>(), format::b_fs_yx_fsv16))
                opt_deconv_layers_b_fs_yx_fsv16 += 1;
        }

        // list of layers that do not support yxfb or perform worse than bfyx
        if (prim.type() == cldnn::detection_output::type_id() || prim.type() == cldnn::proposal::type_id() ||
            prim.type() == cldnn::roi_pooling::type_id() || prim.type() == cldnn::deconvolution::type_id() ||
            prim.type() == cldnn::resample::type_id() || prim.type() == cldnn::reorg_yolo::type_id())
            lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_only_layer, 1);

        if (prim.is_in_data_flow() &&
            prim.type() != cldnn::convolution::type_id() &&
            prim.type() != cldnn::deconvolution::type_id() &&
            prim.type() != cldnn::activation::type_id() &&
            prim.type() != cldnn::pooling::type_id() &&
            prim.type() != cldnn::eltwise::type_id() &&
            prim.type() != cldnn::permute::type_id() &&
            prim.type() != cldnn::reshape::type_id() &&
            prim.type() != cldnn::detection_output::type_id() &&
            prim.type() != cldnn::quantize::type_id() &&
            prim.type() != cldnn::custom_gpu_primitive::type_id() &&
            prim.type() != cldnn::concatenation::type_id() &&
            prim.type() != cldnn::fully_connected::type_id() &&
            prim.type() != cldnn::reorder::type_id() &&
            prim.type() != cldnn::input_layout::type_id() &&
            prim.type() != cldnn::softmax::type_id() &&
            prim.type() != cldnn::prior_box::type_id() &&
            prim.type() != cldnn::border::type_id() &&
            prim.type() != cldnn::resample::type_id() &&
            prim.type() != cldnn::crop::type_id() &&
            prim.type() != cldnn::depth_to_space::type_id() &&
            prim.type() != cldnn::shuffle_channels::type_id() &&
            (prim.type() != cldnn::mvn::type_id()
             || (prim.as<mvn>().get_input_layout().data_type != data_types::u8 &&
                 prim.as<mvn>().get_input_layout().data_type != data_types::i8)
             || prim.as<mvn>().get_primitive()->across_channels()) &&
            prim.type() != cldnn::arg_max_min::type_id() &&
            prim.type() != cldnn::dft::type_id() &&
            prim.type() != cldnn::grid_sample::type_id() &&
            prim.type() != cldnn::mutable_data::type_id() &&
            prim.type() != cldnn::reduce::type_id() &&
            prim.type() != cldnn::strided_slice::type_id() &&
            prim.type() != cldnn::region_yolo::type_id() &&
            prim.type() != cldnn::normalize::type_id() &&
            prim.type() != cldnn::group_normalization::type_id() &&
            prim.type() != cldnn::mvn::type_id() &&
            prim.type() != cldnn::gather::type_id() &&
            prim.type() != cldnn::scatter_nd_update::type_id() &&
            prim.type() != cldnn::broadcast::type_id() &&
            prim.type() != cldnn::ctc_loss::type_id() &&
            prim.type() != cldnn::non_max_suppression::type_id() &&
            prim.type() != cldnn::non_max_suppression_gather::type_id() &&
            prim.type() != cldnn::roi_align::type_id() &&
            prim.type() != cldnn::matrix_nms::type_id() &&
            prim.type() != cldnn::adaptive_pooling::type_id() &&
            prim.type() != cldnn::bucketize::type_id() &&
            prim.type() != cldnn::roll::type_id() &&
            prim.type() != cldnn::multiclass_nms::type_id() &&
            prim.type() != cldnn::prior_box::type_id() &&
            prim.type() != cldnn::roi_pooling::type_id() &&
            prim.type() != cldnn::resample::type_id() &&
            prim.type() != cldnn::eye::type_id() &&
            prim.type() != cldnn::generate_proposals::type_id() &&
            prim.type() != cldnn::reverse::type_id() &&
            prim.type() != cldnn::reorg_yolo::type_id() &&
            prim.type() != cldnn::gemm::type_id() &&
            prim.type() != cldnn::tile::type_id() &&
            prim.type() != cldnn::scatter_elements_update::type_id() &&
            prim.type() != cldnn::gather_tree::type_id() &&
            prim.type() != cldnn::experimental_detectron_detection_output::type_id() &&
            prim.type() != cldnn::convert_color::type_id() &&
            prim.type() != cldnn::unique_count::type_id() &&
            prim.type() != cldnn::unique_gather::type_id() &&
            prim.type() != cldnn::experimental_detectron_generate_proposals_single_image::type_id() &&
            prim.type() != cldnn::scaled_dot_product_attention::type_id()) {
            can_use_fsv16 = false;
        }

        if (prim.type() == cldnn::quantize::type_id() &&
            (prim.get_output_layout().data_type == data_types::i8 || prim.get_output_layout().data_type == data_types::u8)) {
            is_quantized_int8_model = true;
        }

        if (prim.type() == cldnn::crop::type_id()) {
            total_crop_layers++;
        }

        if (prim.is_in_data_flow() &&
            prim.type() != cldnn::convolution::type_id() &&
            prim.type() != cldnn::pooling::type_id() &&
            prim.type() != cldnn::eltwise::type_id() &&
            prim.type() != cldnn::reorder::type_id() &&
            prim.type() != cldnn::permute::type_id() &&
            prim.type() != cldnn::reshape::type_id() &&
            prim.type() != cldnn::input_layout::type_id() &&
            prim.type() != cldnn::activation::type_id() &&
            prim.type() != cldnn::dft::type_id() &&
            prim.type() != cldnn::grid_sample::type_id() &&
            prim.type() != cldnn::softmax::type_id() &&
            prim.type() != cldnn::fully_connected::type_id() &&
            prim.type() != cldnn::scatter_nd_update::type_id() &&
            prim.type() != cldnn::broadcast::type_id() &&
            prim.type() != cldnn::quantize::type_id() &&
            prim.type() != cldnn::ctc_loss::type_id() &&
            prim.type() != cldnn::non_max_suppression::type_id() &&
            prim.type() != cldnn::non_max_suppression_gather::type_id() &&
            prim.type() != cldnn::roi_align::type_id() &&
            prim.type() != cldnn::matrix_nms::type_id() &&
            prim.type() != cldnn::adaptive_pooling::type_id() &&
            prim.type() != cldnn::bucketize::type_id() &&
            prim.type() != cldnn::roll::type_id() &&
            prim.type() != cldnn::resample::type_id() &&
            prim.type() != cldnn::prior_box::type_id() &&
            prim.type() != cldnn::roi_pooling::type_id() &&
            prim.type() != cldnn::eye::type_id() &&
            prim.type() != cldnn::generate_proposals::type_id() &&
            prim.type() != cldnn::reverse::type_id() &&
            prim.type() != cldnn::reorg_yolo::type_id() &&
            prim.type() != cldnn::gemm::type_id() &&
            prim.type() != cldnn::tile::type_id() &&
            prim.type() != cldnn::scatter_elements_update::type_id() &&
            prim.type() != cldnn::gather_tree::type_id() &&
            prim.type() != cldnn::experimental_detectron_detection_output::type_id() &&
            prim.type() != cldnn::deconvolution::type_id() &&
            prim.type() != cldnn::multiclass_nms::type_id() &&
            prim.type() != cldnn::normalize::type_id() &&
            prim.type() != cldnn::group_normalization::type_id() &&
            prim.type() != cldnn::deconvolution::type_id() &&
            prim.type() != cldnn::unique_count::type_id() &&
            prim.type() != cldnn::unique_gather::type_id() &&
            prim.type() != cldnn::experimental_detectron_generate_proposals_single_image::type_id()) {
            can_use_bs_fs_yx_bsv16_fsv16 = false;
        }
    }

    size_t total_conv_layers = lo.get_total_conv_count();
    // Due to fact that single winograd convolution is faster than b_fs_yx_fsv16 and
    // using them together leads do redundant reorders, whole topology switch
    // will be performed if at least half of layers can use b_fs_yx_fsv16.
    // b_fs_yx_fsv16 deconv is faster than bfyx deconv with winograd convolution together,
    // whole topology switch will be perform if at lease one layer can use b_fs_yx_fsv16.
    // Crop layers are poorly optimized in fsv16 layout so whole topology stays in bfyx
    // if there are many crops (2x more then b_fs_yx_fsv16 convolutions)
    const float cond_denom = total_conv_layers > 0 ? 1.0f / static_cast<float>(total_conv_layers) : 1.0f;
    size_t num_of_conv_b_fs_yx_fsv16 = lo.get_optimized_conv_count({format::b_fs_yx_fsv16, false});

    bool should_use_b_fs_yx_fsv16_conv = is_quantized_int8_model ||
                                         (can_use_fsv16 &&
                                          total_conv_layers > 11 &&
                                          (num_of_conv_b_fs_yx_fsv16 * cond_denom > 0.5f || opt_deconv_layers_b_fs_yx_fsv16 >= 1) &&
                                          num_of_conv_b_fs_yx_fsv16 * 2 > total_crop_layers);

    bool should_use_fs_b_yx_fsv32_conv = total_conv_layers > 11 &&
                                         total_grouped_conv_layers == 0 &&
                                         total_1x1_fm_conv_layers * cond_denom < 0.8f;

    bool should_use_b_fs_zyx_fsv32_conv = total_asym_quantized_conv_layers > 1;

    bool should_use_bs_fs_yx_bsv16_fsv16 = can_use_bs_fs_yx_bsv16_fsv16 &&
                                  total_conv_layers > 11 &&
                                  total_conv_layers == lo.get_optimized_conv_count({format::bs_fs_yx_bsv16_fsv16, false}) &&
                                  total_grouped_conv_layers == 0 &&
                                  total_dw_splitted_conv_layers == 0 &&
                                  total_dw_conv_layers == 0;

    if (should_use_fs_b_yx_fsv32_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::fs_b_yx_fsv32_network, 1);

    if (should_use_b_fs_zyx_fsv32_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::b_fs_zyx_fsv32_network, 1);

    if (should_use_b_fs_yx_fsv16_conv)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::b_fs_yx_fsv16_network, 1);

    if (lo.get_optimized_conv_count({format::b_fs_zyx_fsv16, false}) >= 1 || opt_deconv_layers_b_fs_zyx_fsv16 >= 1)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::b_fs_zyx_fsv16_network, 1);

    if (should_use_bs_fs_yx_bsv16_fsv16)
        lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bs_fs_yx_bsv16_fsv16_network, 1);

#ifdef ENABLE_ONEDNN_FOR_GPU
    bool enable_onednn_for_tests = get_config().get_optimize_data() || is_internal_program();
    auto& engine = get_engine();
    if (engine.get_device_info().vendor_id == INTEL_VENDOR_ID &&
        get_config().get_queue_type() == QueueTypes::in_order &&
        enable_onednn_for_tests) {
            if (engine.get_device_info().supports_immad) {
                lo.add_all_onednn_impls_optimization_attribute();
            } else {
                if (get_config().get_use_onednn()) {
                    lo.enable_onednn_for<lstm_seq>();
                }
            }
        }
#endif
}

std::pair<int64_t, int64_t> program::get_estimated_device_mem_usage() {
    auto max_alloc_size = get_engine().get_device_info().max_alloc_mem_size;
    memory_pool pool(get_engine(), get_config());
    int64_t const_sum = 0;

#ifdef __unix__
    rlimit limit;
    int64_t cur_vmem = -1;
    if (getrlimit(RLIMIT_AS, &limit) == 0) {
        cur_vmem = limit.rlim_cur;
    }
#endif
    std::vector<program_node*> nodes_to_allocate{};
    for (auto node : processing_order) {
        nodes_to_allocate.push_back(node);
    }

    std::sort(nodes_to_allocate.begin(),
              nodes_to_allocate.end(),
              [](program_node* const& lhs, program_node* const& rhs) {
                  return (lhs->get_output_layout().bytes_count() > rhs->get_output_layout().bytes_count());
              });
    auto& engine = get_engine();
    int64_t host_alloc = 0;
    // just to prevent the memories from being freed during allocation
    std::unordered_set<memory::ptr> allocated_mem_ptrs;
    for (const auto& node : nodes_to_allocate) {
        auto out_size = node->get_output_layout().bytes_count();
        if (out_size > max_alloc_size) {
            // to consider: if the base batch size is > 1, should we allow this single output allocation to host?
            host_alloc += out_size;
            continue;
        }
        #ifdef __unix__
        // Check whether the host mem allocation might exceed avialalbe system VRAM or physical memory
        // Temporal solution for linux OoO memory killer
        // TODO: Ultimate solution will be the "estimation without actual allocation" mechanism for this issue,
        // which is also expected for better estimation performance
        int64_t max_global_mem_size = engine.get_device_info().max_global_mem_size;
        int64_t total_host_alloc_size = out_size + host_alloc + engine.get_used_device_memory(allocation_type::usm_host);
        if (engine.get_device_info().dev_type == cldnn::device_type::integrated_gpu)
            total_host_alloc_size += engine.get_used_device_memory(allocation_type::usm_device);
        if ((cur_vmem != -1 && total_host_alloc_size > cur_vmem * 0.5) || (total_host_alloc_size >= max_global_mem_size)) {
            GPU_DEBUG_INFO << "Estimated host mem usage calculated with default base batch size(16) exceeds the available memory ("
                           << cur_vmem << ")" << std::endl;
            return {-1L, -1L};
        }
        #endif

        if (node->can_be_optimized())
            continue;
        if (node->is_type<data>()) {
            const_sum += out_size;
        } else if (node->have_user_with_type<concatenation>() && node->get_users().size() == 1 && node->get_users().front()->can_be_optimized()) {
            continue;
        } else if (node->is_type<mutable_data>() && node->get_dependencies().empty()) {
            continue;
        } else {
            allocated_mem_ptrs.insert(primitive_inst::allocate_output(engine,
                                                                      pool,
                                                                      *node,
                                                                      *node->get_kernel_impl_params(),
                                                                      node->get_memory_dependencies(),
                                                                      0,
                                                                      false,
                                                                      0,
                                                                      false,
                                                                      node->is_output()));
        }
    }

    return std::make_pair(const_sum, get_engine().get_used_device_memory(allocation_type::usm_device));
}

void program::cancel_compilation_context() {
    if (_compilation_context != nullptr)
        _compilation_context->cancel();
}

void program::save(cldnn::BinaryOutputBuffer& ob) const {
    std::map<cldnn::memory::ptr, std::vector<const cldnn::program_node*>> mutable_datas_ptrs;
    ob << nodes_map.size();

    for (auto& node : nodes_map) {
        ob.setKernelImplParams(node.second->get_kernel_impl_params().get());

        if (node.second->is_type<data>() && node.second->as<data>().get_primitive()->mem == nullptr) {
            auto& data_node = node.second->as<data>();
            if (data_node.get_attached_memory_ptr() == nullptr) {
                ob << false;
                continue;
            } else {
                node.second->as<data>().typed_desc()->mem = data_node.get_attached_memory_ptr();
            }
        }

        ob << true;

        ob << node.second->desc;

        if (node.second->is_type<mutable_data>()) {
            mutable_datas_ptrs[node.second->as<mutable_data>().get_attached_memory_ptr()].push_back(node.second.get());
        }
    }

    std::list<std::pair<primitive_id, primitive_id>> output_sharing_mutable_datas;
    for (auto item : mutable_datas_ptrs) {
        if (item.second.size() != 2)
            continue;

        output_sharing_mutable_datas.push_back({item.second[0]->id(), item.second[1]->id()});
    }

    ob << output_sharing_mutable_datas.size();
    for (auto& shared_mem_pair : output_sharing_mutable_datas) {
        ob << shared_mem_pair.first;
        ob << shared_mem_pair.second;
    }

    for (auto& node : nodes_map) {
        ob << node.first;
        node.second->save(ob);
        ob << node.second->get_dependant_shape_of_nodes().size();
        for (auto& dep_node : node.second->get_dependant_shape_of_nodes()) {
            ob << dep_node->id();
        }
    }

    ob << inputs.size();
    for (auto& input : inputs) {
        ob << input->id();
    }

    ob << outputs.size();
    for (auto& output : outputs) {
        ob << output->id();
    }

    ob << _is_body_program;
    ob << _can_be_optimized;
    auto onednn_impls_size = get_layout_optimizer().get_all_onednn_impls_optimization_attribute().size();
    ob << onednn_impls_size;
    for (const auto& onednn_impl : get_layout_optimizer().get_all_onednn_impls_optimization_attribute()) {
        ob << prim_map_storage::instance().get_type_string(onednn_impl.first);
        ob << onednn_impl.second;
    }

    processing_order.save(ob);

    {
        auto& kernels_cache = get_kernels_cache();
        std::vector<primitive_id> impl_ids;
        for (auto& node : processing_order) {
            if (node->get_selected_impl() != nullptr) {
                impl_ids.emplace_back(node->id());
                kernels_cache.add_to_cached_kernels(node->get_selected_impl()->get_kernels());
            }
        }
        ob << kernels_cache;
        ob << impl_ids;
        for (auto& impl_id : impl_ids) {
            std::string type_name = get_node_ptr(impl_id)->get_selected_impl()->m_manager->get_type_info().name;
            ob << type_name;
            auto params = get_node_ptr(impl_id)->get_kernel_impl_params();
            ob.setKernelImplParams(params.get());
            ob << get_node_ptr(impl_id)->selected_impl;
            ob << get_node_ptr(impl_id)->get_selected_impl()->get_cached_kernel_ids(kernels_cache);
        }
    }

    ob << optimized_out.size();
    for (auto& opt_prim : optimized_out) {
        ob << opt_prim;
    }

    ob << prim_info.size();
    for (auto& p_info : prim_info) {
        ob << p_info.original_id;
        ob << p_info.type_id;
        ob << p_info.c_dependencies;
        ob << p_info.c_users;
        ob << p_info.c_fused_ids;
        ob << p_info.output_layout;
        ob << p_info.layout_str;
        ob << p_info.kernel_id;
        ob << make_data(&p_info.runtime_precision, sizeof(data_types));
        ob << p_info.is_cpu;
        ob << p_info.exec_id;
    }

    ob << allocating_order.size();
    for (auto const& node_id : allocating_order) {
        ob << node_id;
    }

    ob << state_initializers.size();
    for (auto& state_initializer : state_initializers) {
        ob << state_initializer.first;
        ob << state_initializer.second;
    }
}

void program::load(cldnn::BinaryInputBuffer& ib) {
    init_program();

    std::shared_ptr<WeightsMemory> weights_memory = nullptr;
    std::string weights_path = _config.get_weights_path();
    auto model_ptr = _config.get_model();
    if (_config.get_cache_mode() == ov::CacheMode::OPTIMIZE_SIZE) {
        if (model_ptr) {
            weights_memory = std::make_shared<WeightsMemory>(model_ptr);
        } else if (!weights_path.empty()) {
            ov::util::validate_weights_path(weights_path);
            weights_memory = std::make_shared<WeightsMemory>(ov::load_mmap_object(weights_path));
        } else {
            OPENVINO_THROW("Weights path or model is required for cache mode OPTIMIZE_SIZE");
        }
    }

    size_t num_nodes;
    ib >> num_nodes;
    bool is_valid_data_node;
    for (size_t i = 0; i < num_nodes; ++i) {
        ib >> is_valid_data_node;
        if (!is_valid_data_node)
            continue;

        std::shared_ptr<cldnn::primitive> prim;
        ib >> prim;
        if (auto data_prim = dynamic_cast<cldnn::data*>(prim.get())) {
            data_prim->load_weights(ib, weights_memory);
        }
        get_or_create(prim);
    }

    size_t num_output_sharing_mutable_datas;
    ib >> num_output_sharing_mutable_datas;
    for (size_t i = 0; i < num_output_sharing_mutable_datas; ++i) {
        primitive_id md_id1, md_id2;
        ib >> md_id1;
        ib >> md_id2;

        auto& md_node1 = get_node(md_id1).as<mutable_data>();
        auto& md_node2 = get_node(md_id2).as<mutable_data>();

        md_node2.typed_desc()->mem = md_node1.typed_desc()->mem;
        md_node2.replace_memory(md_node2.typed_desc()->mem);
    }

    for (size_t i = 0; i < num_nodes; ++i) {
        primitive_id prim_id;
        ib >> prim_id;
        auto& p_node = get_node(prim_id);
        p_node.load(ib);
        size_t num_dep_nodes;
        ib >> num_dep_nodes;
        for (size_t i = 0; i < num_dep_nodes; ++i) {
            ib >> prim_id;
            auto& dep_node = get_node(prim_id);
            p_node.add_dependant_shape_of_node(&dep_node);
        }
    }

    ib >> num_nodes;
    inputs.clear();
    for (size_t i = 0; i < num_nodes; ++i) {
        primitive_id prim_id;
        ib >> prim_id;
        auto& p_node = get_node(prim_id);
        inputs.emplace_back(&p_node);
    }

    ib >> num_nodes;
    outputs.clear();
    for (size_t i = 0; i < num_nodes; ++i) {
        primitive_id prim_id;
        ib >> prim_id;
        auto& p_node = get_node(prim_id);
        outputs.emplace_back(&p_node);
    }

    ib >> _is_body_program;
    ib >> _can_be_optimized;

    size_t num_of_onednn_impls;
    ib >> num_of_onednn_impls;
    for (size_t num = 0; num < num_of_onednn_impls; num++) {
        primitive_id p_id{};
        bool enabled;
        ib >> p_id;
        ib >> enabled;
        auto ptype_id = prim_map_storage::instance().get_type_id(p_id);
        get_layout_optimizer().set_value_onednn(ptype_id, enabled);
    }

    _loaded_from_cache = true;

    processing_order.load(ib, *this);
    set_layout_optimizer_attributes(*_layout_optimizer);

    {
        auto& kernels_cache = get_kernels_cache();
        ib >> kernels_cache;

        std::vector<primitive_id> impl_ids;
        ib >> impl_ids;

        for (auto& impl_id : impl_ids) {
            auto& p_node = get_node(impl_id);
            std::string type_name;
            ib >> type_name;
            ov::DiscreteTypeInfo type(type_name.c_str());
            auto impl_manager = p_node.type()->get(type);

            auto params = p_node.get_kernel_impl_params();
            ib.setKernelImplParams(params.get());
            ib >> p_node.selected_impl;

            p_node.selected_impl->m_manager = impl_manager.get();

            std::vector<std::string> cached_kernel_ids;
            ib >> cached_kernel_ids;
            p_node.selected_impl->init_by_cached_kernels(get_kernels_cache(), cached_kernel_ids);
        }
    }

    size_t optimized_out_size;
    ib >> optimized_out_size;
    optimized_out.clear();
    for (size_t i = 0; i < optimized_out_size; i++) {
        primitive_id opt_prim;
        ib >> opt_prim;
        optimized_out.emplace_back(opt_prim);
    }

    size_t prims_info_size;
    ib >> prims_info_size;
    prim_info.clear();
    for (size_t i = 0; i < prims_info_size; i++) {
        primitive_id original_id;
        std::string type_id;
        primitive::primitive_id_arr c_dependencies;
        primitive::primitive_id_arr c_users;
        primitive::primitive_id_arr c_fused_ids;
        layout output_layout;
        std::string layout_str;
        std::string kernel_id;
        data_types runtime_precision;
        bool is_cpu;
        int exec_id;

        ib >> original_id;
        ib >> type_id;
        ib >> c_dependencies;
        ib >> c_users;
        ib >> c_fused_ids;
        ib >> output_layout;
        ib >> layout_str;
        ib >> kernel_id;
        ib >> make_data(&runtime_precision, sizeof(data_types));
        ib >> is_cpu;
        ib >> exec_id;
        primitive_info p_info(original_id, type_id, c_dependencies, c_users, c_fused_ids,
                              output_layout, layout_str, kernel_id, runtime_precision, is_cpu, exec_id);
        prim_info.emplace_back(p_info);
    }

    size_t allocating_order_size;
    ib >> allocating_order_size;
    allocating_order.clear();
    for (size_t i = 0; i < allocating_order_size; i++) {
        primitive_id node_id;
        ib >> node_id;
        allocating_order.emplace_back(node_id);
    }

    size_t state_initializers_size;
    ib >> state_initializers_size;
    state_initializers.clear();
    for (size_t i = 0; i < state_initializers_size; i++) {
        std::string variable_id;
        std::vector<primitive_id> initializers;
        ib >> variable_id;
        ib >> initializers;
        state_initializers[variable_id] = initializers;
    }
}
