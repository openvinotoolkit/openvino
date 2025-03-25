// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/runtime_configurator.hpp"

#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/mha_parallel_wa_optimizer.hpp"
#include "snippets/lowered/pass/solve_buffer_memory.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/loop_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {

using namespace ov::snippets::pass;
using namespace ov::snippets::lowered;
using namespace ov::snippets::lowered::pass;

#ifdef SNIPPETS_DEBUG_CAPS
std::string RuntimeConfig::to_string() const {
    std::stringstream out;
    out << " ========== RuntimeConfig state ==========\n" <<
           "tensor_rank: " << tensor_rank << "\n" <<
           "tile_rank: " << tile_rank << "\n" <<
           "master_shape: " << ov::Shape(master_shape) << "\n";
    out << "io_data_offsets: " << "\n";
    for (size_t i = 0; i < io_data_offsets.size(); ++i)
        out << "\t[" << i << "]" << ov::Shape(io_data_offsets[i]) << "\n";
    out << "buffer_scratchpad_size: " << buffer_scratchpad_size << "\n";
    out << "buffer_cluster_offsets: " << "\n";
    for (size_t i = 0; i < buffer_cluster_offsets.size(); ++i)
        out << "\t[" << i << "]" << buffer_cluster_offsets[i] << "\n";
    return out.str();
}
#endif

RuntimeConfigurator::RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c)
    : m_config(std::move(c)) {
    OPENVINO_ASSERT(m_config, "Runtime config is nullptr!");
}

void RuntimeConfigurator::reset_kernel_executor_table() const {
    m_config->kernel_executor_table = std::make_shared<ov::snippets::KernelExecutorTable>();
}

const std::shared_ptr<RuntimeConfig>& RuntimeConfigurator::get_updated_config(const lowered::LinearIRCPtr& linear_ir) {
    // First initialization
    if (m_io_num == 0)
        initialization(linear_ir);

    update(linear_ir);
    // Note: after 'update' is finished, io_shapes can be corrupted, so we move it to latest_shapes to avoid copying
    m_config->latest_shapes = std::move(m_config->io_shapes);
    return m_config;
}

void RuntimeConfigurator::initialization(const lowered::LinearIRCPtr& linear_ir) {
    init_data_info(linear_ir);
    init_tensor_rank(linear_ir);
    init_buffer_info(linear_ir);

    OPENVINO_ASSERT(m_io_num > 0, "LinearIR must have parameters and results");
    m_config->latest_shapes.resize(m_io_num);
    m_config->io_data_offsets.resize(m_io_num);
    m_config->tile_rank = linear_ir->get_config().m_loop_depth;

    RuntimeOptimizer::register_if_applicable<MHAParallelWAOptimizer>(m_intermediate_optimizers, linear_ir, this);
}

void RuntimeConfigurator::update(const lowered::LinearIRCPtr& linear_ir) {
    m_config->master_shape = linear_ir->get_master_shape();
    m_config->io_shapes = extract_shapes();
    m_config->io_layouts = extract_layouts();
    if (linear_ir->is_dynamic())
        update_loop_info(linear_ir);

    m_intermediate_optimizers.run(*linear_ir);

    // Update KernelExecutor Table should be before `update_buffer_scratchpad_size`
    // because `ComputeAllocationSize` depends on subtensors which are updated in the table
    get_kernel_executor_table()->update_state(linear_ir);
    update_buffer_scratchpad_size(linear_ir);

    update_data_offsets();
    m_final_optimizers.run(*linear_ir);
}

void RuntimeConfigurator::update_tensor_rank(const ov::snippets::VectorDims& master_shape) const {
    m_config->tensor_rank = master_shape.size();
}

void RuntimeConfigurator::init_tensor_rank(const lowered::LinearIRCPtr& linear_ir) const {
    m_config->tensor_rank = linear_ir->get_master_shape().size();
}

void RuntimeConfigurator::init_data_info(const lowered::LinearIRCPtr& linear_ir) {
    const auto& parameters = linear_ir->get_parameters();
    const auto& results = linear_ir->get_results();
    m_in_num = parameters.size();
    m_io_num = m_in_num + results.size();
    m_io_descs.reserve(m_io_num);
    m_io_data_sizes.reserve(m_io_num);

    auto update_io_parameters = [&](const PortDescriptorPtr& desc, const ov::element::Type& etype) {
        OPENVINO_ASSERT(desc, "IO Descriptor is missed!");
        OPENVINO_ASSERT(desc->get_shape().size() == desc->get_layout().size() || desc->get_layout().empty(),
                        "Incompatible ranks of shape and layout!");
        m_io_descs.push_back(desc);
        m_io_data_sizes.push_back(etype.size());
    };

    for (const auto& param : parameters) {
        // input->shape changing ops->load
        PortDescriptorPtr desc = nullptr;
        const auto& shape_infer_seq = utils::get_first_child_shape_infer_expr_seq(param);
        ExpressionPtr mem_desc_expr = param;
        if (!shape_infer_seq.empty()) {
            // [160048] Reorder, as any another ShapeInferOp, should just propagate input shape to output using target order
            //          without data movement. However, currently we have to save desc of input of the Reorder
            //          to support correct input data offsets calculations and MHAParallelWAOptimizer pass work.
            //          Please, remove this code part when the mentioned ticket is completed.
            const auto& reorder_it = std::find_if(shape_infer_seq.cbegin(), shape_infer_seq.cend(),
                                                  [](const ExpressionPtr& expr) {
                                                     return ov::is_type<op::Reorder>(expr->get_node());
                                                  });
            if (reorder_it != shape_infer_seq.cend()) {
                const auto& reorder = *reorder_it;
                const auto& etype = reorder->get_node()->get_output_element_type(0);
                update_io_parameters(reorder->get_input_port_descriptor(0), etype);
                continue;
            }

            mem_desc_expr = shape_infer_seq.back();
        }

        auto consumer_inputs = mem_desc_expr->get_output_port_connector(0)->get_consumers();
        for (const auto& child_input : consumer_inputs) {
            const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
            if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                desc = child_input.get_descriptor_ptr();
                break;
            }
        }
        OPENVINO_ASSERT(desc, "Descriptor is missed!");
        const auto& etype = mem_desc_expr->get_node()->get_output_element_type(0);
        update_io_parameters(desc, etype);
    }
    for (const auto& result : results) {
        // store->shape changing ops->result
        const auto& shape_infer_seq = utils::get_first_parent_shape_infer_expr_seq(result);
        const auto& mem_desc_expr = shape_infer_seq.empty() ? result : shape_infer_seq.back();
        const auto& desc = mem_desc_expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
        const auto& etype = mem_desc_expr->get_node()->get_input_element_type(0);
        update_io_parameters(desc, etype);
    }
}

void RuntimeConfigurator::init_buffer_info(const lowered::LinearIRCPtr& linear_ir) {
    std::map<size_t, std::set<lowered::BufferExpressionPtr>> dynamic_buffer_clusters, static_buffer_clusters;

    // All needed checks are in Validate pass
    const auto& buffer_expressions = linear_ir->get_buffers();
    for (const auto& buffer_expr : buffer_expressions) {
        // TODO [143395] : MemoryManager should provide exact containers with needed buffers (static or dynamic) without any `is_defined()`
        auto& clusters = buffer_expr->is_defined() ? static_buffer_clusters : dynamic_buffer_clusters;
        clusters[buffer_expr->get_cluster_id()].insert(buffer_expr);
    }

    const auto cluster_count = dynamic_buffer_clusters.size() + static_buffer_clusters.size();
    m_config->buffer_scratchpad_size = linear_ir->get_static_buffer_scratchpad_size();
    m_config->buffer_cluster_offsets.resize(cluster_count, utils::get_dynamic_value<size_t>());

    for (const auto& p : static_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;

        OPENVINO_ASSERT(cluster.size() > 0, "Incorrect size of buffer cluster");
        size_t cluster_offset = (*cluster.cbegin())->get_offset();
        m_config->buffer_cluster_offsets[cluster_id] = cluster_offset;
    }

    m_dynamic_buffer_clusters = std::move(dynamic_buffer_clusters);
}

void RuntimeConfigurator::update_expanded_loop_info(const lowered::ExpandedLoopInfoPtr& expanded_loop_info,
                                                    LoopInfoRuntimeParamsMap& initialized_info) {
    const auto& current_unified_loop_info = expanded_loop_info->get_unified_loop_info();

    OPENVINO_ASSERT(initialized_info.count(current_unified_loop_info) > 0, "UnifiedLoopInfo must be updated before ExpandedLoopInfo");
    auto& cur_initialized_info = initialized_info.at(current_unified_loop_info);
    auto& current_work_amount = cur_initialized_info.work_amount;
    const auto& ptr_increments = cur_initialized_info.ptr_increments;
    const auto& finalization_offsets = cur_initialized_info.finalization_offsets;

    const auto& decomposed_loop_type = expanded_loop_info->get_type();

    // If the specific iteration is not needed, we skip loop evaluation - set zero as work amount is enough
    if (!lowered::pass::InsertSpecificIterations::is_decomposed_loop_needed(current_unified_loop_info, decomposed_loop_type, current_work_amount)) {
        expanded_loop_info->set_work_amount(0);
        if (expanded_loop_info->is_evaluate_once())
            expanded_loop_info->set_increment(0);
        return;
    }

    const auto work_amount =
        lowered::pass::InsertSpecificIterations::get_decomposed_loop_work_amount(current_unified_loop_info, decomposed_loop_type, current_work_amount);
    expanded_loop_info->set_work_amount(work_amount);
    // Update remaining Loop work amount
    current_work_amount -= work_amount;

    // Update only `finalization offsets`. `Ptr increments` are always zeroed in this case
    auto updated_finalization_offsets = current_work_amount > 0 ? std::vector<int64_t>(finalization_offsets.size(), 0) : finalization_offsets;
    if (expanded_loop_info->is_evaluate_once()) {
        expanded_loop_info->set_increment(work_amount);
        // work_amount is equal to increment in cases with `evaluate_once`
        for (size_t i = 0; i < updated_finalization_offsets.size(); ++i)
            updated_finalization_offsets[i] += ptr_increments[i] * work_amount;
    } else {
        expanded_loop_info->update_ptr_increments(ptr_increments);
    }
    expanded_loop_info->update_finalization_offsets(updated_finalization_offsets);
}

void RuntimeConfigurator::update_loop_info(const lowered::LinearIRCPtr& linear_ir) {
    LoopInfoRuntimeParamsMap initialized_info;
    auto updater = [&](const lowered::LoopInfoPtr& loop_info) {
        if (const auto unified_loop_info = ov::as_type_ptr<lowered::UnifiedLoopInfo>(loop_info)) {
            if (initialized_info.count(unified_loop_info) == 0) {
                utils::update_runtime_parameters(unified_loop_info);
                initialized_info[unified_loop_info] = get_loop_runtime_params(unified_loop_info);
            }
        } else if (const auto expanded_loop_info = ov::as_type_ptr<lowered::ExpandedLoopInfo>(loop_info)) {
            update_expanded_loop_info(expanded_loop_info, initialized_info);
        } else {
            OPENVINO_THROW("Failed to update loop info: unknown type!");
        }
    };

    lowered::LoopInfoSet updated_loops;
    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        p.second->apply(updater, updated_loops);
    }
}

void RuntimeConfigurator::update_buffer_scratchpad_size(const lowered::LinearIRCPtr& linear_ir) const {
    const auto& loop_manager = linear_ir->get_loop_manager();
    // Align initial buffer scratchpad size with cache line size
    m_config->buffer_scratchpad_size = utils::rnd_up(linear_ir->get_static_buffer_scratchpad_size(), lowered::pass::SolveBufferMemory::byte_alignment);

    auto is_not_executed = [&loop_manager](const lowered::ExpressionPtr& buffer_expr) {
        const auto& loop_ids = buffer_expr->get_loop_ids();
        return std::any_of(loop_ids.cbegin(), loop_ids.cend(),
                          [&loop_manager](size_t loop_id) { return loop_manager->get_loop_info(loop_id)->get_work_amount() == 0; });
    };

    for (const auto& p : m_dynamic_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;

        auto& cluster_offset = m_config->buffer_cluster_offsets[cluster_id];
        cluster_offset = utils::get_dynamic_value<size_t>();

        size_t additional_size = 0;
        for (const auto& buffer_expr : cluster) {
            // No need to calculate allocation size of Buffers which are in Loops with `work_amount = 0` - they won't be executed
            if (is_not_executed(buffer_expr))
                continue;
            buffer_expr->init_allocation_size(loop_manager, m_config->tile_rank);
            const auto& allocation_size = buffer_expr->get_allocation_size();
            OPENVINO_ASSERT(!utils::is_dynamic_value(allocation_size), "Buffer scratchpad size must be defined!");
            additional_size = std::max(allocation_size * buffer_expr->get_node()->get_element_type().size(), additional_size);
        }

        // Align with cache line size. The experiments shows that it affects performance.
        additional_size = utils::rnd_up(additional_size, lowered::pass::SolveBufferMemory::byte_alignment);

        cluster_offset = m_config->buffer_scratchpad_size;
        OPENVINO_ASSERT(!utils::is_dynamic_value(cluster_offset), "Offset of the cluster must be defined!");
        m_config->buffer_scratchpad_size += additional_size;
    }

    OPENVINO_ASSERT(!utils::is_dynamic_value(m_config->buffer_scratchpad_size), "Buffer scratchpad size must be defined!");
}

void RuntimeConfigurator::update_data_offsets() const {
    const auto& shapes = m_config->io_shapes;
    const auto& layouts = m_config->io_layouts;
    OPENVINO_ASSERT(shapes.size() == m_io_num, "Number of custom shapes must be 0 or be equal to m_io_num");
    OPENVINO_ASSERT(layouts.size() == m_io_num, "Number of custom layouts must be 0 or be equal to m_io_num");
    for (size_t i = 0; i < m_io_num; ++i) {
        // offsets represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    offsets: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    offsets: s1*s3, s3,       0,  1
        const auto& shape = shapes[i];
        OPENVINO_ASSERT(m_config->tensor_rank >= shape.size(), "Incorrect tensor rank!");
        if (shape == m_config->latest_shapes[i])
            continue;
        if (utils::is_dynamic_vdims(shape))
            return;

        const auto idx_stride = m_config->tensor_rank - shape.size();
        compute_offsets(shape, i, idx_stride);

        auto& offsets = m_config->io_data_offsets[i];
        const auto& layout = layouts[i];
        if (!layout.empty()) {
            std::vector<size_t> reordered_offsets(offsets.size());
            const auto is_input = i < m_in_num;
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
            }
            offsets = std::move(reordered_offsets);
        }
    }
}

std::vector<VectorDims> RuntimeConfigurator::extract_shapes() const {
    std::vector<VectorDims> shapes(m_io_num);
    for (size_t i = 0; i < m_io_num; ++i)
        shapes[i] = m_io_descs[i]->get_shape();
    return shapes;
}

std::vector<std::vector<size_t>> RuntimeConfigurator::extract_layouts() const {
    std::vector<std::vector<size_t>> layouts(m_io_num);
    for (size_t i = 0; i < m_io_num; ++i)
        layouts[i] = m_io_descs[i]->get_layout();
    return layouts;
}

void RuntimeConfigurator::compute_offsets(const ov::snippets::VectorDims& shape, size_t idx, size_t idx_stride) const {
    auto& offsets = m_config->io_data_offsets[idx];
    auto dim_step = m_io_data_sizes[idx];

    offsets.resize(m_config->tensor_rank);
    std::fill(offsets.begin(), offsets.end(), 0);
    offsets[offsets.size() - 1] = dim_step;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        dim_step *= shape[i + 1];
        offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
    }
}

void RuntimeConfigurator::set_kernel_executor_table(std::shared_ptr<KernelExecutorTable> table) const {
    OPENVINO_ASSERT(table, "Failed to update Kernel Executor Table: passed table is missed");
    m_config->kernel_executor_table = std::move(table);
}

RuntimeConfigurator::UnifiedLoopInfoRtParams RuntimeConfigurator::get_loop_runtime_params(const lowered::UnifiedLoopInfoPtr& loop_info) {
    RuntimeConfigurator::UnifiedLoopInfoRtParams rt_params;
    rt_params.work_amount = loop_info->get_work_amount();
    const auto count = loop_info->get_input_count() + loop_info->get_output_count();
    rt_params.ptr_increments.resize(count);
    rt_params.finalization_offsets.resize(count);

    size_t idx = 0;
    loop_info->iterate_through_descs(
        [&rt_params, &idx](const lowered::UnifiedLoopInfo::LoopPortDesc& desc) {
            rt_params.ptr_increments[idx] = desc.ptr_increment;
            rt_params.finalization_offsets[idx] = desc.finalization_offset;
            ++idx;
        });
    return rt_params;
}
} // namespace snippets
} // namespace ov
