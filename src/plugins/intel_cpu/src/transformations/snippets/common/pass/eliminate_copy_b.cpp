// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eliminate_copy_b.hpp"

#include <cstddef>
#include <memory>
#include <set>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/utils/utils.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#elif defined(OPENVINO_ARCH_ARM64)
#    include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#endif

namespace ov::intel_cpu {
namespace {

bool is_supported_copy_b(const std::shared_ptr<ov::Node>& node) {
#if defined(OPENVINO_ARCH_X86_64)
    const auto copy_b = ov::as_type_ptr<BrgemmCopyB>(node);
    OPENVINO_ASSERT(copy_b, "EliminateCopyB expects BrgemmCopyB node on x64");
    // TODO [157340]: support external repacking for copyB with compensations.
    return !copy_b->get_config().with_compensations();
#elif defined(OPENVINO_ARCH_ARM64)
    OPENVINO_ASSERT(ov::is_type<aarch64::GemmCopyB>(node), "EliminateCopyB expects GemmCopyB node on aarch64");
    return true;
#else
    return false;
#endif
}

}  // namespace

bool pass::EliminateCopyB::should_extract(size_t param_idx) const {
    return m_runtime_repacking_supported || m_compile_time_repacking_idxs.count(param_idx) > 0;
}

bool pass::EliminateCopyB::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(EliminateCopyB);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::EliminateCopyB")

#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
    auto m_param = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto m_rank_norm = ov::pass::pattern::optional<ov::snippets::op::RankNormalization>(m_param);
#    if defined(OPENVINO_ARCH_X86_64)
    auto m_copy_b = ov::pass::pattern::wrap_type<BrgemmCopyB>({m_rank_norm});
#    else
    auto m_copy_b = ov::pass::pattern::wrap_type<aarch64::GemmCopyB>({m_rank_norm});
#    endif
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(m_copy_b);

    bool status = false;
    for (const auto& n : model->get_ordered_ops()) {
        if (!matcher->match(n)) {
            continue;
        }

        const auto& pattern_map = matcher->get_pattern_value_map();
        const auto& copy_b_out = pattern_map.at(m_copy_b);
        const auto copy_b_node = copy_b_out.get_node_shared_ptr();
        OPENVINO_ASSERT(copy_b_node, "CopyB node is null in EliminateCopyB transformation");
        if (!is_supported_copy_b(copy_b_node) || transformation_callback(copy_b_node)) {
            return false;
        }

        const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(pattern_map.at(m_param).get_node_shared_ptr());
        const auto param_idx = static_cast<size_t>(model->get_parameter_index(param));
        OPENVINO_ASSERT(param_idx < model->get_parameters().size(),
                        "Parameter index is invalid in EliminateCopyB transformation");
        if (!should_extract(param_idx)) {
            continue;
        }

        const auto& in_desc = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(copy_b_node->input(0));
        const auto& layout = in_desc->get_layout();

        // Update external repacking config for the further pipeline stages to mark this input as repacked.
        m_input_repackers[param_idx] = InputRepacker();

        // Since repacking is moved out of Subgraph body,
        // the rest weights subgraph must be updated with precision after repacking.
        param->set_element_type(copy_b_node->get_output_element_type(0));
        // Note: validation is called manually since set_element_type doesn't update output element type.
        param->validate_and_infer_types();
        if (pattern_map.count(m_rank_norm)) {
            pattern_map.at(m_rank_norm).get_node_shared_ptr()->validate_and_infer_types();
        }

        // If there is non-planar layout, insert Reorder to support shape inference.
        if (!ov::snippets::utils::is_planar_layout(layout)) {
            const auto& subtensor = in_desc->get_subtensor();
            const auto& reorder = std::make_shared<ov::snippets::op::Reorder>(copy_b_node->input_value(0), layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reorder->input(0), subtensor, layout);
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(reorder->output(0), subtensor);

            OPENVINO_ASSERT(ov::replace_node_update_name(copy_b_node, reorder),
                            "Failed to replace output in EliminateCopyB transformation");
        } else {
            OPENVINO_ASSERT(ov::replace_output_update_name(copy_b_out, copy_b_node->input_value(0)),
                            "Failed to replace output in EliminateCopyB transformation");
        }

        status = true;
    }

    return status;
#else
    return false;
#endif
}
}  // namespace ov::intel_cpu
