// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSForwardBase;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSForwardBase is a base class for all forward transformations.
 */
class ov::pass::transpose_sinking::TSForwardBase : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSForwardBase", "0");
    TSForwardBase() = default;

    template <class... Types>
    void create_pattern(std::vector<size_t> transpose_indices = {},
                        const std::function<bool(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                                                 const std::shared_ptr<ov::op::v0::Constant>& transpose_order)>&
                            if_transpose_sinkable = utils::if_transpose_sinkable_default) {
        m_if_transpose_sinkable = if_transpose_sinkable;
        m_transpose_indices = std::move(transpose_indices);
        m_pattern = ov::pass::pattern::wrap_type<Types...>([&](const Output<Node>& output) -> bool {
            return if_node_has_transpose_inputs(output, m_transpose_indices, m_if_transpose_sinkable);
        });
    }

    using sinking_function =
        std::function<bool(const std::shared_ptr<Node>& main_node, const utils::TransposeInputsInfo& transpose_info)>;

    void transpose_sinking(const std::string& pass_name, const sinking_function& sinking_transformation = nullptr);

protected:
    static bool default_inputs_update(const std::shared_ptr<Node>& main_node,
                                      const utils::TransposeInputsInfo& transpose_info);

    void default_outputs_update(const std::shared_ptr<Node>& main_node,
                                const utils::TransposeInputsInfo& transpose_info);

private:
    static bool if_node_has_transpose_inputs(
        const Output<Node>& output,
        const std::vector<size_t>& transpose_indices,
        const std::function<bool(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                                 const std::shared_ptr<ov::op::v0::Constant>& transpose_order)>&);

    std::shared_ptr<Node> m_pattern;
    std::function<bool(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                       const std::shared_ptr<ov::op::v0::Constant>& transpose_order)>
        m_if_transpose_sinkable = utils::if_transpose_sinkable_default;
    std::vector<size_t> m_transpose_indices;
};
