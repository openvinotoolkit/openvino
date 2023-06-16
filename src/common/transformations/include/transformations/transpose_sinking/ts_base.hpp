// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
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
 * @ingroup ie_transformation_common_api
 * @brief TSForwardBase is a base class for all forward transformations.
 */
class ov::pass::transpose_sinking::TSForwardBase : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSForwardBase", "0");
    TSForwardBase() = default;

    template <class... Types>
    void create_pattern(bool const_transpose_input, std::vector<size_t> transpose_indices = {}) {
        m_const_transpose_input = const_transpose_input;
        m_tranpose_indices = std::move(transpose_indices);
        m_pattern = ov::pass::pattern::wrap_type<Types...>([&](const Output<Node>& output) -> bool {
            return if_node_has_transpose_inputs(output, m_const_transpose_input, m_tranpose_indices);
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
    static bool if_node_has_transpose_inputs(const Output<Node>& output,
                                             bool const_transpose_input,
                                             const std::vector<size_t>& transpose_indices);

    std::shared_ptr<Node> m_pattern;
    bool m_const_transpose_input = true;
    std::vector<size_t> m_tranpose_indices;
};