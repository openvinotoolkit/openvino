// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "fa_utils.hpp"

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "snippets/op/fa.hpp"

namespace ov::intel_cpu {

/**
 * @interface FA
 * @brief The operation represet a flash attention alg op
 * @ingroup snippets
 */
class FACPU : public snippets::op::FA {
public:
    using FAConfig = fa_utils::FAConfig;
    OPENVINO_OP("FACPU", "SnippetsOpset", snippets::op::FA);  // mark_loop check FACPU

    FACPU(const ov::OutputVector& inputs,
          const FAConfig& config,
          const std::vector<PortDescriptor>& input_descs = {},
          const PortDescriptor& output_desc = {0, 0},
          const std::vector<size_t>& layout_a = {},
          const std::vector<size_t>& layout_b = {},
          const std::vector<size_t>& layout_c = {},
          const std::vector<size_t>& layout_d = {});
    FACPU();

    const FAConfig& get_config() const {
        return m_config;
    }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c,
                                                     const std::vector<size_t>& layout_d);
    const FAConfig m_config;
};

}  // namespace ov::intel_cpu
