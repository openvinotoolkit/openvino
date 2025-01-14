// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface Validate
 * @brief The pass validates OV model on correctness after all common optimizations
 * @ingroup snippets
 */
class Validate: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::Validate");
    Validate(const std::shared_ptr<ov::pass::PassConfig>& pass_config) : m_pass_config(pass_config) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    bool is_supported_constant(const std::shared_ptr<const ov::Node>& op);
    bool is_supported_convert(const std::shared_ptr<const ov::Node>& op);
    bool is_supported_matmul(const std::shared_ptr<const ov::Node>& op);
    bool is_supported_softmax(const std::shared_ptr<const ov::Node>& op);
    bool is_supported_fq(const std::shared_ptr<const ov::Node>& node);
    bool is_supported_transpose(const std::shared_ptr<const ov::Node>& node);
    bool is_supported_op(const std::shared_ptr<const ov::Node>& node);

    // Pass config of CommonOptimizations that contains information: which of common passes are disabled
    std::shared_ptr<ov::pass::PassConfig> m_pass_config;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
