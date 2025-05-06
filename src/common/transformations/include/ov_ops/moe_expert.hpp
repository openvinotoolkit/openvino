// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
/// \brief MOE expert
///
/// \note MOE
class TRANSFORMATIONS_API MOEExpert : public ov::op::util::SubGraphOp {
public:
    OPENVINO_OP("MOEExpert", "ie_internal_opset", ov::op::util::SubGraphOp);

    MOEExpert() = default;

    struct Config {
        size_t topk = 0;
        size_t expert_num = 0;
        size_t hidden_size = 0;
        size_t expert_no = 0;
        bool has_non_zero = true;
    };

    MOEExpert(const OutputVector& args, const Config& config, const std::shared_ptr<ov::Model>& body);

    const Config& get_config() const;
    void set_config(const Config& config);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config{};
};

class TRANSFORMATIONS_API MOEExpert2 : public ov::op::Op {
public:
    OPENVINO_OP("MOEExpert2", "ie_internal_opset", ov::op::util::SubGraphOp);

    MOEExpert2() = default;

    struct Config {
        size_t topk = 0;
        size_t expert_num = 0;
        size_t hidden_size = 0;
        size_t expert_no = 0;
        size_t fused_router_logic = false;
    };

    MOEExpert2(const OutputVector& args, const Config& config, const std::vector<std::shared_ptr<ov::Model>>& body);

    const Config& get_config() const;
    void set_config(const Config& config);
    const std::vector<std::shared_ptr<ov::Model>> get_body() const { return m_body; }
    std::vector<std::shared_ptr<ov::Model>> get_body() { return m_body; }
    void add_body(int expert_no, std::shared_ptr<ov::Model> model) {
        OPENVINO_ASSERT(expert_no == static_cast<int>(m_body.size()));
        m_body.push_back(model);
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config{};
    std::vector<std::shared_ptr<ov::Model>> m_body;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
