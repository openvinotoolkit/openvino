// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
///
/// \brief MOE experts
class TRANSFORMATIONS_API MOEExpert : public ov::op::Op {
public:
    OPENVINO_OP("MOEExpert", "ie_internal_opset");

    MOEExpert() = default;

    struct Config {
        size_t topk = 0;
        size_t expert_num = 0;
        size_t hidden_size = 0;
        size_t fused_router_logic = false;
    };

    // 0: weight, 1: scale, 2: zp
    struct ConstsPerExpert {
        std::array<std::shared_ptr<ov::Node>, 3> gate;
        std::array<std::shared_ptr<ov::Node>, 3> up;
        std::array<std::shared_ptr<ov::Node>, 3> down;
    };

    MOEExpert(const OutputVector& args, const Config& config, const std::vector<ConstsPerExpert>& consts);

    const Config& get_config() const;
    void set_config(const Config& config);
    const std::vector<ConstsPerExpert> get_consts() const { return m_consts; }
    std::vector<ConstsPerExpert> get_consts() { return m_consts; }
    void add_consts(int expert_no, const ConstsPerExpert& consts) {
        OPENVINO_ASSERT(expert_no == static_cast<int>(m_consts.size()));
        m_consts.push_back(consts);
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Config m_config{};
    std::vector<ConstsPerExpert> m_consts;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
