// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {
///
/// \brief MOE experts
class TRANSFORMATIONS_API MOE : public ov::op::Op {
public:
    OPENVINO_OP("MOE", "ie_internal_opset");

    MOE() = default;

    struct Config {
        size_t topk{};
        size_t expert_num{};
        size_t hidden_size{};
        size_t intermediate_size{};
        size_t group_size{};              // quantized group size, 0 for no group size. same for gate/up/down
        ov::element::Type weight_type{};  // same for gate/up/down
        ov::element::Type scale_type{};   // same for gate/up/down
        ov::element::Type zp_type{};      // same for gate/up/down
        bool operator==(const Config& rhs) const {
            return std::tie(topk,
                            expert_num,
                            hidden_size,
                            intermediate_size,
                            group_size,
                            weight_type,
                            scale_type,
                            zp_type) == std::tie(rhs.topk,
                                                 rhs.expert_num,
                                                 rhs.hidden_size,
                                                 rhs.intermediate_size,
                                                 rhs.group_size,
                                                 rhs.weight_type,
                                                 rhs.scale_type,
                                                 rhs.zp_type);
        }
    };

    // 0: weight, 1: scale, 2: zp
    struct ConstsPerExpert {
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> gates;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> ups;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> downs;
    };
    struct Attributes {
        // expert config
        Config config;
        // expert weight/scale/zp
        std::vector<ConstsPerExpert> consts;
    };

    MOE(const OutputVector& args, const Attributes& attrs);

    const Config& get_config() const;
    void set_config(const Config& config);
    const std::vector<ConstsPerExpert>& get_consts() const {
        return m_attrs.consts;
    }

    void add_consts(size_t expert_no, const ConstsPerExpert& consts) {
        OPENVINO_ASSERT(expert_no == m_attrs.consts.size(),
                        "MOE add_consts failed. Expected expert number: ",
                        m_attrs.consts.size(),
                        ", current: ",
                        expert_no);
        m_attrs.consts.push_back(consts);
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    Attributes m_attrs;
};

}  // namespace ov::op::internal
