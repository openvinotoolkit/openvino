// Copyright (C) 2023 Intel Corporation
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

}  // namespace internal
}  // namespace op
}  // namespace ov
