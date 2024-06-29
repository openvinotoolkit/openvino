// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"
#include "util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
class RankConstant : public ov::op::v0::Constant {
public:
    OPENVINO_OP("FullyConnected", "gpu_opset");

    RankConstant() = default;
    //RankConstant(const element::Type& type, const Shape& shape, const void* data, size_t rank);
    RankConstant(const Constant& constant,
                 size_t rank);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

protected:
    size_t m_rank;
    TP_MODE m_tp_mode = TP_MODE::ALL_GATHERH;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov