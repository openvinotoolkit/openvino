// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rank_constant.hpp"
#include "matmul_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/*RankConstant::RankConstant(const element::Type& type, const Shape& shape, const void* data, size_t rank)
    : Constant(type, shape, data), m_rank(rank) {
    validate_and_infer_types();
}
*/
RankConstant::RankConstant(const Constant& constant,
                 size_t rank) : Constant(constant), m_rank(rank) {
                 }
    
std::shared_ptr<ov::Node> RankConstant::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Constant>(*this);
}

void RankConstant::validate_and_infer_types() {
    Constant::validate_and_infer_types();
}

bool RankConstant::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("rank", m_rank);
    Constant::visit_attributes(visitor);
    return true;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
