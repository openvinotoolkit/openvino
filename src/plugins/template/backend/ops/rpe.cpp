#include "evaluate_node.hpp"


template <>
bool evaluate_node<ov::op::v5::RotRPEAttentionWeightWithIndexComputation>(std::shared_ptr<ov::Node> node,
                                    ov::TensorVector& outputs,
                                    const ov::TensorVector& inputs) {

    
    auto op = ov::as_type_ptr<ov::op::v5::RotRPEAttentionWeightWithIndexComputation>(node);
    return op->evaluate(outputs, inputs); 
}

template <>
bool evaluate_node<ov::op::v5::RotRPEProjectValueWithIndexComputation>(std::shared_ptr<ov::Node> node,
                                    ov::TensorVector& outputs,
                                    const ov::TensorVector& inputs) {

    
    auto op = ov::as_type_ptr<ov::op::v5::RotRPEProjectValueWithIndexComputation>(node);
    return op->evaluate(outputs, inputs); 
}
