#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {

class TSSN : public Op {
public:
    OPENVINO_OP("TSSN", "extension");

    TSSN() = default;
    
    TSSN(const Output<Node>& x, 
         const Output<Node>& h_prev,
         const Output<Node>& A,
         const Output<Node>& B,
         const Output<Node>& C,
         uint32_t func_id_a = 0,
         uint32_t func_id_b = 0,
         uint32_t func_id_c = 0,
         uint32_t func_id_agg = 0,
         bool packed_weights = false);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool has_evaluate() const override;
    
    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;

private:
    uint32_t m_func_id_a;
    uint32_t m_func_id_b;
    uint32_t m_func_id_c;
    uint32_t m_func_id_agg;
    bool m_packed_weights;

    void unpack_weights(const uint8_t* packed, int8_t* unpacked, size_t count) const;
};

} // namespace v0
} // namespace op
} // namespace ov
