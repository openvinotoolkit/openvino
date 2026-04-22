// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/gather_nd_base.hpp"

#include "element_visitor.hpp"
#include "gather_nd_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/gather_nd.hpp"

namespace ov {
namespace op {

namespace gather_nd {
namespace {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t DATA_ET, class DT = fundamental_type_for<DATA_ET>>
    static result_type visit(const Tensor& data,
                             const Tensor& indices,
                             Tensor& out,
                             const Shape& data_shape,
                             const Shape& indices_shape,
                             const Shape& out_shape,
                             const int batch_dims) {
        using namespace ov::element;
        return IF_TYPE_OF(util_GatherNDBase_indices_type,
                          OV_PP_ET_LIST(i32, i64),
                          EvaluateByIndicesType,
                          indices.get_element_type(),
                          data.data<const DT>(),
                          indices,
                          out.data<DT>(),
                          data_shape,
                          indices_shape,
                          out_shape,
                          batch_dims);
    }

private:
    struct EvaluateByIndicesType : element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t INDICES_ET, class DT, class IT = fundamental_type_for<INDICES_ET>>
        static result_type visit(const DT* const data,
                                 const Tensor& indices,
                                 DT* const output,
                                 const Shape& data_shape,
                                 const Shape& indices_shape,
                                 const Shape& out_shape,
                                 const int batch_dims) {
            reference::gather_nd(data,
                                 indices.data<const IT>(),
                                 output,
                                 data_shape,
                                 indices_shape,
                                 out_shape,
                                 batch_dims);
            return true;
        }
    };
};

}  // namespace
}  // namespace gather_nd

namespace util {

GatherNDBase::GatherNDBase(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : Op({data, indices}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}

void GatherNDBase::validate_inputs_and_infer_shape() {
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);
}

bool GatherNDBase::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

bool GatherNDBase::has_evaluate() const {
    OV_OP_SCOPE(util_GatherNDBase_has_evaluate);
    switch (get_input_element_type(1)) {
    case element::i32:
    case element::i64:
        break;
    default:
        return false;
    }
    switch (get_input_element_type(0)) {
    case element::boolean:
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool GatherNDBase::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(util_GatherNDBase_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    auto& output = outputs[0];

    const auto out_shapes =
        gather_nd::gather_nd_base_shape_infer(this, std::vector<PartialShape>{data.get_shape(), indices.get_shape()});
    output.set_shape(out_shapes[0].to_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(util_GatherNDBase_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64),
                                      gather_nd::Evaluate,
                                      data.get_element_type(),
                                      data,
                                      indices,
                                      output,
                                      data.get_shape(),
                                      indices.get_shape(),
                                      output.get_shape(),
                                      static_cast<int>(m_batch_dims));
}

}  // namespace util
}  // namespace op
}  // namespace ov
