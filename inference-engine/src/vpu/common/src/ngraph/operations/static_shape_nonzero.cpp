// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include "vpu/ngraph/operations/static_shape_nonzero.hpp"

#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo StaticShapeNonZero::type_info;

StaticShapeNonZero::StaticShapeNonZero(const Output<Node>& input, const element::Type& output_type)
        : Op({input}), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void StaticShapeNonZero::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1,
                          "StaticShapeNonZero must have only 1 input, provided: ",
                          get_input_size());

    const auto& arg_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, arg_shape.is_static(),
                          "StaticShapeNonZero doesn't support dynamic input shape");

    const auto& input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et.is_static() && (input_et.is_integral_number() || input_et.is_real() || input_et == ngraph::element::boolean),
                          "StaticShapeNonZero input data type needs to be a static numeric type. Got: ",
                          input_et);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i32 || m_output_type == element::i64,
                          "StaticShapeNonZero output data type can be either i32 or i64");

    const auto total_dim_size = Dimension(shape_size(arg_shape.to_shape()));
    set_output_type(0, m_output_type, {arg_shape.rank(), total_dim_size});
    set_output_type(1, m_output_type, {Dimension(2)});
}

std::shared_ptr<Node> StaticShapeNonZero::clone_with_new_inputs(
        const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<StaticShapeNonZero>(new_args.at(0), m_output_type);
}

bool StaticShapeNonZero::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

namespace {

template <element::Type_t OutType>
void evaluateStaticShapeNonZero(const Shape& inputShape,
                                const HostTensorPtr& nonZeroOutput,
                                const HostTensorPtr& outIndices,
                                const HostTensorPtr& outShape) {
    const auto nonZeroOutputBuffer = nonZeroOutput->get_data_ptr<OutType>();
    const auto outIndicesBuffer = outIndices->get_data_ptr<OutType>();
    const auto outShapeBuffer = outShape->get_data_ptr<OutType>();

    const auto totalInputSize = shape_size(inputShape);
    const auto inputRank = nonZeroOutput->get_partial_shape()[0].get_length();
    const auto nonZeroCount = nonZeroOutput->get_partial_shape()[1].get_length();

    for (int64_t i = 0; i < inputRank; ++i) {
        for (int64_t j = 0; j < nonZeroCount; j++) {
            outIndicesBuffer[i * totalInputSize + j] = nonZeroOutputBuffer[i * nonZeroCount + j];
        }
    }

    outShapeBuffer[0] = static_cast<typename ngraph::element_type_traits<OutType>::value_type>(inputRank);
    outShapeBuffer[1] = static_cast<typename ngraph::element_type_traits<OutType>::value_type>(nonZeroCount);
}

} // namespace

bool StaticShapeNonZero::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const {
    const auto& input = inputs[0];
    const auto& outIndices = outputs[0];
    const auto& outShape = outputs[1];

    const auto nonZeroOutput = std::make_shared<ngraph::runtime::HostTensor>(
            outIndices->get_element_type(),
            PartialShape{input->get_partial_shape().rank(), Dimension::dynamic()});
    bool rc = ngraph::opset3::NonZero().evaluate({nonZeroOutput}, {input});

    switch (nonZeroOutput->get_element_type()) {
        case element::Type_t::i32:
            evaluateStaticShapeNonZero<element::Type_t::i32>(input->get_shape(), nonZeroOutput, outIndices, outShape);
            break;
        case element::Type_t::i64:
            evaluateStaticShapeNonZero<element::Type_t::i64>(input->get_shape(), nonZeroOutput, outIndices, outShape);
            break;
        default: rc = false; break;
    }

    return rc;
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
