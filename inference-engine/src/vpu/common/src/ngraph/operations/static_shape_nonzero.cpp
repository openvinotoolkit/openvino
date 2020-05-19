// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

std::shared_ptr<Node> StaticShapeNonZero::copy_with_new_args(
        const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<StaticShapeNonZero>(new_args.at(0), m_output_type);
}

bool StaticShapeNonZero::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

namespace {

template <typename InType, typename OutType>
void staticShapeNonZeroReference(const InType* input, OutType* outIndices, OutType* outShape, const Shape& inputShape) {
    auto strides = row_major_strides(inputShape);
    auto totalDimSize = shape_size(inputShape);

    const auto getCoord = [&strides](int offset){
        std::vector<size_t> coord;
        for (const size_t& stride : strides) {
            coord.insert(coord.begin(), offset / stride);
            offset %= stride;
        }

        return coord;
    };

    const auto addCoordToIndices = [&outIndices, &totalDimSize](const std::vector<size_t> &coord,
                                                                size_t nonZeroCount) {
        for (int j = 0; j < coord.size(); ++j) {
            outIndices[j * totalDimSize + nonZeroCount] = coord[j];
        }
    };

    const InType zeroValue = InType{0};
    const auto isNonZero = [&input, &zeroValue](size_t i) {
        return input[i] != zeroValue;
    };

    size_t nonZeroCount = 0;
    for (size_t i = 0; i < totalDimSize; ++i) {
        if (isNonZero(i)) {
            addCoordToIndices(getCoord(i), nonZeroCount++);
        }
    }

    outShape[0] = nonZeroCount;
    outShape[1] = inputShape.size();
}

template <element::Type_t InType>
bool evaluate(const HostTensorPtr& input,
              const HostTensorPtr& outIndices,
              const HostTensorPtr& outShape) {
    bool rc = true;

    switch (outIndices->get_element_type()) {
        case element::Type_t::i64:
            staticShapeNonZeroReference(input->get_data_ptr<InType>(),
                                        outIndices->get_data_ptr<element::Type_t::i64>(),
                                        outShape->get_data_ptr<element::Type_t::i64>(),
                                        input->get_shape());
            break;
        case element::Type_t::i32:
            staticShapeNonZeroReference(input->get_data_ptr<InType>(),
                                        outIndices->get_data_ptr<element::Type_t::i32>(),
                                        outShape->get_data_ptr<element::Type_t::i32>(),
                                        input->get_shape());
            break;
        default: rc = false; break;
    }

    return rc;
}

bool evaluateStaticShapeNonZero(const HostTensorPtr& input,
                                const HostTensorPtr& outIndices,
                                const HostTensorPtr& outShape) {
    bool rc = true;

    switch (input->get_element_type()) {
        TYPE_CASE(i8)(input, outIndices, outShape);
            break;
        TYPE_CASE(i16)(input, outIndices, outShape);
            break;
        TYPE_CASE(i32)(input, outIndices, outShape);
            break;
        TYPE_CASE(i64)(input, outIndices, outShape);
            break;
        TYPE_CASE(u8)(input, outIndices, outShape);
            break;
        TYPE_CASE(u16)(input, outIndices, outShape);
            break;
        TYPE_CASE(u32)(input, outIndices, outShape);
            break;
        TYPE_CASE(u64)(input, outIndices, outShape);
            break;
        TYPE_CASE(bf16)(input, outIndices, outShape);
            break;
        TYPE_CASE(f32)(input, outIndices, outShape);
            break;
        TYPE_CASE(f64)(input, outIndices, outShape);
            break;
        default: rc = false; break;
    }

    return rc;
}

} // namespace

bool StaticShapeNonZero::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) {
    return evaluateStaticShapeNonZero(inputs[0], outputs[0], outputs[1]);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
