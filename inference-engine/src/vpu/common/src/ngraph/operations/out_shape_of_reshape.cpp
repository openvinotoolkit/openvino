// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/error.hpp>
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo OutShapeOfReshape::type_info;

OutShapeOfReshape::OutShapeOfReshape(
        const Output<Node>& inDataShape,
        const Output<Node>& outShapeDescriptor,
        bool specialZero) : Op({inDataShape, outShapeDescriptor}), m_specialZero(specialZero) {
    constructor_validate_and_infer_types();
}

void OutShapeOfReshape::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have only 2 inputs, provided: ", get_input_size());

    const auto& inDataShapeTensorShape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic input data shape");
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have input data shape tensor with rank 1, provided: ",
                          inDataShapeTensorShape.rank().get_length());

    const auto& outShapeDescriptorTensorShape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic output shape descriptor");
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have output shape descriptor tensor with rank 1, provided: ",
                          outShapeDescriptorTensorShape.rank().get_length());

    const auto& inDataShapeTensorType = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          inDataShapeTensorType.is_static() &&
                          inDataShapeTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") input data type needs to be an integral type. Got: ",
                          inDataShapeTensorType);
    const auto& outShapeDescriptorTensorType = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          outShapeDescriptorTensorType.is_static() &&
                          outShapeDescriptorTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") shape descriptor type needs to be an integral type. Got: ",
                          outShapeDescriptorTensorType);

    set_output_type(0, element::i64, outShapeDescriptorTensorShape);
}

std::shared_ptr<Node> OutShapeOfReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<OutShapeOfReshape>(new_args.at(0), new_args.at(1), m_specialZero);
}

bool OutShapeOfReshape::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("special_zero", m_specialZero);
    return true;
}

namespace {

template<element::Type_t ET>
bool getShapeFromHostTensorData(const HostTensorPtr& data, Shape& result) {
    using T = typename element_type_traits<ET>::value_type;
    T* dataPtr = data->get_data_ptr<ET>();
    if (!dataPtr) {
        return false;
    }
    size_t outputRank = data->get_shape()[0];

    for (int i = 0; i < outputRank; i++) {
        result.push_back(dataPtr[i]);
    }

    return true;
}

template<element::Type_t ET>
bool setShapeToHostTensorData(const HostTensorPtr& data, const Shape& shape) {
    using T = typename element_type_traits<ET>::value_type;
    T* dataPtr = data->get_data_ptr<ET>();
    if (!dataPtr) {
        return false;
    }
    size_t outputRank = data->get_shape()[0];
    if (shape.size() != outputRank) {
        return false;
    }

    for (int i = 0; i < outputRank; i++) {
        dataPtr[i] = shape[i];
    }
    return true;
}

bool getShapeFromHostTensorData(const HostTensorPtr& data, Shape& shape) {
    bool rc = false;
    switch (data->get_element_type()) {
        case element::Type_t::i8:
            rc = getShapeFromHostTensorData<element::Type_t::i8>(data, shape);
            break;
        case element::Type_t::i16:
            rc = getShapeFromHostTensorData<element::Type_t::i16>(data, shape);
            break;
        case element::Type_t::i32:
            rc = getShapeFromHostTensorData<element::Type_t::i32>(data, shape);
            break;
        case element::Type_t::i64:
            rc = getShapeFromHostTensorData<element::Type_t::i64>(data, shape);
            break;
        case element::Type_t::u8:
            rc = getShapeFromHostTensorData<element::Type_t::u8>(data, shape);
            break;
        case element::Type_t::u16:
            rc = getShapeFromHostTensorData<element::Type_t::u16>(data, shape);
            break;
        case element::Type_t::u32:
            rc = getShapeFromHostTensorData<element::Type_t::u32>(data, shape);
            break;
        case element::Type_t::u64:
            rc = getShapeFromHostTensorData<element::Type_t::u64>(data, shape);
            break;
        default: rc = false;
    }
    return rc;
}

bool setShapeToHostTensorData(const HostTensorPtr& data, const Shape& shape) {
    bool rc = false;
    switch (data->get_element_type()) {
        case element::Type_t::i8:
            rc = setShapeToHostTensorData<element::Type_t::i8>(data, shape);
            break;
        case element::Type_t::i16:
            rc = setShapeToHostTensorData<element::Type_t::i16>(data, shape);
            break;
        case element::Type_t::i32:
            rc = setShapeToHostTensorData<element::Type_t::i32>(data, shape);
            break;
        case element::Type_t::i64:
            rc = setShapeToHostTensorData<element::Type_t::i64>(data, shape);
            break;
        case element::Type_t::u8:
            rc = setShapeToHostTensorData<element::Type_t::u8>(data, shape);
            break;
        case element::Type_t::u16:
            rc = setShapeToHostTensorData<element::Type_t::u16>(data, shape);
            break;
        case element::Type_t::u32:
            rc = setShapeToHostTensorData<element::Type_t::u32>(data, shape);
            break;
        case element::Type_t::u64:
            rc = setShapeToHostTensorData<element::Type_t::u64>(data, shape);
            break;
        default: rc = false;
    }
    return rc;
}

bool evaluateOutShapeOfReshape(
        const HostTensorPtr& inDataShapeTensor,
        const HostTensorPtr& outShapeDescriptorTensor,
        bool specialZero,
        const HostTensorPtr& outShapeTensor) {
    if (!inDataShapeTensor || !outShapeDescriptorTensor || !outShapeTensor) {
        return false;
    }
    Shape inputShape;
    Shape outputShape;

    if (!getShapeFromHostTensorData(inDataShapeTensor, inputShape)) {
        return false;
    }
    if (!getShapeFromHostTensorData(outShapeDescriptorTensor, outputShape)) {
        return false;
    }

    if (std::any_of(outputShape.begin(), outputShape.end(), [](int64_t value) { return value < -1; })) {
        return false;
    }

    int zeroDimsCount = std::count_if(outputShape.begin(), outputShape.end(),
                                      [](int64_t value) { return value == 0; });
    int negativeDimsCount = std::count_if(outputShape.begin(), outputShape.end(),
                                          [](int64_t value) { return value == -1; });
    if (negativeDimsCount > 1) {
        return false;
    }

    size_t outputRank = outputShape.size();

    if (!(zeroDimsCount && specialZero) && !negativeDimsCount) {
        if (shape_size(inputShape) != shape_size(outputShape)) {
            return false;
        }
    } else {
        int negativeDimIdx = -1;

        size_t inputTotalDimCount = shape_size(inputShape);
        size_t outputTotalDimCount = 1;


        // compute the output shape
        for (size_t i = 0; i < outputRank; i++) {
            if (outputShape[i] == 0 && specialZero) {
                // Copy input_shape[i] for zero values
                if (i > inputShape.size() - 1) {
                    return false;
                }
                outputShape[i] = inputShape[i];
                outputTotalDimCount *= inputShape[i];
            } else if (outputShape[i] == -1) {
                negativeDimIdx = i;
            } else {
                outputTotalDimCount *= outputShape[i];
            }
        }

        if (negativeDimIdx != -1) {
            // Infer size such that number of output elements matches
            // input elements
            if (outputTotalDimCount == 0) {
                if (inputTotalDimCount != 0) {
                    return false;
                }
                outputShape[negativeDimIdx] = 0;
            } else {
                if (inputTotalDimCount % outputTotalDimCount != 0) {
                    return false;
                }
                outputShape[negativeDimIdx] = inputTotalDimCount / outputTotalDimCount;
            }
        }
    }

    if (!setShapeToHostTensorData(outShapeTensor, outputShape)) {
        return false;
    }

    return true;
}

}  // namespace

bool OutShapeOfReshape::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) {
    return evaluateOutShapeOfReshape(inputs[0], inputs[1], m_specialZero, outputs[0]);
}


}  // namespace op
}  // namespace vpu
}  // namespace ngraph
