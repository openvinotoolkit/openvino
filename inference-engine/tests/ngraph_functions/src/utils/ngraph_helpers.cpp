// Copyright (C) 2019 Intel Corporationconvert2OutputVector
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <queue>

#include <ngraph/op/util/op_types.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/specialize_function.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset.hpp>

namespace ngraph {
namespace helpers {
std::ostream &operator<<(std::ostream &os, const ReductionType &m) {
    switch (m) {
        case Mean:
            os << "Mean";
            break;
        case Max:
            os << "Max";
            break;
        case Min:
            os << "Min";
            break;
        case Prod:
            os << "Prod";
            break;
        case Sum:
            os << "Sum";
            break;
        case LogicalOr:
            os << "LogicalOr";
            break;
        case LogicalAnd:
            os << "LogicalAnd";
            break;
        case LogicalXor:
            os << "LogicalXor";
            break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const PadMode &m) {
    switch (m) {
        case PadMode::CONSTANT:
            os << "CONSTANT";
            break;
        case PadMode::EDGE:
            os << "EDGE";
            break;
        case PadMode::REFLECT:
            os << "REFLECT";
            break;
        case PadMode::SYMMETRIC:
            os << "SYMMETRIC";
            break;
    }
    return os;
}

OutputVector convert2OutputVector(const std::vector<std::shared_ptr<Node>> &nodes) {
    OutputVector outs;
    std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<Node> &n) {
        for (const auto &out_p : n->outputs()) {
            outs.push_back(out_p);
        }
    });
    return outs;
}

std::vector<std::vector<std::uint8_t>> interpreterFunction(const std::shared_ptr<Function> &function, const std::vector<std::vector<std::uint8_t>> &inputs,
                                                           const std::vector<ngraph::element::Type_t> convertType) {
    runtime::Backend::set_backend_shared_library_search_directory("");
    auto backend = runtime::Backend::create("INTERPRETER");

    const auto &parameters = function->get_parameters();
    const auto &parametersNumber = parameters.size();
    const auto &inputsNumber = inputs.size();
    NGRAPH_CHECK(parametersNumber == inputsNumber,
                 "Got function (", function->get_friendly_name(), ") with ", parametersNumber, " parameters, but ",
                 inputsNumber, " input blobs");

    auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
    for (const auto &parameter : parameters) {
        const auto &parameterIndex = function->get_parameter_index(parameter);
        const auto &parameterShape = parameter->get_shape();
        const auto &parameterType = parameter->get_element_type();
        const auto &parameterSize = shape_size(parameterShape) * parameterType.size();

        const auto &input = inputs[parameterIndex];
        const auto &inputSize = input.size();
        NGRAPH_CHECK(parameterSize == inputSize,
                     "Got parameter (", parameter->get_friendly_name(), ") of size ", parameterSize,
                     " bytes, but corresponding input with index ", parameterIndex,
                     " has ", inputSize, " bytes");

        auto tensor = backend->create_tensor(parameterType, parameterShape);
        tensor->write(input.data(), parameterSize);
        inputTensors.push_back(tensor);
    }

    auto outputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
    const auto &results = function->get_results();
    for (size_t i = 0; i < results.size(); ++i) {
        outputTensors.push_back(std::make_shared<HostTensor>());
    }

    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);
    auto outputs = std::vector<std::vector<std::uint8_t>>(results.size());
    for (size_t resultIndex = 0; resultIndex < results.size(); resultIndex++) {
        auto& output = outputs[resultIndex];
        const auto& outputTensor = outputTensors[resultIndex];
        output.resize(shape_size(outputTensor->get_shape()) * outputTensor->get_element_type().size());
        outputTensors[resultIndex]->read(output.data(), output.size());
        if (!convertType.empty() && convertType[resultIndex] != element::Type_t::undefined &&
                outputTensor->get_element_type() != element::Type(convertType[resultIndex]))
            output = convertOutputPrecision(
                output,
                outputTensor->get_element_type(),
                convertType[resultIndex],
                shape_size(outputTensors[resultIndex]->get_shape()));
    }

    return outputs;
}

std::shared_ptr<Function> foldFunction(const std::shared_ptr<Function> &function,
                                       const std::vector<std::vector<std::uint8_t>> &inputs) {
    std::vector<element::Type> paramElementTypes;
    std::vector<PartialShape> paramShapes;
    for (const auto &param : function->get_parameters()) {
        paramElementTypes.emplace_back(param->get_element_type());
        paramShapes.emplace_back(param->get_shape());
    }

    auto inBuffers = std::vector<void *>(inputs.size());
    std::transform(inputs.cbegin(), inputs.cend(), inBuffers.begin(),
                   [](const std::vector<std::uint8_t> &input) {
                       // const_cast added to satisfy specialize_function interface
                       // which requires inputs as std::vector<void *>
                       return const_cast<std::uint8_t *>(input.data());
                   });

    const auto &foldedFunc = specialize_function(function, paramElementTypes, paramShapes, inBuffers);
    ngraph::pass::ConstantFolding().run_on_function(foldedFunc);
    for (const auto &op : foldedFunc->get_ops()) {
        NGRAPH_CHECK(op::is_constant(op) || op::is_output(op) || op::is_parameter(op),
                     "Function was not fully folded to constant state!\n",
                     "At least one non constant node with type ", op->get_type_name(),
                     " present in function.");
    }
    return foldedFunc;
}

std::vector<std::vector<std::uint8_t>> getConstData(const std::shared_ptr<Function> &function, std::vector<ngraph::element::Type_t> convertType) {
    size_t numOutputs = function->get_output_size();
    auto outputs = std::vector<std::vector<std::uint8_t>>(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        const auto &output = function->output(i).get_node_shared_ptr();
        NGRAPH_CHECK(output->inputs().size() == 1);
        auto parrentNode = output->input_value(0).get_node_shared_ptr();
        NGRAPH_CHECK(op::is_constant(parrentNode), "Function was not fully folded to constant state!\n",
                     "Parent node of one of results is not constant and has type ", parrentNode->get_type_name());

        const auto data = std::dynamic_pointer_cast<opset1::Constant>(parrentNode)->get_data_ptr<std::uint8_t>();
        const auto dataSize = shape_size(parrentNode->get_shape()) * parrentNode->get_element_type().size();
        outputs[i].resize(dataSize);
        std::copy(data, data + dataSize, outputs[i].data());
        if (!convertType.empty() && convertType[i] != element::Type_t::undefined && parrentNode->get_element_type() != element::Type(convertType[i]))
            outputs[i] = convertOutputPrecision(outputs[i], parrentNode->get_element_type(), convertType[i], shape_size(parrentNode->get_shape()));
    }
    return outputs;
}

namespace {

std::string toString(const NodeTypeInfo& typeInfo) {
    return std::string(typeInfo.name) + " ver. " + std::to_string(typeInfo.version);
}

void CompareShapes(const PartialShape& actual, const PartialShape& expected) {
    NGRAPH_CHECK(actual.relaxes(expected) && actual.refines(expected), "Functions compare: Different shape detected ", actual, " and ", expected);
}

void CompareNodes(const Node& actual, const Node& expected) {
    const auto& actualType   = actual.get_type_info();
    const auto& expectedType = expected.get_type_info();
    NGRAPH_CHECK(actualType == expectedType, "Functions compare: data types must be equal ", toString(actualType), " != ", toString(expectedType));

    const auto& numActualInputs = actual.inputs().size();
    const auto& numExpectedInputs = expected.inputs().size();
    NGRAPH_CHECK(numActualInputs == numExpectedInputs, "Functions compare: numbers of inputs are different: ", numActualInputs, " and ", numExpectedInputs);

    const auto& numActualOutputs = actual.outputs().size();
    const auto& numExpectedOutputs = expected.outputs().size();
    NGRAPH_CHECK(numActualOutputs == numExpectedOutputs, "Functions compare: numbers of outputs are different: ",
                 numActualOutputs, " and ", numExpectedOutputs);
}

}  // namespace

void CompareFunctions(const Function& actual, const Function& expected) {
    const auto& actualOrderedOps = actual.get_ordered_ops();
    const auto& expectedOrderedOps = expected.get_ordered_ops();

    NGRAPH_CHECK(expectedOrderedOps.size() == actualOrderedOps.size(),
                 "Functions compare: expected and actual ops number should be equal "
                 "but got ", expectedOrderedOps.size(), " and ", actualOrderedOps.size(), " respectively");

    for (std::size_t i = 0; i < expectedOrderedOps.size(); i++) {
        const auto& expectedOp = expectedOrderedOps[i];
        const auto& actualOp = actualOrderedOps[i];

        CompareNodes(*actualOp, *expectedOp);
        for (std::size_t i = 0; i < actualOp->inputs().size(); ++i) {
            const auto& actualShape = actualOp->input(i).get_partial_shape();
            const auto& expectedShape = expectedOp->input(i).get_partial_shape();
            CompareShapes(actualShape, expectedShape);
        }

        for (std::size_t i = 0; i < actualOp->outputs().size(); ++i) {
            const auto& actualShape = actualOp->output(i).get_partial_shape();
            const auto& expectedShape = expectedOp->output(i).get_partial_shape();
            CompareShapes(actualShape, expectedShape);
        }
    }
}

std::shared_ptr<ngraph::Node> getNodeSharedPtr(const ngraph::NodeTypeInfo &type_info, const ngraph::OutputVector &outputVector) {
    for (const auto& opset : {ngraph::get_opset3(), ngraph::get_opset2(), ngraph::get_opset1()})
        if (opset.contains_type(type_info)) {
            const auto ngraphNode = std::shared_ptr<ngraph::Node>(opset.create(type_info.name));
            ngraphNode->set_arguments(outputVector);
            ngraphNode->validate_and_infer_types();
            return ngraphNode;
        }
    NGRAPH_UNREACHABLE("supported opsets does not contain op with name: ", type_info.name, " version: ", type_info.version);
}

template <typename fromPrec, typename toPrec>
std::vector<std::uint8_t> convertPrecision(std::vector<std::uint8_t> &buffer, const size_t elementsCount, const size_t elementSize) {
    std::vector<std::uint8_t> convertedData(elementsCount * elementSize);
    const fromPrec *src = reinterpret_cast<const fromPrec *>(buffer.data());
    toPrec *dst = reinterpret_cast<toPrec *>(convertedData.data());
    for (size_t i = 0; i < elementsCount; i++)
        dst[i] = static_cast<toPrec>(src[i]);
    return convertedData;
}

bool is_tensor_iterator_exist(const std::shared_ptr<ngraph::Function> & func) {
    const auto& ops = func->get_ops();
    for (const auto& node : ops) {
        const auto& ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(node);
        if (ti) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> convertOutputPrecision(std::vector<std::uint8_t> &output, const element::Type_t &fromPrecision, const element::Type_t &toPrecision,
                                                                                                                                const size_t elementsCount) {
    switch (fromPrecision) {
        case element::Type_t::u8: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<uint8_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<uint8_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<uint8_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<uint8_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<uint8_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<uint8_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<uint8_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<uint8_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::u16: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<uint16_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<uint16_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<uint16_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<uint16_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<uint16_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<uint16_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<uint16_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<uint16_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::i8: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<int8_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<int8_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<int8_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<int8_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<int8_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<int8_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<int8_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<int8_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::i16: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<int16_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<int16_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<int16_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<int16_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<int16_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<int16_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<int16_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<int16_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::i32: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<int32_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<int32_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<int32_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<int32_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<int32_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<int32_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<int32_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<int32_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::i64: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<int64_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<int64_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<int64_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<int64_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<int64_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<int64_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<int64_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<int64_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::u64: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<uint64_t, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<uint64_t, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<uint64_t, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<uint64_t, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<uint64_t, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<uint64_t, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<uint64_t, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<uint64_t, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::f32: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<float, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<float, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<float, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<float, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<float, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<float, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<float, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f16: {
                // ngraph float16 has single ctor from float
              return convertPrecision<float, ngraph::float16>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<float, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                                                                                        element::Type(toPrecision).get_type_name());
            }
        }
        case element::Type_t::boolean: {
            switch (toPrecision) {
            case element::Type_t::u8: {
                return convertPrecision<char, uint8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u16: {
                return convertPrecision<char, uint16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i8: {
                return convertPrecision<char, int8_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i16: {
                return convertPrecision<char, int16_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i32: {
                return convertPrecision<char, int32_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::i64: {
                return convertPrecision<char, int64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::f32: {
                return convertPrecision<char, float>(output, elementsCount, element::Type(toPrecision).size());
            }
            case element::Type_t::u64: {
                return convertPrecision<char, uint64_t>(output, elementsCount, element::Type(toPrecision).size());
            }
            default:
                throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " to: " +
                                         element::Type(toPrecision).get_type_name());
            }
        }
        default:
            throw std::runtime_error("convertOutputPrecision can't convert from: " + element::Type(fromPrecision).get_type_name() + " precision");
    }
}

std::ostream& operator<<(std::ostream & os, ngraph::helpers::EltwiseTypes type) {
    switch (type) {
        case ngraph::helpers::EltwiseTypes::SUBTRACT:
            os << "Sub";
            break;
        case ngraph::helpers::EltwiseTypes::MULTIPLY:
            os << "Prod";
            break;
        case ngraph::helpers::EltwiseTypes::ADD:
            os << "Sum";
            break;
        case ngraph::helpers::EltwiseTypes::DIVIDE:
            os << "Div";
            break;
        case ngraph::helpers::EltwiseTypes::SQUARED_DIFF:
            os << "SqDiff";
            break;
        case ngraph::helpers::EltwiseTypes::POWER:
            os << "Pow";
            break;
        case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
            os << "FloorMod";
            break;
        case ngraph::helpers::EltwiseTypes::MOD:
            os << "Mod";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::helpers::SqueezeOpType type) {
    switch (type) {
        case ngraph::helpers::SqueezeOpType::SQUEEZE:
            os << "Squeeze";
            break;
        case ngraph::helpers::SqueezeOpType::UNSQUEEZE:
            os << "Unsqueeze";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ngraph::helpers::InputLayerType type) {
    switch (type) {
        case ngraph::helpers::InputLayerType::CONSTANT:
            os << "CONSTANT";
            break;
        case ngraph::helpers::InputLayerType::PARAMETER:
            os << "PARAMETER";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_INPUT_LAYER_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::helpers::ComparisonTypes type) {
    switch (type) {
        case ngraph::helpers::ComparisonTypes::EQUAL:
            os << "Equal";
            break;
        case ngraph::helpers::ComparisonTypes::NOT_EQUAL:
            os << "NotEqual";
            break;
        case ngraph::helpers::ComparisonTypes::GREATER:
            os << "Greater";
            break;
        case ngraph::helpers::ComparisonTypes::GREATER_EQUAL:
            os << "GreaterEqual";
            break;
        case ngraph::helpers::ComparisonTypes::LESS:
            os << "Less";
            break;
        case ngraph::helpers::ComparisonTypes::LESS_EQUAL:
            os << "LessEqual";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::helpers::LogicalTypes type) {
    switch (type) {
        case ngraph::helpers::LogicalTypes::LOGICAL_AND:
            os << "LogicalAnd";
            break;
        case ngraph::helpers::LogicalTypes::LOGICAL_OR:
            os << "LogicalOr";
            break;
        case ngraph::helpers::LogicalTypes::LOGICAL_NOT:
            os << "LogicalNot";
            break;
        case ngraph::helpers::LogicalTypes::LOGICAL_XOR:
            os << "LogicalXor";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::InterpolateMode type) {
    switch (type) {
        case ngraph::op::v4::Interpolate::InterpolateMode::cubic:
            os << "cubic";
            break;
        case ngraph::op::v4::Interpolate::InterpolateMode::linear:
            os << "linear";
            break;
        case ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx:
            os << "linear_onnx";
            break;
        case ngraph::op::v4::Interpolate::InterpolateMode::nearest:
            os << "nearest";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::CoordinateTransformMode type) {
    switch (type) {
        case ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners:
            os << "align_corners";
            break;
        case ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric:
            os << "asymmetric";
            break;
        case ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel:
            os << "half_pixel";
            break;
        case ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel:
            os << "pytorch_half_pixel";
            break;
        case ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn:
            os << "tf_half_pixel_for_nn";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::NearestMode type) {
    switch (type) {
        case ngraph::op::v4::Interpolate::NearestMode::ceil:
            os << "ceil";
            break;
        case ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil:
            os << "round_prefer_ceil";
            break;
        case ngraph::op::v4::Interpolate::NearestMode::floor:
            os << "floor";
            break;
        case ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor:
            os << "round_prefer_floor";
            break;
        case ngraph::op::v4::Interpolate::NearestMode::simple:
            os << "simple";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::ShapeCalcMode type) {
    switch (type) {
        case ngraph::op::v4::Interpolate::ShapeCalcMode::scales:
            os << "scales";
            break;
        case ngraph::op::v4::Interpolate::ShapeCalcMode::sizes:
            os << "sizes";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, TensorIteratorBody type) {
    switch (type) {
        case TensorIteratorBody::LSTM:
            os << "LSTM";
            break;
        case TensorIteratorBody::RNN:
            os << "RNN";
            break;
        case TensorIteratorBody::GRU:
            os << "GRU";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream & os, SequenceTestsMode type) {
    switch (type) {
        case SequenceTestsMode::PURE_SEQ:
            os << "PURE_SEQ";
            break;
        case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
            os << "CONVERT_TO_TI_RAND_SEQ_LEN_PARAM";
            break;
        case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
            os << "CONVERT_TO_TI_RAND_SEQ_LEN_CONST";
            break;
        case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM:
            os << "CONVERT_TO_TI_MAX_SEQ_LEN_PARAM";
            break;
        case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST:
            os << "CONVERT_TO_TI_MAX_SEQ_LEN_CONST";
            break;
        default:
            throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}
}  // namespace helpers
}  // namespace ngraph
