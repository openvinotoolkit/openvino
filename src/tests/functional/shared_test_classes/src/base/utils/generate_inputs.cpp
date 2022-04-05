// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>
#include "ngraph/ops.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "shared_test_classes/single_layer/roi_align.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace ov {
namespace test {
namespace utils {

namespace {
ov::runtime::Tensor generate(const std::shared_ptr<ov::Node>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    size_t inNodeCnt = node->get_input_size();
    InputGenerateData inGenData;
    if (elemType.is_real()) {
        inGenData.range = 2560;
        inGenData.resolution = 256;
    }
    auto it = inputRanges.find(node->get_type_info());
    if (it != inputRanges.end()) {
        const auto& ranges = it->second;
        if (ranges.size() != 2) {
            throw std::runtime_error("Incorrect size of ranges. It should be 2 (real and int cases)");
        }
        const auto& range = ranges.at(elemType.is_real());
        inGenData = range.size() < inNodeCnt ? range.front() : range.at(port);
    }
    return ov::test::utils::create_and_fill_tensor(elemType, targetShape, inGenData.range,
                                          inGenData.start_from, inGenData.resolution, inGenData.seed);
}

namespace Activation {
ov::runtime::Tensor generate(const ov::element::Type& elemType,
                             const ov::Shape& targetShape,
                             InputGenerateData inGenData = InputGenerateData(10, 20, 32768, 1)) {
    if (!elemType.is_signed()) {
        inGenData.range = 15;
        inGenData.start_from = 0;
    }
    return ov::test::utils::create_and_fill_tensor(elemType, targetShape, inGenData.range, inGenData.start_from, inGenData.resolution, inGenData.seed);
}
} // namespace Activation

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Abs>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Acos>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(-1, 2, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Asin>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(-1, 2, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Atan>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(-1, 2, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Ceiling>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(-1000, 2000, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Clamp>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Cos>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Cosh>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::DetectionOutput>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    InputGenerateData inGenData;
    inGenData.start_from = 0;
    inGenData.range = 1;

    switch (port) {
        case 1:
        case 3:
            inGenData.resolution = 1000;
            break;
        case 2:
            if (node->get_attrs().normalized) {
                inGenData.resolution = 1000;
            } else {
                inGenData.range = 10;
            }
            break;
        default:
            inGenData.resolution = 10;
            break;
    }
    return ov::test::utils::create_and_fill_tensor(elemType, targetShape, inGenData.range, inGenData.start_from, inGenData.resolution, inGenData.seed);
}


ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Elu>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Exp>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Floor>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Gelu>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::HardSigmoid>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    switch (port) {
        case 1: {
            std::vector<float> alpha(node->get_input_shape(1).size(), 0.2f);
            return ov::test::utils::create_tensor<float>(elemType, targetShape, alpha, alpha.size());
        }
        case 2: {
            std::vector<float> beta(node->get_input_shape(2).size(), 0.5f);
            return ov::test::utils::create_tensor<float>(elemType, targetShape, beta, beta.size());
        }
        default: {
            return Activation::generate(elemType, targetShape);
        }
    }

    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::FakeQuantize>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    int seed = 1;
    size_t constDataSize = ngraph::shape_size(targetShape);
    std::vector<float> inputLowData, inputHighData, outputLowData, outputHighData;
    inputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
    if (node->get_levels() != 2) {
        inputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
    } else {
        inputHighData = inputLowData;
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);

        for (int i = 0; i < constDataSize; i++) {
            if (outputLowData[i] > outputHighData[i]) {
                outputLowData[i] = 1;
                outputHighData[i] = 0;
            } else {
                outputLowData[i] = 0;
                outputHighData[i] = 1;
            }
        }
    }

    for (int i = 0; i < constDataSize; i++) {
        inputLowData[i] = std::min(inputLowData[i], inputHighData[i]);
        inputHighData[i] = std::max(inputLowData[i], inputHighData[i]);
        if (inputLowData[i] == inputHighData[i])
            inputHighData[i] += 1;
    }

    for (int i = 0; i < constDataSize; i++) {
        outputLowData[i] = std::min(outputLowData[i], outputHighData[i]);
        outputHighData[i] = std::max(outputLowData[i], outputHighData[i]);
        if (outputLowData[i] == outputHighData[i])
            outputHighData[i] += 1;
    }
    switch (port) {
        case 1:
            return ov::test::utils::create_tensor<float>(elemType, targetShape, inputLowData, inputLowData.size());
        case 2:
            return ov::test::utils::create_tensor<float>(elemType, targetShape, inputHighData, inputHighData.size());
        case 3:
            return ov::test::utils::create_tensor<float>(elemType, targetShape, outputLowData, outputLowData.size());
        case 4:
            return ov::test::utils::create_tensor<float>(elemType, targetShape, outputHighData, outputHighData.size());
        default: {
            float min = +5.f, max = +25.f;

            InputGenerateData inGenData;
            inGenData.range = max - min;
            inGenData.resolution = 1.0f;
            inGenData.seed = seed;

            return ov::test::utils::create_and_fill_tensor(elemType, targetShape, inGenData.range, inGenData.start_from, inGenData.resolution, inGenData.seed);
        }
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Log>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(1, 20, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Negative>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::PRelu>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    switch (port) {
        case 1: {
            std::vector<float> negativeSlope(node->get_input_shape(1).size(), -0.01f);
            return ov::test::utils::create_tensor<float>(elemType, targetShape, negativeSlope, negativeSlope.size());
        }
        default: {
            return Activation::generate(elemType, targetShape);
        }
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ov::op::v0::PSROIPooling>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    const auto &inputShape = node->get_input_shape(0);
    if (port == 1) {
        ov::runtime::Tensor tensor = ov::test::utils::create_and_fill_tensor(elemType, targetShape);
        LayerTestsDefinitions::PSROIPoolingLayerTest::fillROITensor(tensor.data<float>(),
                                                                    tensor.get_size() / 5,
                                                                    inputShape[0],
                                                                    inputShape[2],
                                                                    inputShape[3],
                                                                    node->get_group_size(),
                                                                    node->get_spatial_scale(),
                                                                    node->get_spatial_bins_x(),
                                                                    node->get_spatial_bins_y(),
                                                                    node->get_mode());
        return tensor;
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ov::op::v0::ROIPooling>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 1) {
        const auto &inputShape = node->get_input_shape(0);
        ov::runtime::Tensor tensor = ov::test::utils::create_and_fill_tensor(elemType, targetShape);
#define CASE(X) case X: ::CommonTestUtils::fill_roi_raw_ptr(                   \
    tensor.data<element_type_traits<X>::value_type>(),                         \
    tensor.get_size(),                                                         \
    node->get_input_shape(0).front() - 1,                                      \
    inputShape[2],                                                             \
    inputShape[3],                                                             \
    1.0f,                                                                      \
    node->get_method() == "max"); break;                                       \

    switch (elemType) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
        CASE(ov::element::Type_t::u1)
        CASE(ov::element::Type_t::i4)
        CASE(ov::element::Type_t::u4)
        default: OPENVINO_UNREACHABLE("Unsupported element type: ", elemType);
    }
#undef CASE
        return tensor;
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Selu>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    switch (port) {
        case 1: {
            std::vector<float> alpha(node->get_input_shape(1).size(), 1.6732f);
            return ov::test::utils::create_tensor<float>(elemType, targetShape, alpha, alpha.size());
        }
        case 2: {
            std::vector<float> lambda(node->get_input_shape(2).size(), 1.0507f);
            return ov::test::utils::create_tensor<float>(elemType, targetShape, lambda, lambda.size());
        }
        default: {
            return Activation::generate(elemType, targetShape);
        }
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Sigmoid>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Sign>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Sin>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Sinh>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Sqrt>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(1, 20, 32768, 1));
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Tan>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v0::Tanh>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::GatherTree>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    auto &shape = node->get_input_shape(0);
    auto maxBeamIndx = shape.at(2) - 1;

    switch (port) {
        case 2:
        case 3: {
            InputGenerateData inGenData;
            inGenData.start_from = maxBeamIndx / 2;
            inGenData.range = maxBeamIndx;
            return ov::test::utils::create_and_fill_tensor(elemType, targetShape, inGenData.range, inGenData.start_from, inGenData.resolution, inGenData.seed);
        }
        default:
            return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
    }
}

namespace LogicalOp {
ov::runtime::Tensor generate(const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return create_and_fill_tensor(elemType, targetShape, 2, 0);
}
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::LogicalAnd>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::LogicalNot>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::LogicalOr>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::LogicalXor>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::ReduceLogicalAnd>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v1::ReduceLogicalOr>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return LogicalOp::generate(elemType, targetShape);
}



ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v3::Bucketize>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    InferenceEngine::Blob::Ptr blobPtr;
    switch (port) {
        case 0: {
            auto data_size = shape_size(targetShape);
            return create_and_fill_tensor(elemType, targetShape, data_size * 5, 0, 10, 7235346);
        }
        case 1: {
            return  create_and_fill_tensor_unique_sequence(elemType, targetShape, 0, 10, 8234231);
        }
        default:
            return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ov::op::v3::ROIAlign>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    switch (port) {
        case 1: {
            if (node->get_sampling_ratio() != 0) {
                const auto &inputShape = node->get_input_shape(0);
                std::vector<float> blobData(node->get_shape()[0] * 4);
                LayerTestsDefinitions::ROIAlignLayerTest::fillCoordTensor(blobData,
                                                                          inputShape[2],
                                                                          inputShape[3],
                                                                          node->get_spatial_scale(),
                                                                          node->get_sampling_ratio(),
                                                                          node->get_pooled_h(),
                                                                          node->get_pooled_w());
                return ov::test::utils::create_tensor<float>(ov::element::f32, targetShape, blobData);
            } else {
                return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
            }
        }
        case 2: {
            std::vector<int> roiIdxVector(node->get_shape()[0]);
            LayerTestsDefinitions::ROIAlignLayerTest::fillIdxTensor(roiIdxVector, node->get_shape()[0]);
            return ov::test::utils::create_tensor<int>(elemType, targetShape, roiIdxVector);
        }
        default:
            return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v4::HSwish>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v4::Mish>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v4::Proposal>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 1) {
        return ov::test::utils::create_and_fill_tensor_normal_distribution(elemType, targetShape, 0.0f, 0.2f, 7235346);
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v4::SoftPlus>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v4::Swish>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::BatchNormInference>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return ov::test::utils::create_and_fill_tensor_consistently(elemType, targetShape, 3, 0, 1);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::GRUSequence>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, m_max_seq_len, 0);
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::HSigmoid>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::LSTMSequence>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, m_max_seq_len, 0);
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::NonMaxSuppression>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    switch (port) {
        case 1: {
            ov::runtime::Tensor tensor = ov::runtime::Tensor(elemType, targetShape);

            const size_t range = 1;
            const size_t startFrom = 0;
            const size_t k = 1000;
            const int seed = 1;
            std::default_random_engine random(seed);
            std::uniform_int_distribution<int32_t> distribution(k * startFrom, k * (startFrom + range));

            auto *dataPtr = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); i++) {
                auto value = static_cast<float>(distribution(random));
                dataPtr[i] = value / static_cast<float>(k);
            }
            return tensor;
        }
        default:
            return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
    }
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v3::EmbeddingSegmentsSum>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 2) {
        ov::runtime::Tensor tensor = ov::runtime::Tensor(elemType, targetShape);

        const auto &outputShape = node->get_output_shape(0);
        const size_t range = outputShape[0] - 1; // values in segmentsIds should be less than num_segments
        const size_t startFrom = 0;
        const int seed = 1;
        std::default_random_engine random(seed);
        switch (elemType) {
            case element::Type_t::i32: {
                std::uniform_int_distribution<int32_t> distribution(startFrom, (startFrom + range));

                auto *dataPtr = tensor.data<int32_t>();
                for (size_t i = 0; i < tensor.get_size(); i++) {
                    dataPtr[i] = distribution(random);
                }
                return tensor;
            }
            case element::Type_t::i64: {
                std::uniform_int_distribution<int64_t> distribution(startFrom, (startFrom + range));

                auto *dataPtr = tensor.data<int64_t>();
                for (size_t i = 0; i < tensor.get_size(); i++) {
                    dataPtr[i] = distribution(random);
                }
                return tensor;
            }
            default:
                OPENVINO_UNREACHABLE("Unsupported element type for segment_ids: ", elemType);
        }
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

template<ov::element::Type_t elemType>
ov::runtime::Tensor generate_unique_possibilities(const ov::Shape &targetShape) {
    using value_type = typename element_type_traits<elemType>::value_type;
    ov::runtime::Tensor tensor = ov::runtime::Tensor(elemType, targetShape);
    const size_t k = targetShape[0];
    std::vector<size_t> indices(k);
    std::iota(indices.begin(), indices.end(), 0lu);
    std::default_random_engine random;
    std::shuffle(indices.begin(), indices.end(), random);

    auto dataPtr = tensor.data<value_type>();
    for (size_t i = 0; i < k; ++i) {
        // our goal is to have unique values for both f32 and f16 to avoid false failures because of the same possibilities
        dataPtr[i] = ov::float16::from_bits(static_cast<  uint16_t>(indices[i]));
    }
    return tensor;
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronTopKROIs>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 1) {
        switch (elemType) {
            case element::Type_t::f16:
                return generate_unique_possibilities<element::Type_t::f16>(targetShape);
            case element::Type_t::f32:
                return generate_unique_possibilities<element::Type_t::f32>(targetShape);
            default:
                OPENVINO_UNREACHABLE("Unsupported element type: ", elemType);
        }
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}


ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::RNNSequence>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return ov::test::utils::create_and_fill_tensor(elemType, targetShape, m_max_seq_len, 0);
    }
    return generate(std::dynamic_pointer_cast<ov::Node>(node), port, elemType, targetShape);
}

ov::runtime::Tensor generate(const std::shared_ptr<ngraph::op::v5::Round>& node,
                             size_t port,
                             const ov::element::Type& elemType,
                             const ov::Shape& targetShape) {
    return Activation::generate(elemType, targetShape, InputGenerateData(-10, 20, 4));
}

template<typename T>
ov::runtime::Tensor generateInput(const std::shared_ptr<ov::Node>& node,
                                  size_t port,
                                  const ov::element::Type& elemType,
                                  const ov::Shape& targetShape) {
    return generate(ngraph::as_type_ptr<T>(node), port, elemType, targetShape);
}
} // namespace

InputsMap getInputMap() {
    static InputsMap inputsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateInput<NAMESPACE::NAME>},

#include "ngraph/opsets/opset1_tbl.hpp"
#include "ngraph/opsets/opset2_tbl.hpp"
#include "ngraph/opsets/opset3_tbl.hpp"
#include "ngraph/opsets/opset4_tbl.hpp"
#include "ngraph/opsets/opset5_tbl.hpp"
#include "ngraph/opsets/opset6_tbl.hpp"
#include "ngraph/opsets/opset7_tbl.hpp"
#include "ngraph/opsets/opset8_tbl.hpp"

#undef NGRAPH_OP
    };
    return inputsMap;
}

} // namespace utils
} // namespace test
} // namespace ov
