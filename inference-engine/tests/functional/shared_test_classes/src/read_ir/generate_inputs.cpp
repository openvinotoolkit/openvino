// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/ops.hpp"

#include "shared_test_classes/single_layer/roi_align.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/read_ir/generate_inputs.hpp"

namespace LayerTestsDefinitions {

namespace {
InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::Node> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

namespace Activation {
InferenceEngine::Blob::Ptr generate(const InferenceEngine::InputInfo& info,
                                    bool inPrcSigned,
                                    int32_t data_start_from = -10,
                                    uint32_t data_range = 20,
                                    int32_t resolution = 32768) {
    if (!inPrcSigned) {
        data_range = 15;
        data_start_from = 0;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range,
                                            data_start_from,
                                            resolution);
}
} // namespace Activation

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Abs> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Acos> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), -1, 2);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Asin> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), -1, 2);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Atan> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), -1, 2);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Ceiling> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), -1000, 2000);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Clamp> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Cos> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Cosh> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::DetectionOutput> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    InferenceEngine::Blob::Ptr blob;
    blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    int32_t resolution = 1;
    uint32_t range = 1;
    switch (port) {
        case 1:
        case 3:
            resolution = 1000;
            break;
        case 2:
            if (node->get_attrs().normalized) {
                resolution = 1000;
            } else {
                range = 10;
            }
            break;
        default:
            resolution = 10;
            break;
    }
    CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, range, 0, resolution);
    return blob;
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Elu> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Erf> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Exp> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Floor> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Gelu> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::HardSigmoid> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    switch (port) {
        case 1: {
            std::vector<float> alpha(node->get_input_shape(1).size(), 0.2f);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), alpha.data(), alpha.size());
        }
        case 2: {
            std::vector<float> beta(node->get_input_shape(2).size(), 0.5f);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), beta.data(), beta.size());
        }
        default: {
            return Activation::generate(info, node->get_input_element_type(0).is_signed());
        }
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::FakeQuantize> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    auto constShapes = node->get_input_shape(0);
    int seed = 1;
    size_t constDataSize = ngraph::shape_size(constShapes);
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
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), inputLowData.data(), inputLowData.size());
        case 2:
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), inputHighData.data(), inputHighData.size());
        case 3:
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), outputLowData.data(), outputLowData.size());
        case 4:
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), outputHighData.data(), outputHighData.size());
        default: {
            float resolution = 1.0f, min = +5.f, max = +25.f;
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), max - min, min, resolution, seed);
        }
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Log> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), 1, 20);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Negative> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::PRelu> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    switch (port) {
        case 1: {
            std::vector<float> negativeSlope(node->get_input_shape(1).size(), -0.01f);
            FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), negativeSlope.data(), negativeSlope.size());
        }
        default: {
            return Activation::generate(info, node->get_input_element_type(0).is_signed());
        }
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::PSROIPooling> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    const auto& inputShape = node->get_input_shape(0);
    if (port == 1) {
        InferenceEngine::Blob::Ptr blob;
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        PSROIPoolingLayerTest::fillROITensor(blob->buffer(),
                                             blob->size() / 5,
                                             inputShape[0],
                                             inputShape[2],
                                             inputShape[3],
                                             node->get_group_size(),
                                             node->get_spatial_scale(),
                                             node->get_spatial_bins_x(),
                                             node->get_spatial_bins_y(),
                                             node->get_mode());
        return blob;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::ROIPooling> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    const auto& inputShape = node->get_input_shape(0);
    if (port == 1) {
        InferenceEngine::Blob::Ptr blob;
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        CommonTestUtils::fill_data_roi<InferenceEngine::Precision::FP32>(blob,
                                                                         node->get_input_shape(0).front() - 1,
                                                                         inputShape[2],
                                                                         inputShape[3],
                                                                         1.0f,
                                                                         node->get_method() == "max");
        return blob;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Selu> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    switch (port) {
        case 1: {
            std::vector<float> alpha(node->get_input_shape(1).size(), 1.6732f);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), alpha.data(), alpha.size());
        }
        case 2: {
            std::vector<float> lambda(node->get_input_shape(1).size(), 1.0507f);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), lambda.data(), lambda.size());
        }
        default:
            return Activation::generate(info, node->get_input_element_type(0).is_signed());
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Sigmoid> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Sign> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Sin> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Sinh> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Sqrt> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), 1, 20);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Tan> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v0::Tanh> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::Divide> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128):
           FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 100, 101);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::FloorMod> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128):
           FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 4, 2);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::GatherTree> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    auto& shape = node->get_input_shape(0);
    auto maxBeamIndx = shape.at(2) - 1;

    switch (port) {
        case 2:
        case 3:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx, maxBeamIndx / 2);
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx);
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::LogicalAnd> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::LogicalNot> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::LogicalOr> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::LogicalXor> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v1::Power> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return info.getPrecision().is_float() ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128):
           FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 4, 2);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v3::Bucketize> node,
                                                   const InferenceEngine::InputInfo& info,
                                                   size_t port) {
    InferenceEngine::Blob::Ptr blobPtr;
    switch (port) {
        case 0: {
            auto data_shape = info.getTensorDesc().getDims();
            auto data_size = std::accumulate(begin(data_shape), end(data_shape), 1, std::multiplies<uint64_t>());
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_size * 5, 0, 10, 7235346);
        }
        case 1: {
            return FuncTestUtils::createAndFillBlobUniqueSequence(info.getTensorDesc(), 0, 10, 8234231);
        }
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v3::ROIAlign> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    const auto& inputShape = node->get_input_shape(0);
    switch (port) {
        case 1: {
            std::vector<float> blobData(node->get_shape()[0] * 4);
            ROIAlignLayerTest::fillCoordTensor(blobData,
                                               inputShape[2],
                                               inputShape[3],
                                               node->get_spatial_scale(),
                                               node->get_sampling_ratio(),
                                               node->get_pooled_h(),
                                               node->get_pooled_w());
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), blobData.data(), blobData.size());
        }
        case 2: {
            std::vector<int> roiIdxVector(node->get_shape()[0]);
            ROIAlignLayerTest::fillIdxTensor(roiIdxVector, node->get_shape()[0]);
            return FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), roiIdxVector.data(), roiIdxVector.size());
        }
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
    }
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v4::HSwish> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v4::Mish> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v4::Proposal> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    if (port == 0) {
        return FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 1, 0, 1000, 8234231);
    }
    return FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, 7235346);
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v4::SoftPlus> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v4::Swish> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::BatchNormInference node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), 3, 0, 1);
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::GRUSequence node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), m_max_seq_len, 0);
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v5::HSigmoid> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed());
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::Loop node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    auto tdesc = info.getTensorDesc();
    auto blob = make_blob_with_precision(tdesc);
    blob->allocate();

    if (tdesc.getLayout() == InferenceEngine::SCALAR) {
        auto scalar_1d = CommonTestUtils::make_reshape_view(blob, {1});
        unsigned int max_iter_num = 10;
        CommonTestUtils::fill_data_with_broadcast(scalar_1d, 0, {static_cast<float>(max_iter_num)});
    } else {
        int start_value = 7;
        CommonTestUtils::fill_data_with_broadcast(blob, 0, {static_cast<float>(start_value)});
    }
    return blob;
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::LSTMSequence node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), m_max_seq_len, 0);
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::NonMaxSuppression node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    if (port == 1) {
        InferenceEngine::Blob::Ptr blob;
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP32>(blob, 1, 0, 1000);
        return blob;
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const ngraph::op::v5::RNNSequence node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    if (port == 2) {
        unsigned int m_max_seq_len = 10;
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), m_max_seq_len, 0);
    }
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc());
}

InferenceEngine::Blob::Ptr generate(const std::shared_ptr<ngraph::op::v5::Round> node,
                                    const InferenceEngine::InputInfo& info,
                                    size_t port) {
    return Activation::generate(info, node->get_input_element_type(0).is_signed(), -10, 20, 4);
}

template<typename T>
InferenceEngine::Blob::Ptr generateInput(const std::shared_ptr<ngraph::Node> node,
                                         const InferenceEngine::InputInfo& info,
                                         size_t port) {
    return generate(ngraph::as_type_ptr<T>(node), info, port);
}
} // namespace

InputsMap getInputMap() {
    static InputsMap inputsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, generateInput<NAMESPACE::NAME>},
    #include "ngraph/opsets/opset1_tbl.hpp"
    #include "ngraph/opsets/opset2_tbl.hpp"
    #include "ngraph/opsets/opset3_tbl.hpp"
    #include "ngraph/opsets/opset4_tbl.hpp"
    #include "ngraph/opsets/opset5_tbl.hpp"
    #include "ngraph/opsets/opset6_tbl.hpp"
#undef NGRAPH_OP
    };
    return inputsMap;
}

} // namespace LayerTestsDefinitions
