// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/matrix_nms.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

using namespace ngraph;
using namespace InferenceEngine;
using ngraph::helpers::operator<<;

std::string MatrixNmsLayerTest::getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
    std::vector<InputShape> shapes;
    InputPrecisions inPrecisions;
    op::v8::MatrixNms::SortResultType sortResultType;
    element::Type outType;
    int backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    TopKParams topKParams;
    ThresholdParams thresholdParams;
    bool normalized;
    bool outStaticShape;
    std::string targetDevice;
    std::tie(shapes, inPrecisions, sortResultType, outType, topKParams, thresholdParams,
        backgroudClass, normalized, decayFunction, outStaticShape, targetDevice) = obj.param;

    ElementType paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;

    int nmsTopK, keepTopK;
    std::tie(nmsTopK, keepTopK) = topKParams;

    float score_threshold, gaussian_sigma, post_threshold;
    std::tie(score_threshold, gaussian_sigma, post_threshold) = thresholdParams;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }

    result << ")_paramsPrec=" << paramsPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "sortResultType=" << sortResultType << "_normalized=" << normalized << "_";
    result << "outType=" << outType << "_nmsTopK=" << nmsTopK << "_keepTopK=" << keepTopK << "_";
    result << "backgroudClass=" << backgroudClass << "_decayFunction=" << decayFunction << "_";
    result << "score_threshold=" << score_threshold << "_gaussian_sigma=" << gaussian_sigma << "_";
    result << "post_threshold=" << post_threshold << "_outStaticShape=" << outStaticShape <<"_TargetDevice=" << targetDevice;
    return result.str();
}

void MatrixNmsLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) {
            tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);

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
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void MatrixNmsLayerTest::GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch) {
    size_t it = 0;
    size_t numBoxes = 0, numClasses = 0;
    const auto& funcInputs = function->inputs();
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        const auto& dims = inputs[funcInput.get_node_shared_ptr()].get_shape();

        if (it == 1) {
            numClasses = dims[1];
        } else {
            numBatches = dims[0];
            numBoxes = dims[1];
        }
        it++;
    }

    ASSERT_TRUE(numBatches > 0 && numBoxes > 0 && numClasses > 0)
        << "Expected numBatches, numBoxes, numClasses > 0, got:" << numBatches << ", " << numBoxes << ", " << numClasses;

    auto realClasses = numClasses;
    if (m_attrs.background_class >= 0 && m_attrs.background_class < numClasses) {
       realClasses = realClasses - 1;
    }

    size_t maxOutputBoxesPerClass = 0;
    if (m_attrs.nms_top_k >= 0)
       maxOutputBoxesPerClass = std::min(numBoxes, static_cast<size_t>(m_attrs.nms_top_k));
    else
       maxOutputBoxesPerClass = numBoxes;

    maxOutputBoxesPerBatch  = maxOutputBoxesPerClass * realClasses;
    if (m_attrs.keep_top_k >= 0)
       maxOutputBoxesPerBatch =
               std::min(maxOutputBoxesPerBatch, static_cast<size_t>(m_attrs.keep_top_k));
}

void MatrixNmsLayerTest::compare(const std::vector<ov::Tensor> &expectedOutputs,
                                 const std::vector<ov::Tensor> &actualOutputs) {
    auto batchIndex = -1;
    size_t numBatches(0), maxOutputBoxesPerBatch(0);
    GetOutputParams(numBatches, maxOutputBoxesPerBatch);
    std::vector<int32_t> numPerBatch(numBatches);
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& actual = actualOutputs[outputIndex];
        const auto _dims = actual.get_shape();
        if (_dims.size() == 1 && _dims[0] == numBatches) {
            batchIndex = outputIndex;
            if (actual.get_element_type() == ov::element::i32) {
                auto buffer = actual.data<int32_t>();
                std::copy_n(buffer, numBatches, numPerBatch.begin());
            } else {
                auto buffer = actual.data<int64_t>();
                std::copy_n(buffer, numBatches, numPerBatch.begin());
            }
        }
    }

    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0 ; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];
        const auto actualBuffer = static_cast<uint8_t*>(actual.data());
        const auto expectedBuffer = static_cast<uint8_t*>(expected.data());

        //Compare Selected Outputs & Selected Indices
        if (outputIndex != batchIndex) {
            if (outputIndex == 2) {
                if (expected.get_size() != actual.get_size())
                    throw std::runtime_error("Expected and actual size 3rd output have different size");
            }

            const auto& precision = actual.get_element_type();
            auto expected_offset = 0;
            auto actual_offset = 0;
            for (size_t i = 0; i < numPerBatch.size(); i++) {
                auto validNums = numPerBatch[i];
                switch (precision) {
                    case ov::element::f32: {
                        switch (expected.get_element_type()) {
                            case ov::element::f32:
                                LayerTestsUtils::LayerTestsCommon::Compare(
                                        reinterpret_cast<const float *>(expectedBuffer) + expected_offset * 6,
                                        reinterpret_cast<const float *>(actualBuffer) + actual_offset * 6, validNums * 6, 1e-5f);
                                break;
                            case ov::element::f64:
                                LayerTestsUtils::LayerTestsCommon::Compare(
                                        reinterpret_cast<const double *>(expectedBuffer) + expected_offset * 6,
                                        reinterpret_cast<const float *>(actualBuffer) + actual_offset * 6, validNums *6, 1e-5f);
                                break;
                            default:
                                break;
                        }
                        if (m_outStaticShape) {
                            const auto fBuffer = static_cast<float*>(actual.data());
                            for (size_t tailing = validNums * 6; tailing < maxOutputBoxesPerBatch * 6; tailing++) {
                                ASSERT_TRUE(std::abs(fBuffer[(actual_offset * 6 + tailing)] - -1.f) < 1e-5)
                                    << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                            }
                        }
                        break;
                    }
                    case ov::element::i32: {
                        switch (expected.get_element_type()) {
                            case ov::element::i32:
                                LayerTestsUtils::LayerTestsCommon::Compare(
                                        reinterpret_cast<const int32_t *>(expectedBuffer) + expected_offset,
                                        reinterpret_cast<const int32_t *>(actualBuffer) + actual_offset, validNums, 0);
                                break;
                            case ov::element::i64:
                                LayerTestsUtils::LayerTestsCommon::Compare(
                                        reinterpret_cast<const int64_t *>(expectedBuffer) + expected_offset,
                                        reinterpret_cast<const int32_t *>(actualBuffer) + actual_offset, validNums, 0);
                                break;
                            default:
                                break;
                        }
                        if (m_outStaticShape) {
                            const auto iBuffer = actual.data<int32_t>();
                            for (size_t tailing = validNums; tailing < maxOutputBoxesPerBatch; tailing++) {
                                ASSERT_TRUE(iBuffer[actual_offset + tailing] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                            }
                        }
                        break;
                    }
                    case ov::element::i64: {
                        switch (expected.get_element_type()) {
                        case ov::element::i32:
                            LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int32_t*>(expectedBuffer) + expected_offset,
                                                                       reinterpret_cast<const int64_t*>(actualBuffer) + actual_offset, validNums, 0);
                            break;
                        case ov::element::i64:
                            LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int64_t*>(expectedBuffer) + expected_offset,
                                                                       reinterpret_cast<const int64_t*>(actualBuffer) + actual_offset, validNums, 0);
                            break;
                        default:
                            break;
                        }
                        if (m_outStaticShape) {
                            const auto iBuffer = actual.data<int64_t>();
                            for (size_t tailing = validNums; tailing < maxOutputBoxesPerBatch; tailing++) {
                                ASSERT_TRUE(iBuffer[actual_offset + tailing] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                            }
                        }
                        break;
                    }
                    default:
                        FAIL() << "Comparator for " << precision << " precision isn't supported";
                }
                if (!m_outStaticShape) {
                    expected_offset += validNums;
                    actual_offset += validNums;
                } else {
                    expected_offset += validNums;
                    actual_offset += maxOutputBoxesPerBatch;
                }
            }
        } else {
            if (outputIndex == 2) {
                if (expected.get_size() != actual.get_size())
                    throw std::runtime_error("Expected and actual size 3rd output have different size");
            }

            const auto& precision = actual.get_element_type();
            size_t size = expected.get_size();
            switch (precision) {
                case ov::element::i32: {
                    switch (expected.get_element_type()) {
                        case ov::element::i32:
                            LayerTestsUtils::LayerTestsCommon::Compare(
                                    reinterpret_cast<const int32_t *>(expectedBuffer),
                                    reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                            break;
                        case ov::element::i64:
                            LayerTestsUtils::LayerTestsCommon::Compare(
                                    reinterpret_cast<const int64_t *>(expectedBuffer),
                                    reinterpret_cast<const int32_t *>(actualBuffer), size, 0);
                            break;
                        default:
                            break;
                    }
                    break;
                }
                case ov::element::i64: {
                    switch (expected.get_element_type()) {
                    case ov::element::i32:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                            reinterpret_cast<const int32_t*>(expectedBuffer),
                            reinterpret_cast<const int64_t*>(actualBuffer), size, 0);
                        break;
                    case ov::element::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(
                            reinterpret_cast<const int64_t*>(expectedBuffer),
                            reinterpret_cast<const int64_t*>(actualBuffer), size, 0);
                        break;
                    default:
                        break;
                    }
                    break;
                }
                default:
                    FAIL() << "Comparator for " << precision << " precision isn't supported";
            }
        }
    }
}

void MatrixNmsLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    InputPrecisions inPrecisions;
    TopKParams topKParams;
    ThresholdParams thresholdParams;

    std::tie(shapes, inPrecisions, m_attrs.sort_result_type, m_attrs.output_type, topKParams, thresholdParams,
        m_attrs.background_class, m_attrs.normalized, m_attrs.decay_function, m_outStaticShape, targetDevice) = this->GetParam();

    std::tie(m_attrs.nms_top_k, m_attrs.keep_top_k) = topKParams;
    std::tie(m_attrs.score_threshold, m_attrs.gaussian_sigma, m_attrs.post_threshold) = thresholdParams;

    init_input_shapes(shapes);

    ElementType paramsPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, maxBoxPrec, thrPrec) = inPrecisions;
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(paramsPrec, shape));
    }
    auto nms = std::make_shared<opset8::MatrixNms>(params[0], params[1], m_attrs);

    function = std::make_shared<Function>(nms, params, "MatrixNMS");
}

} // namespace subgraph
} // namespace test
} // namespace ov
