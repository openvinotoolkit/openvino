// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/matrix_nms.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ov {
namespace test {
std::string MatrixNmsLayerTest::getTestCaseName(const testing::TestParamInfo<NmsParams>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    op::v8::MatrixNms::SortResultType sort_result_type;
    ov::element::Type out_type;
    int backgroudClass;
    op::v8::MatrixNms::DecayFunction decayFunction;
    TopKParams top_k_params;
    ThresholdParams threshold_params;
    bool normalized;
    std::string target_device;
    std::tie(shapes, model_type, sort_result_type, out_type, top_k_params, threshold_params,
        backgroudClass, normalized, decayFunction, target_device) = obj.param;

    int nms_top_k, keep_top_k;
    std::tie(nms_top_k, keep_top_k) = top_k_params;

    float score_threshold, gaussian_sigma, post_threshold;
    std::tie(score_threshold, gaussian_sigma, post_threshold) = threshold_params;

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

    using ov::test::utils::operator<<;
    result << ")_model_type=" << model_type << "_";
    result << "sortResultType=" << sort_result_type << "_normalized=" << normalized << "_";
    result << "out_type=" << out_type << "_nms_top_k=" << nms_top_k << "_keep_top_k=" << keep_top_k << "_";
    result << "backgroudClass=" << backgroudClass << "_decayFunction=" << decayFunction << "_";
    result << "score_threshold=" << score_threshold << "_gaussian_sigma=" << gaussian_sigma << "_";
    result << "post_threshold=" << post_threshold <<"_TargetDevice=" << target_device;
    return result.str();
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
    if (targetDevice != ov::test::utils::DEVICE_GPU) {
        SubgraphBaseTest::compare(expectedOutputs, actualOutputs);
        return;
    }
    std::cout << actualOutputs[2].get_element_type() << std::endl;
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
                        const auto fBuffer = static_cast<float*>(actual.data());
                        for (size_t tailing = validNums * 6; tailing < maxOutputBoxesPerBatch * 6; tailing++) {
                            ASSERT_TRUE(std::abs(fBuffer[(actual_offset * 6 + tailing)] - -1.f) < 1e-5)
                                << "Invalid default value: " << fBuffer[i] << " at index: " << i;
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
                        const auto iBuffer = actual.data<int32_t>();
                        for (size_t tailing = validNums; tailing < maxOutputBoxesPerBatch; tailing++) {
                            ASSERT_TRUE(iBuffer[actual_offset + tailing] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
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
                        const auto iBuffer = actual.data<int64_t>();
                        for (size_t tailing = validNums; tailing < maxOutputBoxesPerBatch; tailing++) {
                            ASSERT_TRUE(iBuffer[actual_offset + tailing] == -1) << "Invalid default value: " << iBuffer[i] << " at index: " << i;
                        }
                        break;
                    }
                    default:
                        FAIL() << "Comparator for " << precision << " precision isn't supported";
                }
                expected_offset += validNums;
                actual_offset += maxOutputBoxesPerBatch;
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
    ov::element::Type model_type;
    TopKParams top_k_params;
    ThresholdParams threshold_params;

    std::tie(shapes, model_type, m_attrs.sort_result_type, m_attrs.output_type, top_k_params, threshold_params,
        m_attrs.background_class, m_attrs.normalized, m_attrs.decay_function, targetDevice) = this->GetParam();

    std::tie(m_attrs.nms_top_k, m_attrs.keep_top_k) = top_k_params;
    std::tie(m_attrs.score_threshold, m_attrs.gaussian_sigma, m_attrs.post_threshold) = threshold_params;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }
    auto nms = std::make_shared<ov::op::v8::MatrixNms>(params[0], params[1], m_attrs);

    function = std::make_shared<ov::Model>(nms, params, "MatrixNMS");
}
} // namespace test
} // namespace ov
