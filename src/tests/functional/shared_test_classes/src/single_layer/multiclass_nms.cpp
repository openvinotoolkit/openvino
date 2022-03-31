// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/multiclass_nms.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

using namespace ngraph;
using namespace InferenceEngine;
using ngraph::helpers::operator<<;

std::string MulticlassNmsLayerTest::getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj) {
    std::vector<InputShape> shapes;
    InputPrecisions inPrecisions;
    int32_t nmsTopK, backgroundClass, keepTopK;
    element::Type outType;

    op::util::NmsBase::SortResultType sortResultType;

    InputfloatVar inFloatVar;
    InputboolVar inboolVar;

    std::string targetDevice;

    std::tie(shapes, inPrecisions, nmsTopK, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, targetDevice) = obj.param;

    ElementType paramsPrec, roisnumPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, roisnumPrec, maxBoxPrec, thrPrec) = inPrecisions;

    float iouThr, scoreThr, nmsEta;
    std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

    bool sortResCB, normalized;
    std::tie(sortResCB, normalized) = inboolVar;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
    }

    result << ")_paramsPrec=" << paramsPrec << "_roisnumPrec=" << roisnumPrec << "_maxBoxPrec=" << maxBoxPrec << "_thrPrec=" << thrPrec << "_";
    result << "nmsTopK=" << nmsTopK << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_backgroundClass=" << backgroundClass << "_";
    result << "keepTopK=" << keepTopK << "_outType=" << outType << "_";
    result << "sortResultType=" << sortResultType << "_sortResCrossBatch=" << sortResCB << "_nmsEta=" << nmsEta << "_normalized=" << normalized << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MulticlassNmsLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    ASSERT_TRUE(funcInputs.size() == 2 || funcInputs.size() == 3) << "Expected 3 inputs or 2 inputs.";
    for (int i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) { // scores
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
        } else if (i == 0) { // bboxes
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        } else { // roisnum
            /* sum of rois is no larger than num_bboxes. */
            ASSERT_TRUE(targetInputStaticShapes[i].size() == 1) << "Expected shape size 1 for input roisnum, got: " << targetInputStaticShapes[i];

            // returns num random values whose sum is max_num.
            auto _generate_roisnum = [](int num, int max_num) {
                std::vector<int> array;
                std::vector<int> results(num);

                array.push_back(0);
                for (auto i = 0; i < num-1; i++) {
                    array.push_back(std::rand() % max_num);
                }
                array.push_back(max_num);

                std::sort(array.begin(), array.end());

                for (auto i = 0; i < num; i++) {
                    results[i] = array[i+1] - array[i];
                }

                return results;
            };
            auto roisnum = _generate_roisnum(targetInputStaticShapes[i][0], targetInputStaticShapes[0][1]/*num_bboxes*/);

            tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            if (tensor.get_element_type() == ov::element::i32) {
                auto dataPtr = tensor.data<int32_t>();
                for (size_t i = 0; i < roisnum.size(); i++) {
                    dataPtr[i] = static_cast<int32_t>(roisnum[i]);
                }
            } else {
                auto dataPtr = tensor.data<int64_t>();
                for (size_t i = 0; i < roisnum.size(); i++) {
                    dataPtr[i] = static_cast<int64_t>(roisnum[i]);
                }
            }
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void MulticlassNmsLayerTest::GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch) {
    const auto& funcInputs = function->inputs();
    const auto& boxes_dims = inputs[funcInputs[0].get_node_shared_ptr()].get_shape();
    const auto& scores_dims = inputs[funcInputs[1].get_node_shared_ptr()].get_shape();

    const auto shared = (scores_dims.size() == 3);
    if (!shared)
        ASSERT_TRUE(funcInputs.size() > 2) << "Expected 3 inputs when input 'score' is 2D.";

    const auto& roisnum_dims = funcInputs.size() >2 ? inputs[funcInputs[2].get_node_shared_ptr()].get_shape() : Shape();

    const auto numBoxes = shared ? boxes_dims[1] : boxes_dims[1];
    auto numClasses = shared ? scores_dims[1] : boxes_dims[0];
    numBatches = shared ? scores_dims[0] : roisnum_dims[0];

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

void MulticlassNmsLayerTest::compare(const std::vector<ov::Tensor> &expectedOutputs,
                                     const std::vector<ov::Tensor> &actualOutputs) {
    auto batchIndex = -1; // output index for output 'selected_num'
    size_t numBatches(0), maxOutputBoxesPerBatch(0);
    GetOutputParams(numBatches, maxOutputBoxesPerBatch);
    std::vector<size_t> numPerBatch(numBatches);

    ASSERT_TRUE(expectedOutputs.size() == 3) << "Expect 3 outputs, got: " << expectedOutputs.size();

    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0; outputIndex--) {
        const auto& actual = actualOutputs[outputIndex];
        const auto _dims = actual.get_shape();
        if (_dims.size() == 1) { // 'selected_num'
            ASSERT_TRUE(_dims[0] == numBatches) << "Expect output 'selected_num' has shape of " << numBatches << ", got: " << _dims[0];
            batchIndex = outputIndex;
            if (actual.get_element_type() == ov::element::i32) {
                auto buffer = actual.data<int32_t>();
                std::copy_n(buffer, numBatches, numPerBatch.begin());
            } else {
                auto buffer = actual.data<int64_t>();
                std::copy_n(buffer, numBatches, numPerBatch.begin());
            }

            break;
        }
    }
    ASSERT_TRUE(batchIndex > -1) << "Expect to get output index for 'selected_num'";

    // reserve order could make sure output 'selected_num' get checked first.
    for (int outputIndex = static_cast<int>(expectedOutputs.size()) - 1; outputIndex >= 0; outputIndex--) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];
        const auto actualBuffer = static_cast<uint8_t*>(actual.data());
        const auto expectedBuffer = static_cast<uint8_t*>(expected.data());

        const auto expected_shape = expected.get_shape();
        const auto actual_shape = actual.get_shape();

        // Compare Selected Outputs & Selected Indices
        if (outputIndex != batchIndex) {
            if (m_outStaticShape) {
                ASSERT_TRUE(expected_shape[0] <= actual_shape[0]) << "Expected the compatible shape, got: " << expected_shape << " and " << actual_shape;
            } else {
                ASSERT_TRUE(expected_shape == actual_shape) << "Expected the same shape, got: " << expected_shape << " and " << actual_shape;
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
                        LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const float*>(expectedBuffer) + expected_offset * 6,
                                                                   reinterpret_cast<const float*>(actualBuffer) + actual_offset * 6, validNums * 6, 1e-5f);
                        break;
                    case ov::element::f64:
                        LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const double*>(expectedBuffer) + expected_offset * 6,
                                                                   reinterpret_cast<const float*>(actualBuffer) + actual_offset * 6, validNums * 6, 1e-5f);
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
                        LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int32_t*>(expectedBuffer) + expected_offset,
                                                                   reinterpret_cast<const int32_t*>(actualBuffer) + actual_offset, validNums, 0);
                        break;
                    case ov::element::i64:
                        LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int64_t*>(expectedBuffer) + expected_offset,
                                                                   reinterpret_cast<const int32_t*>(actualBuffer) + actual_offset, validNums, 0);
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
            ASSERT_TRUE(expected_shape == actual_shape) << "Expected the same shape, got: " << expected_shape << " and " << actual_shape;

            const auto& precision = actual.get_element_type();
            size_t size = expected.get_size();
            switch (precision) {
            case ov::element::i32: {
                switch (expected.get_element_type()) {
                case ov::element::i32:
                    LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int32_t*>(expectedBuffer), reinterpret_cast<const int32_t*>(actualBuffer),
                                                               size, 0);
                    break;
                case ov::element::i64:
                    LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int64_t*>(expectedBuffer), reinterpret_cast<const int32_t*>(actualBuffer),
                                                               size, 0);
                    break;
                default:
                    break;
                }
                break;
            }
            case ov::element::i64: {
                switch (expected.get_element_type()) {
                case ov::element::i32:
                    LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int32_t*>(expectedBuffer), reinterpret_cast<const int64_t*>(actualBuffer),
                                                               size, 0);
                    break;
                case ov::element::i64:
                    LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const int64_t*>(expectedBuffer), reinterpret_cast<const int64_t*>(actualBuffer),
                                                               size, 0);
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

void MulticlassNmsLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    InputPrecisions inPrecisions;
    size_t maxOutBoxesPerClass, backgroundClass, keepTopK;
    element::Type outType;

    op::util::NmsBase::SortResultType sortResultType;

    InputfloatVar inFloatVar;
    InputboolVar inboolVar;

    std::tie(shapes, inPrecisions, maxOutBoxesPerClass, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, targetDevice) =
        this->GetParam();

    init_input_shapes(shapes);

    // input is dynamic shape -> output will be dynamic shape
    // input is static shape -> output will be static shape
    const auto inputDynamicParam = {shapes[0].first, shapes[1].first};
    m_outStaticShape = std::any_of(inputDynamicParam.begin(), inputDynamicParam.end(), [](const ov::PartialShape& shape) {
        return shape.rank() == 0;
    });

    ElementType paramsPrec, roisnumPrec, maxBoxPrec, thrPrec;
    std::tie(paramsPrec, roisnumPrec, maxBoxPrec, thrPrec) = inPrecisions;

    float iouThr, scoreThr, nmsEta;
    std::tie(iouThr, scoreThr, nmsEta) = inFloatVar;

    bool sortResCB, normalized;
    std::tie(sortResCB, normalized) = inboolVar;

    ParameterVector params;
    if (inputDynamicShapes.size() > 2) {
        params = ngraph::builder::makeDynamicParams({paramsPrec, paramsPrec, roisnumPrec}, inputDynamicShapes);
    } else {
        params = ngraph::builder::makeDynamicParams(paramsPrec, inputDynamicShapes);
    }
    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    m_attrs.iou_threshold = iouThr;
    m_attrs.score_threshold = scoreThr;
    m_attrs.nms_eta = nmsEta;
    m_attrs.sort_result_type = sortResultType;
    m_attrs.sort_result_across_batch = sortResCB;
    m_attrs.output_type = outType;
    m_attrs.nms_top_k = maxOutBoxesPerClass;
    m_attrs.keep_top_k = keepTopK;
    m_attrs.background_class = backgroundClass;
    m_attrs.normalized = normalized;

    std::shared_ptr<opset9::MulticlassNms> nms;
    if (paramOuts.size() > 2) {
        nms = std::make_shared<opset9::MulticlassNms>(paramOuts[0], paramOuts[1], paramOuts[2], m_attrs);
    } else {
        nms = std::make_shared<opset9::MulticlassNms>(paramOuts[0], paramOuts[1], m_attrs);
    }

    if (!m_outStaticShape) {
        OutputVector results = {
            std::make_shared<opset5::Result>(nms->output(0)),
            std::make_shared<opset5::Result>(nms->output(1)),
            std::make_shared<opset5::Result>(nms->output(2))
        };
        function = std::make_shared<Function>(results, params, "MulticlassNMS");
    } else {
        auto nms_0_identity = std::make_shared<opset5::Multiply>(nms->output(0), opset5::Constant::create(paramsPrec, Shape {1}, {1}));
        auto nms_1_identity = std::make_shared<opset5::Multiply>(nms->output(1), opset5::Constant::create(outType, Shape {1}, {1}));
        auto nms_2_identity = std::make_shared<opset5::Multiply>(nms->output(2), opset5::Constant::create(outType, Shape {1}, {1}));
        OutputVector results = {
            std::make_shared<opset5::Result>(nms_0_identity),
            std::make_shared<opset5::Result>(nms_1_identity),
            std::make_shared<opset5::Result>(nms_2_identity)
        };
        function = std::make_shared<Function>(results, params, "MulticlassNMS");
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
