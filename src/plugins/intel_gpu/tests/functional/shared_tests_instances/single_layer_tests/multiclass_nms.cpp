// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/op/multiclass_nms.hpp"

namespace ov {
namespace test {
namespace GPULayerTestsDefinitions {

using InputTypes = std::tuple<ov::element::Type,   // input 'boxes' and 'scores' types
                              ov::element::Type>;  // iou_threshold, score_threshold, soft_nms_sigma precisions

using InputfloatVar = std::tuple<float,   // iouThreshold
                                 float,   // scoreThreshold
                                 float>;  // nmsEta

using InputboolVar = std::tuple<bool,   // nmsEta
                                bool>;  // normalized

using MulticlassNmsParamsGPU = std::tuple<std::vector<ov::test::InputShape>,            // Params using to create inputs
                                       InputTypes,                                      // Input precisions
                                       int32_t,                                         // Max output boxes per class
                                       InputfloatVar,                                   // iouThreshold, scoreThreshold, nmsEta
                                       int32_t,                                         // background_class
                                       int32_t,                                         // keep_top_k
                                       ov::element::Type,                               // Output type
                                       ov::op::util::MulticlassNmsBase::SortResultType, // SortResultType
                                       InputboolVar,                                    // Sort result across batch, normalized
                                       std::string>;

class MulticlassNmsLayerTestGPU : public testing::WithParamInterface<MulticlassNmsParamsGPU>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MulticlassNmsParamsGPU>& obj);

protected:
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

    void SetUp() override;
    void GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch);
    ov::op::util::MulticlassNmsBase::Attributes m_attrs;
    bool use_op_v8 = false;
};

class MulticlassNmsLayerTestGPU8 : public MulticlassNmsLayerTestGPU {
protected:
    void SetUp() override;
};

TEST_P(MulticlassNmsLayerTestGPU, Inference) {
    run();
};

TEST_P(MulticlassNmsLayerTestGPU8, Inference) {
    run();
};

std::string MulticlassNmsLayerTestGPU::getTestCaseName(const testing::TestParamInfo<MulticlassNmsParamsGPU>& obj) {
    const auto& [shapes, input_types, nmsTopK, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, targetDevice] = obj.param;

    const auto& [paramsPrec, roisnumPrec] = input_types;

    const auto& [iouThr, scoreThr, nmsEta] = inFloatVar;

    const auto& [sortResCB, normalized] = inboolVar;

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
    result << ")_paramsPrec=" << paramsPrec << "_roisnumPrec=" << roisnumPrec << "_";
    result << "nmsTopK=" << nmsTopK << "_";
    result << "iouThr=" << iouThr << "_scoreThr=" << scoreThr << "_backgroundClass=" << backgroundClass << "_";
    result << "keepTopK=" << keepTopK << "_outType=" << outType << "_";
    result << "sortResultType=" << sortResultType << "_sortResCrossBatch=" << sortResCB << "_nmsEta=" << nmsEta << "_normalized=" << normalized << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MulticlassNmsLayerTestGPU::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    ASSERT_TRUE(funcInputs.size() == 2 || funcInputs.size() == 3) << "Expected 3 inputs or 2 inputs.";
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) { // scores
            const size_t range = 1;
            const size_t start_from = 0;
            const size_t k = 1000;
            const int seed = 1;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                             ov::test::utils::InputGenerateData(start_from, range, k, seed));
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


void MulticlassNmsLayerTestGPU::GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch) {
    const auto& funcInputs = function->inputs();
    const auto& boxes_dims = inputs[funcInputs[0].get_node_shared_ptr()].get_shape();
    const auto& scores_dims = inputs[funcInputs[1].get_node_shared_ptr()].get_shape();

    const auto shared = (scores_dims.size() == 3);
    if (!shared) {
        ASSERT_TRUE(funcInputs.size() > 2) << "Expected 3 inputs when input 'score' is 2D.";
    }

    const auto& roisnum_dims = funcInputs.size() >2 ? inputs[funcInputs[2].get_node_shared_ptr()].get_shape() : Shape();

    const auto numBoxes = shared ? boxes_dims[1] : boxes_dims[1];
    auto numClasses = shared ? scores_dims[1] : boxes_dims[0];
    numBatches = shared ? scores_dims[0] : roisnum_dims[0];

    ASSERT_TRUE(numBatches > 0 && numBoxes > 0 && numClasses > 0)
        << "Expected numBatches, numBoxes, numClasses > 0, got:" << numBatches << ", " << numBoxes << ", " << numClasses;

    auto realClasses = numClasses;
    if (m_attrs.background_class >= 0 && m_attrs.background_class < static_cast<int>(numClasses)) {
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

void MulticlassNmsLayerTestGPU::compare(const std::vector<ov::Tensor> &expectedOutputs,
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
        const auto actualBuffer = static_cast<const uint8_t*>(actual.data());
        const auto expectedBuffer = static_cast<const uint8_t*>(expected.data());

        const auto expected_shape = expected.get_shape();
        const auto actual_shape = actual.get_shape();

        // Compare Selected Outputs & Selected Indices
        if (outputIndex != batchIndex) {
            ASSERT_TRUE(expected_shape[0] <= actual_shape[0]) << "Expected the compatible shape, got: " << expected_shape << " and " << actual_shape;

            const auto& precision = actual.get_element_type();
            auto expected_offset = 0;
            auto actual_offset = 0;
            for (size_t i = 0; i < numPerBatch.size(); i++) {
                auto validNums = numPerBatch[i];
#define TENSOR_COMPARE(elem_type, act_type, expected_offset, actual_offset, size, _threshold)                              \
    case ov::element::Type_t::elem_type: {                                                                                 \
        using tensor_type = ov::fundamental_type_for<ov::element::Type_t::elem_type>;                                      \
        using actual_type = ov::fundamental_type_for<ov::element::Type_t::act_type>;                                       \
        ov::test::utils::compare_raw_data(reinterpret_cast<const tensor_type*>(expectedBuffer) + expected_offset, \
                                          reinterpret_cast<const actual_type*>(actualBuffer) + actual_offset,     \
                                          size, _threshold);                                                      \
        break;                                                                                                             \
    }
                switch (precision) {
                case ov::element::f32: {
                    switch (expected.get_element_type()) {
                    TENSOR_COMPARE(f32, f32, expected_offset * 6, actual_offset * 6, validNums * 6, 1e-5f)
                    TENSOR_COMPARE(f64, f32, expected_offset * 6, actual_offset * 6, validNums * 6, 1e-5f)
                    default:
                        break;
                    }

                    const auto fBuffer = static_cast<const float*>(actual.data());
                    for (size_t tailing = validNums * 6; tailing < maxOutputBoxesPerBatch * 6; tailing++) {
                        ASSERT_TRUE(std::abs(fBuffer[(actual_offset * 6 + tailing)] - -1.f) < 1e-5)
                            << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                    }
                    break;
                }
                case ov::element::i32: {
                    switch (expected.get_element_type()) {
                    TENSOR_COMPARE(i32, i32, expected_offset, actual_offset, validNums, 0)
                    TENSOR_COMPARE(i64, i32, expected_offset, actual_offset, validNums, 0)
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
                    TENSOR_COMPARE(i32, i64, expected_offset, actual_offset, validNums, 0)
                    TENSOR_COMPARE(i64, i64, expected_offset, actual_offset, validNums, 0)
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
            ASSERT_TRUE(expected_shape == actual_shape) << "Expected the same shape, got: " << expected_shape << " and " << actual_shape;

            const auto& precision = actual.get_element_type();
            size_t size = expected.get_size();
            switch (precision) {
            case ov::element::i32: {
                switch (expected.get_element_type()) {
                TENSOR_COMPARE(i32, i32, 0, 0, size, 0)
                TENSOR_COMPARE(i64, i32, 0, 0, size, 0)
                default:
                    break;
                }
                break;
            }
            case ov::element::i64: {
                switch (expected.get_element_type()) {
                TENSOR_COMPARE(i32, i64, 0, 0, size, 0)
                TENSOR_COMPARE(i64, i64, 0, 0, size, 0)
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

void MulticlassNmsLayerTestGPU::SetUp() {
    const auto& [shapes, input_types, maxOutBoxesPerClass, inFloatVar, backgroundClass, keepTopK, outType, sortResultType, inboolVar, _targetDevice] =
        this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes(shapes);

    const auto& [paramsPrec, roisnumPrec] = input_types;

    const auto& [iouThr, scoreThr, nmsEta] = inFloatVar;

    const auto& [sortResCB, normalized] = inboolVar;

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(paramsPrec, inputDynamicShapes.at(0)),
                                std::make_shared<ov::op::v0::Parameter>(paramsPrec, inputDynamicShapes.at(1))};
    if (inputDynamicShapes.size() > 2)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(roisnumPrec, inputDynamicShapes.at(2)));

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

    std::shared_ptr<ov::Node> nms;
    if (!use_op_v8) {
        if (params.size() > 2) {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(params[0], params[1], params[2], m_attrs);
        } else {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(params[0], params[1], m_attrs);
        }
    } else {
        nms =  std::make_shared<ov::op::v8::MulticlassNms>(params[0], params[1], m_attrs);
    }

    function = std::make_shared<ov::Model>(nms, params, "MulticlassNMS");
}

void MulticlassNmsLayerTestGPU8::SetUp() {
    use_op_v8 = true;
    MulticlassNmsLayerTestGPU::SetUp();
}

namespace {
/* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
const std::vector<std::vector<ov::Shape>> shapes2Inputs = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

/* input format #2 with 3 inputs: bboxes C, M, 4, scores C, M, roisnum N */
const std::vector<std::vector<ov::Shape>> shapes3Inputs = {
    {{1, 10, 4}, {1, 10}, {1}},
    {{1, 10, 4}, {1, 10}, {10}},
    {{2, 100, 4}, {2, 100}, {1}},
    {{2, 100, 4}, {2, 100}, {10}}
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 1};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<ov::element::Type> outType = {ov::element::i32, ov::element::i64};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> sortResultType = {
    ov::op::util::MulticlassNmsBase::SortResultType::SCORE,
    ov::op::util::MulticlassNmsBase::SortResultType::CLASSID,
    ov::op::util::MulticlassNmsBase::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto params_v9_2Inputs = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes2Inputs)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(ov::test::utils::DEVICE_GPU));


INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v9_2inputs,
                         MulticlassNmsLayerTestGPU,
                         params_v9_2Inputs,
                         MulticlassNmsLayerTestGPU::getTestCaseName);

const auto params_v9_3Inputs = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes3Inputs)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v9_3inputs,
                         MulticlassNmsLayerTestGPU,
                         params_v9_3Inputs,
                         MulticlassNmsLayerTestGPU::getTestCaseName);

const auto params_v8 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes2Inputs)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(ov::test::utils::DEVICE_GPU));


INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_v8,
                         MulticlassNmsLayerTestGPU8,
                         params_v8,
                         MulticlassNmsLayerTestGPU8::getTestCaseName);

}  // namespace
}  // namespace GPULayerTestsDefinitions
} // namespace test
} // namespace ov
