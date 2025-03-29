// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <random>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/single_op/matrix_nms.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {

class MatrixNmsLayerTestGPU : virtual public MatrixNmsLayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;

private:
    void GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch);
    ov::op::v8::MatrixNms::Attributes m_attrs;

protected:
    void SetUp() override;
};

void MatrixNmsLayerTestGPU::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
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

void MatrixNmsLayerTestGPU::GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch) {
    size_t numBoxes = 0, numClasses = 0;
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        const auto& dims = inputs[funcInput.get_node_shared_ptr()].get_shape();

        if (i == 1) {
            numClasses = dims[1];
        } else {
            numBatches = dims[0];
            numBoxes = dims[1];
        }
    }

    ASSERT_TRUE(numBatches > 0 && numBoxes > 0 && numClasses > 0)
        << "Expected numBatches, numBoxes, numClasses > 0, got:" << numBatches << ", " << numBoxes << ", " << numClasses;

    auto realClasses = numClasses;
    if (m_attrs.background_class >= 0 && m_attrs.background_class < static_cast<int>(numClasses)) {
       realClasses--;
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

void MatrixNmsLayerTestGPU::compare(const std::vector<ov::Tensor> &expectedOutputs,
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
                ASSERT_TRUE(expected.get_size() != actual.get_size())
                    << "Expected and actual size 3rd output have different size";
            }

#define CASE(X, Y, _expected_offset, _actual_offset, _size, _threshold)                                              \
    case X:                                                                                                          \
        ov::test::utils::compare_raw_data(                                                                  \
            reinterpret_cast<const ov::fundamental_type_for<X>*>(expectedBuffer) + _expected_offset,                 \
            reinterpret_cast<const ov::fundamental_type_for<Y>*>(actualBuffer) + _actual_offset, _size, _threshold); \
        break;

            const auto& precision = actual.get_element_type();
            auto expected_offset = 0;
            auto actual_offset = 0;
            for (size_t i = 0; i < numPerBatch.size(); i++) {
                auto validNums = numPerBatch[i];
                switch (precision) {
                    case ov::element::f32: {
                        switch (expected.get_element_type()) {
                            CASE(ov::element::f32, ov::element::f32, expected_offset * 6, actual_offset * 6, validNums *6, 1e-5f)
                            CASE(ov::element::f64, ov::element::f32, expected_offset * 6, actual_offset * 6, validNums *6, 1e-5f)
                            default:
                                break;
                        }
                        const auto fBuffer = static_cast<float*>(actual.data());
                        for (size_t tailing = validNums * 6; tailing < maxOutputBoxesPerBatch * 6; tailing++) {
                            ASSERT_TRUE(std::abs(fBuffer[(actual_offset * 6 + tailing)] + 1.f) < 1e-5)
                                << "Invalid default value: " << fBuffer[i] << " at index: " << i;
                        }
                        break;
                    }
                    case ov::element::i32: {
                        switch (expected.get_element_type()) {
                            CASE(ov::element::i32, ov::element::i32, expected_offset, actual_offset, validNums, 0)
                            CASE(ov::element::i64, ov::element::i32, expected_offset, actual_offset, validNums, 0)
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
                            CASE(ov::element::i32, ov::element::i64, expected_offset, actual_offset, validNums, 0)
                            CASE(ov::element::i64, ov::element::i64, expected_offset, actual_offset, validNums, 0)
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
                        CASE(ov::element::i32, ov::element::i32, 0, 0, size, 0)
                        CASE(ov::element::i64, ov::element::i32, 0, 0, size, 0)
                        default:
                            break;
                    }
                    break;
                }
                case ov::element::i64: {
                    switch (expected.get_element_type()) {
                        CASE(ov::element::i32, ov::element::i64, 0, 0, size, 0)
                        CASE(ov::element::i64, ov::element::i64, 0, 0, size, 0)
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

void MatrixNmsLayerTestGPU::SetUp() {
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

TEST_P(MatrixNmsLayerTestGPU, CompareWithRefs) {
    run();
};

namespace {
const std::vector<std::vector<ov::Shape>> inStaticShapeParams = {{{3, 100, 4}, {3, 1, 100}},
                                                                 {{1, 10, 4}, {1, 100, 10}}};

const std::vector<ov::op::v8::MatrixNms::SortResultType> sortResultType = {ov::op::v8::MatrixNms::SortResultType::CLASSID,
                                                                           ov::op::v8::MatrixNms::SortResultType::SCORE,
                                                                           ov::op::v8::MatrixNms::SortResultType::NONE};
const std::vector<ov::element::Type> outType = {ov::element::i32, ov::element::i64};
const std::vector<TopKParams> topKParams = {TopKParams{-1, 5},
                                            TopKParams{100, -1}};

const std::vector<ThresholdParams> thresholdParams = {ThresholdParams{0.0f, 2.0f, 0.0f},
                                                      ThresholdParams{0.1f, 1.5f, 0.2f}};
const std::vector<int> backgroudClass = {-1, 1};
const std::vector<bool> normalized = {true, false};
const std::vector<ov::op::v8::MatrixNms::DecayFunction> decayFunction = {ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                                         ov::op::v8::MatrixNms::DecayFunction::LINEAR};

const auto nmsParamsStatic =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(sortResultType),
                       ::testing::ValuesIn(outType),
                       ::testing::ValuesIn(topKParams),
                       ::testing::ValuesIn(thresholdParams),
                       ::testing::ValuesIn(backgroudClass),
                       ::testing::ValuesIn(normalized),
                       ::testing::ValuesIn(decayFunction),
                       ::testing::Values(ov::test::utils::DEVICE_GPU));

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTestGPU_static,
                         MatrixNmsLayerTestGPU,
                         nmsParamsStatic,
                         MatrixNmsLayerTestGPU::getTestCaseName);

} // namespace
} // namespace test
} // namespace ov
