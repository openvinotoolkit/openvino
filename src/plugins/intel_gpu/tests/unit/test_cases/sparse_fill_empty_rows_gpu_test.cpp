// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/sparse_fill_empty_rows.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr float REL_EPS = 2e-3f;
constexpr float ABS_EPS = 1e-5f;

namespace helpers {
// TODO: Move to common place.

// Converts float vector to another type vector.
template <typename T>
std::vector<T> ConverFloatVector(const std::vector<float>& vec) {
    std::vector<T> ret;
    ret.reserve(vec.size());
    for (const auto& val : vec) {
        ret.push_back(T(val));
    }
    return ret;
}

// Allocates tensor with given shape and data.
template <typename TDataType>
memory::ptr AllocateTensor(ov::PartialShape shape, const std::vector<TDataType>& data) {
    const layout lo = {shape, ov::element::from<TDataType>(), cldnn::format::bfyx};
    EXPECT_EQ(lo.get_linear_size(), data.size());
    memory::ptr tensor = get_test_engine().allocate_memory(lo);
    set_values<TDataType>(tensor, data);
    return tensor;
}

template <typename T>
void CompareTypedBuffers(const memory::ptr& output, const memory::ptr& expectedOutput, cldnn::stream& stream) {
    mem_lock<T> output_ptr(output, stream);
    mem_lock<T> wanted_output_ptr(expectedOutput, stream);

    ASSERT_EQ(output->get_layout(), expectedOutput->get_layout());
    ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
    for (size_t i = 0; i < output_ptr.size(); ++i)
        ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], REL_EPS, ABS_EPS)) << "at index " << i;
}

void CompareBuffers(const memory::ptr& output, const memory::ptr& expectedOutput, cldnn::stream& stream) {
    ASSERT_EQ(output->get_layout(), expectedOutput->get_layout());
    auto type = output->get_layout().data_type;

    switch (type) {
    case data_types::f32:
        helpers::CompareTypedBuffers<float>(output, expectedOutput, stream);
        break;
    case data_types::i64:
        helpers::CompareTypedBuffers<int64_t>(output, expectedOutput, stream);
        break;
    default:
        GTEST_FAIL() << "Unsupported data type: " << type;
        break;
    }
}

}  // namespace helpers

struct SparseFillEmptyRowsTestParams {
    std::vector<float> indicesData;
    std::vector<float> valuesData;
    std::vector<float> denseShapeData;
    std::vector<float> expectedIndicesOutput;
    std::vector<float> expectedValuesOutput;
    std::vector<float> expectedEmptyRowIndicatorOutput;
    std::string testcaseName;
};

class sparse_fill_empty_rows_test : public ::testing::TestWithParam<SparseFillEmptyRowsTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsTestParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "indicesDataSize=" << param.indicesData.size();
        result << "_valuesDataSize=" << param.valuesData.size();
        result << "_expectedIndicesOutputSize=" << param.expectedIndicesOutput.size();
        result << "_expectedValuesOutputSize=" << param.expectedValuesOutput.size();
        result << "_expectedEmptyRowIndicatorOutputSize=" << param.expectedEmptyRowIndicatorOutput.size();
        result << "_" << param.testcaseName;
        return result.str();
    }

    struct SparseFillEmptyRowsInferenceParams {
        bool center;
        bool normalized;
        memory::ptr values;
        memory::ptr denseShape;
        memory::ptr indices;
        memory::ptr defaultValue;
        memory::ptr expectedIndicesOutput;
        memory::ptr expectedValuesOutput;
        memory::ptr expectedEmptyRowIndicatorOutput;
    };

    template <ov::element::Type_t ET>
    SparseFillEmptyRowsInferenceParams PrepareInferenceParams(const SparseFillEmptyRowsTestParams& testParam) {
        using T = typename ov::element_type_traits<ET>::value_type;
        SparseFillEmptyRowsInferenceParams ret;
        const auto indicesRows = testParam.indicesData.size() / 2;

        ret.indices = helpers::AllocateTensor<int64_t>(
            {static_cast<ov::Dimension::value_type>(indicesRows), 2},
            helpers::ConverFloatVector<int64_t>(testParam.indicesData));
        ret.values = helpers::AllocateTensor<T>(
            ov::PartialShape({static_cast<ov::Dimension::value_type>(testParam.valuesData.size())}),
            helpers::ConverFloatVector<T>(testParam.valuesData));
        ret.denseShape = helpers::AllocateTensor<int64_t>(
            {2}, helpers::ConverFloatVector<int64_t>(testParam.denseShapeData));
        ret.defaultValue = helpers::AllocateTensor<T>({}, {42.0f});

        ret.expectedIndicesOutput = helpers::AllocateTensor<int64_t>(
            {static_cast<ov::Dimension::value_type>(testParam.expectedIndicesOutput.size() / 2), 2},
            helpers::ConverFloatVector<int64_t>(testParam.expectedIndicesOutput));
        ret.expectedValuesOutput = helpers::AllocateTensor<T>(
            ov::PartialShape({static_cast<ov::Dimension::value_type>(testParam.expectedValuesOutput.size())}),
            helpers::ConverFloatVector<T>(testParam.expectedValuesOutput));
        ret.expectedEmptyRowIndicatorOutput = helpers::AllocateTensor<int64_t>(
            ov::PartialShape({static_cast<ov::Dimension::value_type>(testParam.expectedEmptyRowIndicatorOutput.size())}),
            helpers::ConverFloatVector<int64_t>(testParam.expectedEmptyRowIndicatorOutput));

        return ret;
    }

    void Execute(const SparseFillEmptyRowsInferenceParams& params) {
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        topology topology;
        topology.add(input_layout("indices", params.indices->get_layout()));
        topology.add(input_layout("values", params.values->get_layout()));
        topology.add(input_layout("denseShape", params.denseShape->get_layout()));
        topology.add(input_layout("default_value", params.defaultValue->get_layout()));
        std::vector<input_info> inputs = {
            input_info("values"),
            input_info("denseShape"),
            input_info("indices"),
            input_info("default_value"),
        };
        topology.add(sparse_fill_empty_rows("sparse_fill_empty_rows", inputs));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("indices", params.indices);
        network->set_input_data("values", params.values);
        network->set_input_data("denseShape", params.denseShape);
        network->set_input_data("default_value", params.defaultValue);
        auto outputs = network->execute();
        auto output_indices = outputs.at("sparse_fill_empty_rows").get_memory(0);
        auto output_values = outputs.at("sparse_fill_empty_rows").get_memory(1);
        auto output_empty_row_indicator = outputs.at("sparse_fill_empty_rows").get_memory(2);

        // Debug: Print expected and actual output shapes
        std::cout << "Expected Indices Output Shape: " << params.expectedIndicesOutput->get_layout().get_shape() << std::endl;
        std::cout << "Actual Indices Output Shape: " << output_indices->get_layout().get_shape() << std::endl;
        std::cout << "Expected Values Output Shape: " << params.expectedValuesOutput->get_layout().get_shape() << std::endl;
        std::cout << "Actual Values Output Shape: " << output_values->get_layout().get_shape() << std::endl;
        std::cout << "Expected Empty Row Indicator Output Shape: " << params.expectedEmptyRowIndicatorOutput->get_layout().get_shape() << std::endl;
        std::cout << "Actual Empty Row Indicator Output Shape: " << output_empty_row_indicator->get_layout().get_shape() << std::endl;

        helpers::CompareBuffers(output_indices, params.expectedIndicesOutput, get_test_stream());
        helpers::CompareBuffers(output_values, params.expectedValuesOutput, get_test_stream());
        helpers::CompareBuffers(output_empty_row_indicator, params.expectedEmptyRowIndicatorOutput, get_test_stream());
    }

private:
    engine& engine_ = get_test_engine();
};

std::vector<SparseFillEmptyRowsTestParams> generateTestParams() {
    std::vector<SparseFillEmptyRowsTestParams> params;
#define TEST_DATA(indicesData,                                          \
                  valuesData,                                           \
                  denseShapeData,                                       \
                  expectedIndicesOutput,                                \
                  expectedValuesOutput,                                 \
                  expectedEmptyRowIndicatorOutput,                      \
                  testcaseName)                                         \
    params.push_back(SparseFillEmptyRowsTestParams{indicesData,         \
                                     valuesData,                        \
                                     denseShapeData,                    \
                                     expectedIndicesOutput,             \
                                     expectedValuesOutput,              \
                                     expectedEmptyRowIndicatorOutput,   \
                                     testcaseName});

#include "unit_test_utils/tests_data/sparse_fill_empty_rows_data.h"
#undef TEST_DATA

    return params;
}

}  // namespace

#define SparseFillEmptyRows_TEST_P(precision)                                                      \
    TEST_P(sparse_fill_empty_rows_test, ref_comp_##precision) {                                       \
        Execute(PrepareInferenceParams<ov::element::Type_t::precision>(GetParam())); \
    }

SparseFillEmptyRows_TEST_P(f32);

INSTANTIATE_TEST_SUITE_P(sparse_fill_empty_rows_test_suite, sparse_fill_empty_rows_test, testing::ValuesIn(generateTestParams()), sparse_fill_empty_rows_test::getTestCaseName);
