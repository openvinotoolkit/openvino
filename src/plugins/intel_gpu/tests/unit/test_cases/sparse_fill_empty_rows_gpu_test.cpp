// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
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
            ov::Shape{indicesRows, 2},
            helpers::ConverFloatVector<int64_t>(testParam.indicesData));
        ret.values = helpers::AllocateTensor<T>(
            ov::Shape{testParam.valuesData.size()},
            helpers::ConverFloatVector<T>(testParam.valuesData));
        ret.denseShape = helpers::AllocateTensor<int64_t>(
            ov::Shape{2}, helpers::ConverFloatVector<int64_t>(testParam.denseShapeData));
        ret.defaultValue = helpers::AllocateTensor<T>(ov::Shape{}, {42.0f});

        ret.expectedIndicesOutput = helpers::AllocateTensor<int64_t>(
            ov::Shape{testParam.expectedIndicesOutput.size() / 2, 2},
            helpers::ConverFloatVector<int64_t>(testParam.expectedIndicesOutput));
        ret.expectedValuesOutput = helpers::AllocateTensor<T>(
            ov::Shape{testParam.expectedValuesOutput.size()},
            helpers::ConverFloatVector<T>(testParam.expectedValuesOutput));
        ret.expectedEmptyRowIndicatorOutput = helpers::AllocateTensor<int64_t>(
            ov::Shape{testParam.expectedEmptyRowIndicatorOutput.size()},
            helpers::ConverFloatVector<int64_t>(testParam.expectedEmptyRowIndicatorOutput));

        return ret;
    }

    void Execute(const SparseFillEmptyRowsInferenceParams& params, const SparseFillEmptyRowsTestParams& testParams) {
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));
        topology topology;

        topology.add(input_layout("values", params.values->get_layout()));
        topology.add(input_layout("denseShape", params.denseShape->get_layout()));
        topology.add(input_layout("indices", params.indices->get_layout()));
        topology.add(input_layout("default_value", params.defaultValue->get_layout()));
        
        std::vector<input_info> inputs = {
            input_info("values"),
            input_info("denseShape"),
            input_info("indices"),
            input_info("default_value"),
        };
        
        topology.add(sparse_fill_empty_rows(
            "sparse_fill_empty_rows",
            inputs,
            testParams.valuesData,
            helpers::ConverFloatVector<int64_t>(testParams.denseShapeData),
            helpers::ConverFloatVector<int64_t>(testParams.indicesData),
            42.0f
        ));

        topology.add(reorder("output_indices", input_info("sparse_fill_empty_rows", 0), format::bfyx, data_types::i64));
        topology.add(reorder("output_values", input_info("sparse_fill_empty_rows", 1), format::bfyx, params.values->get_layout().data_type));
        topology.add(reorder("output_empty_row_indicator", input_info("sparse_fill_empty_rows", 2), format::bfyx, data_types::i64));

        ExecutionConfig config = get_test_default_config(engine_);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cldnn::network::ptr network = get_network(engine_, topology, config, stream, false);
        network->set_input_data("values", params.values);
        network->set_input_data("denseShape", params.denseShape);
        network->set_input_data("indices", params.indices);
        network->set_input_data("default_value", params.defaultValue);

        auto outputs = network->execute();
        auto output_indices = outputs.at("output_indices").get_memory();
        auto output_values = outputs.at("output_values").get_memory();
        auto output_empty_row_indicator = outputs.at("output_empty_row_indicator").get_memory();

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
    
    TEST_P(sparse_fill_empty_rows_test, ref_comp_f32) {
        const auto& testParams = GetParam();
        Execute(PrepareInferenceParams<ov::element::Type_t::f32>(testParams), testParams);
    }


SparseFillEmptyRows_TEST_P(f32);

INSTANTIATE_TEST_SUITE_P(sparse_fill_empty_rows_test_suite, sparse_fill_empty_rows_test, testing::ValuesIn(generateTestParams()), sparse_fill_empty_rows_test::getTestCaseName);
