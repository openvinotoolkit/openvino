// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/search_sorted.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr float EPS = 2e-3f;

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

// Allocates tensoer with given shape and data.
template <typename TDataType>
memory::ptr AllocateTensor(ov::PartialShape shape, const std::vector<TDataType>& data) {
    const layout lo = {shape, ov::element::from<TDataType>(), cldnn::format::bfyx};
    EXPECT_EQ(lo.get_linear_size(), data.size());
    memory::ptr tensor = get_test_engine().allocate_memory(lo);
    set_values<TDataType>(tensor, data);
    return tensor;
}
}  // namespace helpers

struct SearchSortedTestParams {
    ov::PartialShape sortedShape;
    ov::PartialShape valuesShape;
    bool rightMode;
    std::vector<float> sortedData;
    std::vector<float> valuesData;
    std::vector<int64_t> expectedOutput;
    std::string testcaseName;
};

class search_sorted_test : public ::testing::TestWithParam<SearchSortedTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SearchSortedTestParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "sortedShape=" << param.sortedShape;
        result << "_valuesShape=" << param.valuesShape;
        result << "_rightMode=" << param.rightMode;
        result << "_" << param.testcaseName;
        return result.str();
    }

    struct SearchSortedInferenceParams {
        bool rightMode;
        memory::ptr sorted;
        memory::ptr values;
        memory::ptr expectedOutput;
    };

    template <ov::element::Type_t ET>
    SearchSortedInferenceParams PrepareInferenceParams(const SearchSortedTestParams& testParam) {
        using T = typename ov::element_type_traits<ET>::value_type;
        SearchSortedInferenceParams ret;

        ret.rightMode = testParam.rightMode;

        ret.sorted =
            helpers::AllocateTensor<T>(testParam.sortedShape, helpers::ConverFloatVector<T>(testParam.sortedData));
        ret.values =
            helpers::AllocateTensor<T>(testParam.valuesShape, helpers::ConverFloatVector<T>(testParam.valuesData));
        ret.expectedOutput = helpers::AllocateTensor<int64_t>(testParam.valuesShape, testParam.expectedOutput);

        return ret;
    }

    void Execute(const SearchSortedInferenceParams& params) {
        // Prepare the network.
        auto stream = get_test_stream_ptr(get_test_default_config(engine_));

        topology topology;
        topology.add(input_layout("sorted", params.sorted->get_layout()));
        topology.add(input_layout("values", params.values->get_layout()));
        topology.add(search_sorted("search_sorted", input_info("sorted"), input_info("values"), params.rightMode));

        cldnn::network::ptr network = get_network(engine_, topology, get_test_default_config(engine_), stream, false);

        network->set_input_data("sorted", params.sorted);
        network->set_input_data("values", params.values);

        // Run and check results.
        auto outputs = network->execute();

        auto output = outputs.at("search_sorted").get_memory();
        cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
        cldnn::mem_lock<int64_t> wanted_output_ptr(params.expectedOutput, get_test_stream());

        ASSERT_EQ(output->get_layout(), params.expectedOutput->get_layout());
        ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], EPS));
    }

private:
    engine& engine_ = get_test_engine();
};

std::vector<SearchSortedTestParams> generateTestParams() {
    std::vector<SearchSortedTestParams> params;
#define TEST_DATA(sorted_shape, values_shape, right_mode, sorted_data, values_data, expected_output_data, description) \
    params.push_back(SearchSortedTestParams{sorted_shape,                                                              \
                                            values_shape,                                                              \
                                            right_mode,                                                                \
                                            sorted_data,                                                               \
                                            values_data,                                                               \
                                            expected_output_data,                                                      \
                                            description});

#include "unit_test_utils/tests_data/search_sorted_data.h"
#undef TEST_DATA
    return params;
}

}  // namespace

#define SEARCH_SORTED_TEST_P(precision)                                              \
    TEST_P(search_sorted_test, ref_comp_##precision) {                               \
        Execute(PrepareInferenceParams<ov::element::Type_t::precision>(GetParam())); \
    }

SEARCH_SORTED_TEST_P(f16);
SEARCH_SORTED_TEST_P(u8);

INSTANTIATE_TEST_SUITE_P(search_sorted_test_suit,
                         search_sorted_test,
                         testing::ValuesIn(generateTestParams()),
                         search_sorted_test::getTestCaseName);
