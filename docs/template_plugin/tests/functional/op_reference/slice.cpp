// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct SliceParams {
    SliceParams(const Tensor& data,
                const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                const Tensor& axes,
                const Tensor& output,
                const std::string& test_name = "")
        : m_data(data),
          m_start(start),
          m_stop(stop),
          m_step(step),
          m_axes(axes),
          m_output(output),
          m_test_name(test_name) {}

    Tensor m_data;
    Tensor m_start;
    Tensor m_stop;
    Tensor m_step;
    Tensor m_axes;
    Tensor m_output;
    std::string m_test_name;
};

class ReferenceSliceLayerTest : public testing::TestWithParam<SliceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_data, params.m_start, params.m_stop, params.m_step, params.m_axes);
        inputData = {params.m_data.data,
                     params.m_start.data,
                     params.m_stop.data,
                     params.m_step.data,
                     params.m_axes.data};
        refOutData = {params.m_output.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SliceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "test_name=" << param.m_test_name << "__";
        result << "data_shape=" << param.m_data.shape << "_";
        result << "data_type=" << param.m_data.type << "_";
        result << "axes_shape=" << param.m_axes.shape << "_";
        result << "axes_type=" << param.m_axes.type << "_";
        result << "ind_type=" << param.m_start.type << "_";
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Tensor& data,
                                                    const Tensor& start,
                                                    const Tensor& stop,
                                                    const Tensor& step,
                                                    const Tensor& axes) {
        const auto data_param = std::make_shared<op::Parameter>(data.type, data.shape);
        const auto start_param = std::make_shared<op::Parameter>(start.type, start.shape);
        const auto stop_param = std::make_shared<op::Parameter>(stop.type, stop.shape);
        const auto step_param = std::make_shared<op::Parameter>(step.type, step.shape);
        const auto axes_param = std::make_shared<op::Parameter>(axes.type, axes.shape);

        const auto slice = std::make_shared<op::v8::Slice>(data_param, start_param, stop_param, step_param, axes_param);
        return std::make_shared<Function>(NodeVector{slice},
                                          ParameterVector{data_param, start_param, stop_param, step_param, axes_param});
    }
};

TEST_P(ReferenceSliceLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t DATA_ET, element::Type_t IND_ET, element::Type_t AXIS_ET>
std::vector<SliceParams> generateSliceParams(const element::Type& data_type,
                                             const element::Type& ind_type,
                                             const element::Type& axis_type) {
    using DATA_T = typename element_type_traits<DATA_ET>::value_type;
    using IND_T = typename element_type_traits<IND_ET>::value_type;
    using AXIS_T = typename element_type_traits<AXIS_ET>::value_type;

    std::vector<SliceParams> opParams{
        SliceParams(Tensor{{2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{0, 0}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{2, 2}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{1, 1}},
                    Tensor{{2}, axis_type, std::vector<AXIS_T>{0, 1}},
                    Tensor{{2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4}},
                    "2D_full_axes"),
        SliceParams(Tensor{{2, 4}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{1, 0}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{2, 3}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{1, 2}},
                    Tensor{{2}, axis_type, std::vector<AXIS_T>{0, 1}},
                    Tensor{{1, 2}, data_type, std::vector<DATA_T>{5, 7}},
                    "2D_step_2"),
        SliceParams(Tensor{{2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{0, 3}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{2, -4}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{1, -1}},
                    Tensor{{2}, axis_type, std::vector<AXIS_T>{0, 1}},
                    Tensor{{2, 2}, data_type, std::vector<DATA_T>{2, 1, 4, 3}},
                    "2D_negative_step_mix"),
        SliceParams(Tensor{{2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{-1, 3}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{-4, -4}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{-1, -1}},
                    Tensor{{2}, axis_type, std::vector<AXIS_T>{0, 1}},
                    Tensor{{2, 2}, data_type, std::vector<DATA_T>{4, 3, 2, 1}},
                    "2D_negative_step_only"),
        SliceParams(Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{0, 0, 0}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{2, 2, 2}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{1, 1, 1}},
                    Tensor{{3}, axis_type, std::vector<AXIS_T>{0, 1, 2}},
                    Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_full_axes"),
        SliceParams(Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{0, 0}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{2, 2}},
                    Tensor{{2}, ind_type, std::vector<IND_T>{1, 1}},
                    Tensor{{2}, axis_type, std::vector<AXIS_T>{0, 1}},
                    Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_less_axes"),
        SliceParams(Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{0, 0, 0}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{2, 2, 2}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{1, 1, 1}},
                    Tensor{{3}, axis_type, std::vector<AXIS_T>{-3, -2, -1}},
                    Tensor{{2, 2, 2}, data_type, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_negative_axes"),
        SliceParams(Tensor{{2, 4, 3}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{0, 0, 4}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{2, 4, -5}},
                    Tensor{{3}, ind_type, std::vector<IND_T>{3, 2, -2}},
                    Tensor{{3}, axis_type, std::vector<AXIS_T>{0, 1, 2}},
                    Tensor{{1, 2, 2}, data_type, std::vector<DATA_T>{2, 0, 8, 6}},
                    "3D_mixed_step"),
        SliceParams(
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            Tensor{{4}, ind_type, std::vector<IND_T>{0, 0, 0, 0}},
            Tensor{{4}, ind_type, std::vector<IND_T>{4, 2, 3, 2}},
            Tensor{{4}, ind_type, std::vector<IND_T>{1, 1, 1, 1}},
            Tensor{{4}, axis_type, std::vector<AXIS_T>{0, 1, 2, 3}},
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            "4D_full_axes"),

        SliceParams(
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            Tensor{{4}, ind_type, std::vector<IND_T>{-INT32_MAX, INT32_MAX, INT32_MAX, -INT32_MAX}},
            Tensor{{4}, ind_type, std::vector<IND_T>{INT32_MAX, -INT32_MAX, -INT32_MAX, INT32_MAX}},
            Tensor{{4}, ind_type, std::vector<IND_T>{1, -1, -1, 1}},
            Tensor{{4}, axis_type, std::vector<AXIS_T>{0, 1, 2, 3}},
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{10, 11, 8,  9,  6,  7,  4,  5,  2,  3,  0,  1,
                                                                22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13,
                                                                34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25,
                                                                46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37}},
            "4D_INT_MAX_index"),
        SliceParams(
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            Tensor{{1}, ind_type, std::vector<IND_T>{0}},
            Tensor{{1}, ind_type, std::vector<IND_T>{2}},
            Tensor{{1}, ind_type, std::vector<IND_T>{1}},
            Tensor{{1}, axis_type, std::vector<AXIS_T>{0}},
            Tensor{{2, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
            "4D_half_dim"),
        SliceParams(
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            Tensor{{1}, ind_type, std::vector<IND_T>{0}},
            Tensor{{1}, ind_type, std::vector<IND_T>{4}},
            Tensor{{1}, ind_type, std::vector<IND_T>{2}},
            Tensor{{1}, axis_type, std::vector<AXIS_T>{0}},
            Tensor{{2, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}},
            "4D_half_dim_step"),
        SliceParams(
            Tensor{{4, 2, 3, 2}, data_type, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
            Tensor{{2}, ind_type, std::vector<IND_T>{100, -100}},
            Tensor{{2}, ind_type, std::vector<IND_T>{-100, 100}},
            Tensor{{2}, ind_type, std::vector<IND_T>{-1, 2}},
            Tensor{{2}, axis_type, std::vector<AXIS_T>{2, 0}},
            Tensor{{2, 2, 3, 2}, data_type, std::vector<DATA_T>{4,  5,  2,  3,  0,  1,  10, 11, 8,  9,  6,  7,
                                                                28, 29, 26, 27, 24, 25, 34, 35, 32, 33, 30, 31}},
            "4D_mixed"),
        SliceParams(
            Tensor{{2, 4, 2, 2, 3},
                   data_type,
                   std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                       20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                       40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                       60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                       80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95}},
            Tensor{{5}, ind_type, std::vector<IND_T>{0, 1, -5, 100, 1}},
            Tensor{{5}, ind_type, std::vector<IND_T>{2, 6, 3, -100, 2}},
            Tensor{{5}, ind_type, std::vector<IND_T>{1, 2, 2, -1, 1}},
            Tensor{{5}, axis_type, std::vector<AXIS_T>{-5, 1, 4, 2, 3}},
            Tensor{{2, 2, 2, 1, 2},
                   data_type,
                   std::vector<DATA_T>{21, 23, 15, 17, 45, 47, 39, 41, 69, 71, 63, 65, 93, 95, 87, 89}},
            "5D_mixed"),
    };
    return opParams;
}

std::vector<SliceParams> generateSliceCombinedParams() {
    const std::vector<std::vector<SliceParams>> opTypeParams{
        generateSliceParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::i8>(element::f16,
                                                                                             element::i32,
                                                                                             element::i8),
        generateSliceParams<element::Type_t::f32, element::Type_t::i64, element::Type_t::i32>(element::f32,
                                                                                              element::i64,
                                                                                              element::i32),
        generateSliceParams<element::Type_t::i32, element::Type_t::i64, element::Type_t::i32>(element::i32,
                                                                                              element::i64,
                                                                                              element::i32),
        generateSliceParams<element::Type_t::i64, element::Type_t::i32, element::Type_t::i16>(element::i64,
                                                                                              element::i32,
                                                                                              element::i16),
    };
    std::vector<SliceParams> combinedParams;
    std::for_each(opTypeParams.begin(), opTypeParams.end(), [&](std::vector<SliceParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Slice_With_Hardcoded_Refs,
                         ReferenceSliceLayerTest,
                         ::testing::ValuesIn(generateSliceCombinedParams()),
                         ReferenceSliceLayerTest::getTestCaseName);
}  // namespace
