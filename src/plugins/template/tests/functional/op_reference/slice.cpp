// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

namespace reference_tests {
namespace {
struct SliceParams {
    SliceParams(const reference_tests::Tensor& data,
                const reference_tests::Tensor& start,
                const reference_tests::Tensor& stop,
                const reference_tests::Tensor& step,
                const reference_tests::Tensor& axes,
                const reference_tests::Tensor& output,
                const std::string& test_name = "")
        : m_data(data),
          m_start(start),
          m_stop(stop),
          m_step(step),
          m_axes(axes),
          m_output(output),
          m_test_name(test_name),
          m_default_axes(false) {}

    // Default `axes` input
    SliceParams(const reference_tests::Tensor& data,
                const reference_tests::Tensor& start,
                const reference_tests::Tensor& stop,
                const reference_tests::Tensor& step,
                const reference_tests::Tensor& output,
                const std::string& test_name = "")
        : m_data(data),
          m_start(start),
          m_stop(stop),
          m_step(step),
          m_output(output),
          m_test_name(test_name),
          m_default_axes(true) {}

    reference_tests::Tensor m_data;
    reference_tests::Tensor m_start;
    reference_tests::Tensor m_stop;
    reference_tests::Tensor m_step;
    reference_tests::Tensor m_axes;
    reference_tests::Tensor m_output;
    std::string m_test_name;
    bool m_default_axes = false;
};

class ReferenceSliceLayerTest : public testing::TestWithParam<SliceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.m_default_axes) {
            function = CreateFunction(params.m_data, params.m_start, params.m_stop, params.m_step);
            inputData = {params.m_data.data, params.m_start.data, params.m_stop.data, params.m_step.data};

        } else {
            function = CreateFunction(params.m_data, params.m_start, params.m_stop, params.m_step, params.m_axes);
            inputData = {params.m_data.data,
                         params.m_start.data,
                         params.m_stop.data,
                         params.m_step.data,
                         params.m_axes.data};
        }

        refOutData = {params.m_output.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SliceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "test_name=" << param.m_test_name << "__";
        result << "data_shape=" << param.m_data.shape << "_";
        result << "data_type=" << param.m_data.type << "_";
        result << "ind_type=" << param.m_start.type << "_";
        if (param.m_default_axes) {
            result << "axes_shape="
                   << "default"
                   << "_";
            result << "axes_type="
                   << "default"
                   << "_";
        } else {
            result << "axes_shape=" << param.m_axes.shape << "_";
            result << "axes_type=" << param.m_axes.type << "_";
        }
        result << param.m_test_name;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& data,
                                                 const reference_tests::Tensor& start,
                                                 const reference_tests::Tensor& stop,
                                                 const reference_tests::Tensor& step,
                                                 const reference_tests::Tensor& axes) {
        const auto data_param = std::make_shared<op::v0::Parameter>(data.type, data.shape);
        const auto start_param = std::make_shared<op::v0::Parameter>(start.type, start.shape);
        const auto stop_param = std::make_shared<op::v0::Parameter>(stop.type, stop.shape);
        const auto step_param = std::make_shared<op::v0::Parameter>(step.type, step.shape);
        const auto axes_param = std::make_shared<op::v0::Parameter>(axes.type, axes.shape);

        const auto slice = std::make_shared<op::v8::Slice>(data_param, start_param, stop_param, step_param, axes_param);
        return std::make_shared<Model>(NodeVector{slice},
                                       ParameterVector{data_param, start_param, stop_param, step_param, axes_param});
    }

    // Default `axes` input
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& data,
                                                 const reference_tests::Tensor& start,
                                                 const reference_tests::Tensor& stop,
                                                 const reference_tests::Tensor& step) {
        const auto data_param = std::make_shared<op::v0::Parameter>(data.type, data.shape);
        const auto start_param = std::make_shared<op::v0::Parameter>(start.type, start.shape);
        const auto stop_param = std::make_shared<op::v0::Parameter>(stop.type, stop.shape);
        const auto step_param = std::make_shared<op::v0::Parameter>(step.type, step.shape);

        const auto slice = std::make_shared<op::v8::Slice>(data_param, start_param, stop_param, step_param);
        return std::make_shared<Model>(NodeVector{slice},
                                       ParameterVector{data_param, start_param, stop_param, step_param});
    }
};

TEST_P(ReferenceSliceLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

}  // namespace

template <element::Type_t DATA_ET, element::Type_t IND_ET, element::Type_t AXIS_ET>
std::vector<SliceParams> generateSliceParamsUnsigned() {
    using DATA_T = typename element_type_traits<DATA_ET>::value_type;
    using IND_T = typename element_type_traits<IND_ET>::value_type;
    using AXIS_T = typename element_type_traits<AXIS_ET>::value_type;

    std::vector<SliceParams> test_params{
        SliceParams(Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{0}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{5}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{1}},
                    reference_tests::Tensor{{1}, AXIS_ET, std::vector<AXIS_T>{0}},
                    reference_tests::Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    "1D_full_axes"),
        SliceParams(Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{0}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{5}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{1}},
                    reference_tests::Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    "1D_default_axes"),
        SliceParams(Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{6}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{5}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{1}},
                    reference_tests::Tensor{{1}, AXIS_ET, std::vector<AXIS_T>{0}},
                    reference_tests::Tensor{{0}, DATA_ET, std::vector<DATA_T>{}},
                    "1D_empty_output"),
        SliceParams(Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{0, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, 2}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 1}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    "2D_full_axes"),
        SliceParams(Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{0, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, 2}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 1}},
                    reference_tests::Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    "2D_default_axes"),
        SliceParams(Tensor{{2, 4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, 3}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 2}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{1, 2}, DATA_ET, std::vector<DATA_T>{5, 7}},
                    "2D_step_2"),
        SliceParams(Tensor{{3, 16}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{0, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{3, 16}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 15}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{3, 2}, DATA_ET, std::vector<DATA_T>{0, 15, 16, 31, 32, 47}},
                    "2D_big_step"),
        SliceParams(Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{10, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{20, 2}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 1}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{0, 2}, DATA_ET, std::vector<DATA_T>{}},
                    "2D_empty_output"),
        SliceParams(Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{0, 0, 0}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{2, 2, 2}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{1, 1, 1}},
                    reference_tests::Tensor{{3}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2}},
                    reference_tests::Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_full_axes"),
        SliceParams(Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{0, 0, 0}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{2, 2, 2}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{1, 1, 1}},
                    reference_tests::Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_default_axes"),
        SliceParams(Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{0, 0}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, 2}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, 1}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_less_axes"),
        SliceParams(Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{0, 2, 0}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{2, 2, 2}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{1, 1, 1}},
                    reference_tests::Tensor{{3}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2}},
                    reference_tests::Tensor{{2, 0, 2}, DATA_ET, std::vector<DATA_T>{}},
                    "3D_empty_output"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{0, 0, 0, 0}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{4, 2, 3, 2}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{1, 1, 1, 1}},
                    reference_tests::Tensor{{4}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2, 3}},
                    reference_tests::Tensor{
                        {4, 2, 3, 2},
                        DATA_ET,
                        std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    "4D_full_axes"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{0, 0, 0, 0}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{4, 2, 3, 2}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{1, 1, 1, 1}},
                    reference_tests::Tensor{
                        {4, 2, 3, 2},
                        DATA_ET,
                        std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    "4D_default_axes"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{0}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{2}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{1}},
                    reference_tests::Tensor{{1}, AXIS_ET, std::vector<AXIS_T>{0}},
                    reference_tests::Tensor{{2, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                       8,  9,  10, 11, 12, 13, 14, 15,
                                                                                       16, 17, 18, 19, 20, 21, 22, 23}},
                    "4D_half_dim"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{0}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{2}},
                    reference_tests::Tensor{{1}, AXIS_ET, std::vector<AXIS_T>{0}},
                    reference_tests::Tensor{{2, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,
                                                                                       8,  9,  10, 11, 24, 25, 26, 27,
                                                                                       28, 29, 30, 31, 32, 33, 34, 35}},
                    "4D_half_dim_step"),
        SliceParams(
            reference_tests::Tensor{
                {2, 4, 2, 2, 3},
                DATA_ET,
                std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{0, 0, 0, 0, 0}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{2, 4, 2, 2, 3}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{1, 1, 1, 1, 1}},
            reference_tests::Tensor{{5}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2, 3, 4}},
            reference_tests::Tensor{
                {2, 4, 2, 2, 3},
                DATA_ET,
                std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95}},
            "5D_full_axes"),
    };
    return test_params;
}

template <element::Type_t DATA_ET, element::Type_t IND_ET, element::Type_t AXIS_ET>
std::vector<SliceParams> generateSliceParams() {
    using DATA_T = typename element_type_traits<DATA_ET>::value_type;
    using IND_T = typename element_type_traits<IND_ET>::value_type;
    using AXIS_T = typename element_type_traits<AXIS_ET>::value_type;

    std::vector<SliceParams> test_params{
        SliceParams(Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{5}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{-6}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{-1}},
                    reference_tests::Tensor{{4}, DATA_ET, std::vector<DATA_T>{4, 3, 2, 1}},
                    "1D_negative_step"),
        SliceParams(Tensor{{4}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{0}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{5}},
                    reference_tests::Tensor{{1}, IND_ET, std::vector<IND_T>{-1}},
                    reference_tests::Tensor{{1}, AXIS_ET, std::vector<AXIS_T>{0}},
                    reference_tests::Tensor{{0}, DATA_ET, std::vector<DATA_T>{}},
                    "1D_empty_output_negative_step"),
        SliceParams(Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{0, 3}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, -4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{1, -1}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{2, 1, 4, 3}},
                    "2D_negative_step_mix"),
        SliceParams(Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-1, 3}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-4, -4}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-1, -1}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{2, 2}, DATA_ET, std::vector<DATA_T>{4, 3, 2, 1}},
                    "2D_negative_step_only"),
        SliceParams(Tensor{{3, 16}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{2, 15}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-4, -17}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-1, -15}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{0, 1}},
                    reference_tests::Tensor{{3, 2}, DATA_ET, std::vector<DATA_T>{47, 32, 31, 16, 15, 0}},
                    "2D_negative_big_step"),
        SliceParams(Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{0, 0, 0}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{2, 2, 2}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{1, 1, 1}},
                    reference_tests::Tensor{{3}, AXIS_ET, std::vector<AXIS_T>{-3, -2, -1}},
                    reference_tests::Tensor{{2, 2, 2}, DATA_ET, std::vector<DATA_T>{1, 2, 3, 4, 5, 6, 7, 8}},
                    "3D_negative_axes"),
        SliceParams(Tensor{{2, 4, 3}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{0, 0, 4}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{2, 4, -5}},
                    reference_tests::Tensor{{3}, IND_ET, std::vector<IND_T>{3, 2, -2}},
                    reference_tests::Tensor{{3}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2}},
                    reference_tests::Tensor{{1, 2, 2}, DATA_ET, std::vector<DATA_T>{2, 0, 8, 6}},
                    "3D_mixed_step"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{4},
                                            IND_ET,
                                            std::vector<IND_T>{std::numeric_limits<IND_T>::min(),
                                                               std::numeric_limits<IND_T>::max(),
                                                               std::numeric_limits<IND_T>::max(),
                                                               std::numeric_limits<IND_T>::min()}},
                    reference_tests::Tensor{{4},
                                            IND_ET,
                                            std::vector<IND_T>{std::numeric_limits<IND_T>::max(),
                                                               std::numeric_limits<IND_T>::min(),
                                                               std::numeric_limits<IND_T>::min(),
                                                               std::numeric_limits<IND_T>::max()}},
                    reference_tests::Tensor{{4}, IND_ET, std::vector<IND_T>{1, -1, -1, 1}},
                    reference_tests::Tensor{{4}, AXIS_ET, std::vector<AXIS_T>{0, 1, 2, 3}},
                    reference_tests::Tensor{
                        {4, 2, 3, 2},
                        DATA_ET,
                        std::vector<DATA_T>{10, 11, 8,  9,  6,  7,  4,  5,  2,  3,  0,  1,  22, 23, 20, 21,
                                            18, 19, 16, 17, 14, 15, 12, 13, 34, 35, 32, 33, 30, 31, 28, 29,
                                            26, 27, 24, 25, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37}},
                    "4D_INT_MIN_MAX_index"),
        SliceParams(Tensor{{4, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{100, -100}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-100, 100}},
                    reference_tests::Tensor{{2}, IND_ET, std::vector<IND_T>{-1, 2}},
                    reference_tests::Tensor{{2}, AXIS_ET, std::vector<AXIS_T>{2, 0}},
                    reference_tests::Tensor{{2, 2, 3, 2}, DATA_ET, std::vector<DATA_T>{4,  5,  2,  3,  0,  1,  10, 11,
                                                                                       8,  9,  6,  7,  28, 29, 26, 27,
                                                                                       24, 25, 34, 35, 32, 33, 30, 31}},
                    "4D_mixed"),
        SliceParams(
            reference_tests::Tensor{
                {2, 4, 2, 2, 3},
                DATA_ET,
                std::vector<DATA_T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{0, 1, -5, 100, 1}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{2, 6, 3, -100, 2}},
            reference_tests::Tensor{{5}, IND_ET, std::vector<IND_T>{1, 2, 2, -1, 1}},
            reference_tests::Tensor{{5}, AXIS_ET, std::vector<AXIS_T>{-5, 1, 4, 2, 3}},
            reference_tests::Tensor{
                {2, 2, 2, 1, 2},
                DATA_ET,
                std::vector<DATA_T>{21, 23, 15, 17, 45, 47, 39, 41, 69, 71, 63, 65, 93, 95, 87, 89}},
            "5D_mixed"),
    };
    const auto& unsigned_test_params = generateSliceParamsUnsigned<DATA_ET, IND_ET, AXIS_ET>();
    test_params.insert(test_params.end(), unsigned_test_params.begin(), unsigned_test_params.end());
    return test_params;
}

std::vector<SliceParams> generateSliceCombinedParams() {
    const std::vector<std::vector<SliceParams>> opTypeParams{
        // Mixed types
        generateSliceParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::i8>(),
        generateSliceParams<element::Type_t::bf16, element::Type_t::i32, element::Type_t::i16>(),
        generateSliceParams<element::Type_t::f32, element::Type_t::i64, element::Type_t::i32>(),
        generateSliceParams<element::Type_t::i8, element::Type_t::i16, element::Type_t::i8>(),
        generateSliceParams<element::Type_t::i16, element::Type_t::i8, element::Type_t::i16>(),
        generateSliceParams<element::Type_t::i32, element::Type_t::i64, element::Type_t::i32>(),
        generateSliceParams<element::Type_t::i64, element::Type_t::i32, element::Type_t::i64>(),
        generateSliceParams<element::Type_t::u32, element::Type_t::i32, element::Type_t::i16>(),

        // Unsigned types
        generateSliceParamsUnsigned<element::Type_t::u8, element::Type_t::u8, element::Type_t::u8>(),
        generateSliceParamsUnsigned<element::Type_t::u16, element::Type_t::u16, element::Type_t::u16>(),
        generateSliceParamsUnsigned<element::Type_t::u32, element::Type_t::u32, element::Type_t::u32>(),
        generateSliceParamsUnsigned<element::Type_t::u64, element::Type_t::u64, element::Type_t::u64>(),
        generateSliceParamsUnsigned<element::Type_t::u16, element::Type_t::u8, element::Type_t::u32>(),
        generateSliceParamsUnsigned<element::Type_t::u32, element::Type_t::u16, element::Type_t::u8>(),
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

}  // namespace reference_tests
