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
    SliceParams(const Tensor& data, const Tensor& start, const Tensor& stop, const Tensor& step, const Tensor& axes, const Tensor& output)
        : m_data(data),
          m_start(start),
          m_stop(stop),
          m_step(step),
          m_axes(axes),
          m_output(output)
          {};

    Tensor m_data;
    Tensor m_start;
    Tensor m_stop;
    Tensor m_step;
    Tensor m_axes;
    Tensor m_output;
};

class ReferenceSliceLayerTest : public testing::TestWithParam<SliceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_data, params.m_start, params.m_stop, params.m_step, params.m_axes);
        inputData = {params.m_data.data, params.m_start.data, params.m_stop.data, params.m_step.data, params.m_axes.data};
        refOutData = {params.m_output.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SliceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inShape=" << param.m_data.shape << "_";
        result << "oType=" << param.m_output.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Tensor& data, const Tensor& start, const Tensor& stop, const Tensor& step, const Tensor& axes) {
        const auto data_param = std::make_shared<op::Parameter>(data.type, data.shape);
        const auto start_param = std::make_shared<op::Parameter>(start.type, start.shape);
        const auto stop_param = std::make_shared<op::Parameter>(stop.type, stop.shape);
        const auto step_param = std::make_shared<op::Parameter>(step.type, step.shape);
        const auto axes_param = std::make_shared<op::Parameter>(axes.type, axes.shape);

        const auto slice = std::make_shared<op::v8::Slice>(data_param, start_param, stop_param, step_param, axes_param);
        return std::make_shared<Function>(NodeVector{slice}, ParameterVector{data_param, start_param, stop_param, step_param, axes_param});
    }
};

TEST_P(ReferenceSliceLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SliceParams> generateSliceParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<SliceParams> opParams {SliceParams(
       Tensor{{2, 2}, type, std::vector<T>{1, 2, 3, 4}},
       Tensor{{2}, type, std::vector<T>{0, 0}},
       Tensor{{2}, type, std::vector<T>{2, 2}},
       Tensor{{2}, type, std::vector<T>{1, 1}},
       Tensor{{2}, type, std::vector<T>{0, 1}},
       Tensor{{2, 2}, type, std::vector<T>{1, 2, 3, 4}}),
       SliceParams(
       Tensor{{2, 2, 2}, type, std::vector<T>{1, 2, 3, 4, 4, 6, 7, 8}},
       Tensor{{2}, type, std::vector<T>{0, 0}},
       Tensor{{2}, type, std::vector<T>{2, 2}},
       Tensor{{2}, type, std::vector<T>{1, 1}},
       Tensor{{2}, type, std::vector<T>{0, 1}},
       Tensor{{2, 2, 2}, type, std::vector<T>{1, 2, 3, 4, 4, 6, 7, 8}})
    };
    return opParams;
}

std::vector<SliceParams> generateSliceCombinedParams() {
    const std::vector<std::vector<SliceParams>> opTypeParams {
        generateSliceParams<element::Type_t::i32>(element::i32),
    };
    std::vector<SliceParams> combinedParams;
    std::for_each(opTypeParams.begin(), opTypeParams.end(), [&](std::vector<SliceParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Slice_With_Hardcoded_Refs, ReferenceSliceLayerTest, ::testing::ValuesIn(generateSliceCombinedParams()),
                         ReferenceSliceLayerTest::getTestCaseName);
}  // namespace
