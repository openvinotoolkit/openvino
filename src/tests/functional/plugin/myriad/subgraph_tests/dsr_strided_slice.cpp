// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

struct StridedSliceParams {
    DataShapeWithUpperBound inputShape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisMask;
};

using Parameters = std::tuple<
        StridedSliceParams,
        DataType,   // Net precision
        std::string // Device name
>;

class DSR_StridedSlice : public testing::WithParamInterface<Parameters>,
                         public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        StridedSliceParams params;
        DataType netPrecision;
        std::tie(params, netPrecision, targetDevice) = this->GetParam();

        const auto input = createInputSubgraphWithDSR(netPrecision, params.inputShape);
        const auto stridedSlice = ngraph::builder::makeStridedSlice(input, params.begin, params.end, params.strides, netPrecision, params.beginMask,
                                                    params.endMask, params.newAxisMask, params.shrinkAxisMask, params.ellipsisMask);

        return stridedSlice;
    }
};

TEST_P(DSR_StridedSlice, CompareWithReference) {
    Run();
}

std::vector<StridedSliceParams> testCases = {
    { { { 2, 3, 4, 5, 6 }, { 2, 3, 4, 5, 3 } },
      { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, -1 }, { 1, 1, 1, 1, 1 }, {1, 0, 1, 1, 1}, {1, 0, 1, 1, 1},  {},  {0, 0, 0, 0, 1},  {} },
    { { { 800, 4 }, { 1000, 4 } }, { 0, 0 }, { -1, 0 }, { 2, 1 }, { 1, 0 }, { 0, 1 },  {},  {},  {} },
    { { { 1, 12, 80 }, { 1, 12, 100 } }, { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { { 1, 7, 80 }, { 1, 12, 100 } }, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { { 1, 10, 70 }, { 1, 12, 100 } }, { 0, 4, 0 }, { 0, 9, 0 }, { 1, 2, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { { 1, 10, 60 }, { 1, 12, 100 } }, { 0, -8, 0 }, { 0, -6, 0 }, { 1, 2, 1 }, { 1, 0, 1 }, { 1, 0, 1 },  {},  {},  {} },
    { { { 1, 2, 2, 2 }, { 1, 3, 3, 3 } }, { 0, 0, 0, 0 }, { 1, -1, -1, -1 }, { 1, 2, 1, 1 }, {0, 0, 0, 0}, {1, 1, 1, 1},  {},  {},  {} },
    { { { 4000, 2 }, { 8232, 2 } }, { 0, 1 }, { -1, 2 }, { 1, 1 }, {0, 0 }, {1, 1 },  {},  {},  {} },
};

std::vector<DataType> precisions = {
        ngraph::element::f32,
        ngraph::element::i32
};

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice, DSR_StridedSlice,
                        ::testing::Combine(
                            ::testing::ValuesIn(testCases),
                            ::testing::ValuesIn(precisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
