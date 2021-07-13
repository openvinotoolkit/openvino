// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "low_precision/strided_slice.hpp"

#include "lpt_ngraph_functions/strided_slice_function.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;


inline std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class StridedSliceTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type inputPrecision;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type preicsionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    struct LayerParams {
        std::vector<int64_t> begin;
        std::vector<int64_t> end;
        std::vector<int64_t> strides;
        std::vector<int64_t> beginMask;
        std::vector<int64_t> endMask;
        std::vector<int64_t> newAxisMask;
        std::vector<int64_t> shrinkAxisMask;
        std::vector<int64_t> elipsisMask;
    };

    TestTransformationParams params;
    LayerParams layerParams;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    StridedSliceTransformationTestValues> StridedSliceTransformationParams;

class StridedSliceTransformation : public LayerTransformation, public testing::WithParamInterface<StridedSliceTransformationParams> {
public:
    void SetUp() override {
        const ngraph::PartialShape inputShape = std::get<0>(GetParam());
        const StridedSliceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::StridedSliceFunction::getOriginal(
            testValues.actual.inputPrecision,
            inputShape,
            testValues.actual.dequantization,
            testValues.layerParams.begin,
            testValues.layerParams.end,
            testValues.layerParams.strides,
            testValues.layerParams.beginMask,
            testValues.layerParams.endMask,
            testValues.layerParams.newAxisMask,
            testValues.layerParams.shrinkAxisMask,
            testValues.layerParams.elipsisMask);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::StridedSliceTransformation, ngraph::opset1::StridedSlice>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::StridedSliceFunction::getReference(
            testValues.expected.inputPrecision,
            inputShape,
            testValues.layerParams.begin,
            testValues.layerParams.end,
            testValues.layerParams.strides,
            testValues.layerParams.beginMask,
            testValues.layerParams.endMask,
            testValues.layerParams.newAxisMask,
            testValues.layerParams.shrinkAxisMask,
            testValues.layerParams.elipsisMask,
            testValues.expected.dequantizationBefore,
            testValues.expected.preicsionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<StridedSliceTransformationParams> obj) {
        const ngraph::PartialShape inputShape = std::get<0>(obj.param);
        const StridedSliceTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            inputShape << testValues.actual.inputPrecision << "_" << toString(testValues.params) <<
            testValues.actual.dequantization << "_strided_slice_params_" << testValues.layerParams.begin <<
            testValues.layerParams.end << testValues.layerParams.beginMask <<
            testValues.layerParams.endMask << testValues.layerParams.strides <<
            testValues.layerParams.shrinkAxisMask << testValues.layerParams.newAxisMask;
        return result.str();
    }
};

TEST_P(StridedSliceTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

StridedSliceTransformationTestValues::LayerParams channelSlice = {
    { 0, 0, 0, 0 }, // begin
    { 1, 2, 1, 1 }, // end
    { 1, 1, 1, 1 }, // strided
    { 1, 0, 1, 1 }, // beginMask
    { 1, 0, 1, 1 }, // endMask
    {}, // newAxisMask
    {}, // shrinkAxisMask
    {} // elipsisMask
};

StridedSliceTransformationTestValues::LayerParams spatialDimensionSlice = {
    { 0, 0, 0, 0 },
    { 1, 3, 20, 24 },
    { 1, 1, 1, 1 },
    { 1, 1, 0, 1 },
    { 1, 1, 0, 1 },
    {},
    {},
    {}
};

StridedSliceTransformationTestValues::LayerParams spatialDimensionEndSlice = {
    { 0, 0, 20, 0 },
    { 1, 3, 24, 24 },
    { 1, 1, 1, 1 },
    { 1, 1, 0, 1 },
    { 1, 1, 0, 1 },
    {},
    {},
    {}
};

StridedSliceTransformationTestValues::LayerParams sliceWithRemovedAxis = {
    { 0, 1, 0, 0 }, // begin
    { 1, 2, 1, 1 }, // end
    { 1, 1, 1, 1 }, // strided
    { 1, 0, 1, 1 }, // beginMask
    { 1, 0, 1, 1 }, // endMask
    { 0, 0, 0, 0 }, // newAxisMask
    { 0, 1, 0, 0 }, // shrinkAxisMask
    { 0, 0, 0, 0 } // elipsisMask
};

StridedSliceTransformationTestValues::LayerParams sliceWithAdditionalAxis = {
    { 0, 1, 0, 0 }, // begin
    { 1, 2, 1, 1 }, // end
    { 1, 1, 1, 1 }, // strided
    { 1, 0, 1, 1 }, // beginMask
    { 1, 0, 1, 1 }, // endMask
    { 0, 1, 0, 0 }, // newAxisMask
    { 0, 0, 0, 0 }, // shrinkAxisMask
    { 0, 0, 0, 0 } // elipsisMask
};

namespace testValues1 {
const std::vector<ngraph::PartialShape> inputShapes = {
    {1, 3, 24, 24},
    {Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()}
};

const std::vector<StridedSliceTransformationTestValues> stridedSliceTransformationTestValues = {
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.1f }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 128.f, 128.f }}, {{ 0.1f, 0.1f, 0.1f }}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f }}, {{ 0.1f, 0.01f }}}
        }
    },
    // U8: without subtract
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.1f }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { 0.1f }}
        }
    },
    // I8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 32.f }, { 0.1f }}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // I8: channel slice, per-channel quantization with the same values
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 32.f, 32.f }}, {{ 0.1f, 0.1f, 0.1f }}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // I8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f }}, {{ 0.1f, 0.01f }}}
        }
    },
    // channel slice, not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        channelSlice,
        {
            ngraph::element::f32,
            {{}, { 128.f }, { 0.1f }}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, { 128.f }, { 0.1f }}
        }
    },
    // channel slice, no dequantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {}
        }
    },
    // quantization after convolution
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ngraph::element::f32, {3, 1, 1} }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ngraph::element::f32, {1, 3, 1, 1} }}
        }
    },
    // quantization after convolution
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ngraph::element::f32, {3, 1, 1} }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, { {0.1f, 0.01f}, ngraph::element::f32, {1, 2, 1, 1} }}
        }
    },
    // U8: special dimension slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // U8: without convert
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ngraph::element::f32,
            {{}, { 128.f }, { 0.1f }}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {{}, { 128.f }, { 0.1f }}
        }
    },
    // I8: special dimension slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // I8: special dimension end slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionEndSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // I8: special dimension end slice, per-tensor quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionEndSlice,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 32.f }, { 0.1f }}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithRemovedAxis,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        }
    },
    // U8: channel slice, per-channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithRemovedAxis,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { {128.f, 64.f, 32.f} }, { {0.1f, 0.2f, 0.3f} }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {64.f}, {0.2f}},
        }
    },
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithAdditionalAxis,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {128.f}, {0.1f}}
        }
    },
    // U8: channel slice, per-channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithAdditionalAxis,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { {128.f, 64.f, 32.f} }, { {0.1f, 0.2f, 0.3f} }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {
                {ngraph::element::f32},
                { {128.f, 64.f, 32.f}, ngraph::element::f32, {1, 1, 3, 1, 1} },
                { {0.1f, 0.2f, 0.3f}, ngraph::element::f32, {1, 1, 3, 1, 1} }
            },
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    StridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(stridedSliceTransformationTestValues)),
    StridedSliceTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> inputShapes = {
    { Dimension::dynamic(), Dimension::dynamic(), 4, Dimension::dynamic() },
    { 1, 3, 4, 4 }
};

const std::vector<StridedSliceTransformationTestValues> testValuesWithDQBySpatialDimension = {
    // I8: channel slice, quantization by special dimension
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{32.f, 64.f, 32.f, 64.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 2.f, 1.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            }
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {
                {ngraph::element::f32},
                {{32.f, 64.f, 32.f, 64.f}, ngraph::element::f32, {1, 1, 4, 1}},
                {{3.f, 2.f, 1.f, 3.f}, ngraph::element::f32, {1, 1, 4, 1}}
            }
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    StridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValuesWithDQBySpatialDimension)),
    StridedSliceTransformation::getTestCaseName);
} // namespace testValues2

namespace testValues3 {
const std::vector<ngraph::PartialShape> inputShapesWithDynamicChannels = {
    {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
    PartialShape::dynamic()
};

const std::vector<StridedSliceTransformationTestValues> testValues = {
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.1f }}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}},
            ngraph::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    StridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    StridedSliceTransformation::getTestCaseName);
} // namespace testValues3
} // namespace
