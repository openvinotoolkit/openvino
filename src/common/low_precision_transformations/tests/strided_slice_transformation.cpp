// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "low_precision/strided_slice.hpp"

#include "ov_lpt_models/strided_slice.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class StridedSliceTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type inputPrecision;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type preicsionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
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
    ov::PartialShape,
    StridedSliceTransformationTestValues> StridedSliceTransformationParams;

class StridedSliceTransformation : public LayerTransformation, public testing::WithParamInterface<StridedSliceTransformationParams> {
public:
    void SetUp() override {
        const ov::PartialShape inputShape = std::get<0>(GetParam());
        const StridedSliceTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ov::builder::subgraph::StridedSliceFunction::getOriginal(
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
        transformer.add<ov::pass::low_precision::StridedSliceTransformation, ov::op::v1::StridedSlice>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::StridedSliceFunction::getReference(
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
        const ov::PartialShape inputShape = std::get<0>(obj.param);
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
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
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

namespace inputs_4d {

StridedSliceTransformationTestValues::LayerParams channelSlice2D = {
    {0, 0},  // begin
    {0, 2},  // end
    {1, 1},  // strided
    {1, 0},  // beginMask
    {1, 0},  // endMask
    {0, 0},  // newAxisMask
    {0, 0},  // shrinkAxisMask
    {0, 0}   // elipsisMask
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
    { 0, 0, 0, 0 }  // elipsisMask
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

const std::vector<ov::PartialShape> inputShapes = {
    {1, 3, 24, 24},
    {-1, -1, -1, -1}
};
const std::vector<StridedSliceTransformationTestValues> stridedSliceTransformationTestValues = {
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.1f }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with the same values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 128.f, 128.f }}, {{ 0.1f, 0.1f, 0.1f }}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with the same values, subtraction with Convert from u8 to fp32
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {1, 3, 1, 1}, false, 1ul, element::u8, true},
                {3.f}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{128.f}, element::dynamic, {}, false, 1ul, element::u8, true},
                {3.f}
            }
        }
    },
    // U8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f }}, {{ 0.1f, 0.01f }}}
        }
    },
    // U8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice2D,
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f }}, {{ 0.1f, 0.01f }}}
        }
    },
    // U8: without subtract
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.1f }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, { 0.1f }}
        }
    },
    // I8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, { 32.f }, { 0.1f }}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // I8: channel slice, per-channel quantization with the same values
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 32.f, 32.f }}, {{ 0.1f, 0.1f, 0.1f }}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // I8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        channelSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f }}, {{ 0.1f, 0.01f }}}
        }
    },
    // channel slice, not update precisions
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        channelSlice,
        {
            ov::element::f32,
            {{}, { 128.f }, { 0.1f }}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {{}, { 128.f }, { 0.1f }}
        }
    },
    // channel slice, no dequantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::f32,
            {}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {}
        }
    },
    // quantization after convolution
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ov::element::f32, {3, 1, 1} }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ov::element::f32, {1, 3, 1, 1} }}
        }
    },
    // quantization after convolution
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, { {0.1f, 0.01f, 1.f}, ov::element::f32, {3, 1, 1} }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, { {0.1f, 0.01f}, ov::element::f32, {1, 2, 1, 1} }}
        }
    },
    // U8: special dimension slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // U8: without convert
    {
        LayerTransformation::createParamsU8I8(),
        spatialDimensionSlice,
        {
            ov::element::f32,
            {{}, { 128.f }, { 0.1f }}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {{}, { 128.f }, { 0.1f }}
        }
    },
    // I8: special dimension slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // I8: special dimension end slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionEndSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {{ 32.f, 64.f, 32.f }}, {{ 0.1f, 0.01f, 1.f }}}
        }
    },
    // I8: special dimension end slice, per-tensor quantization with different values
    {
        LayerTransformation::createParamsI8I8(),
        spatialDimensionEndSlice,
        {
            ov::element::i8,
            {{ov::element::f32}, { 32.f }, { 0.1f }}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, { 32.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithRemovedAxis,
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}}
        }
    },
    // U8: channel slice, per-channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithRemovedAxis,
        {
            ov::element::u8,
            {{ov::element::f32}, { {128.f, 64.f, 32.f} }, { {0.1f, 0.2f, 0.3f} }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {64.f}, {0.2f}},
        }
    },
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithAdditionalAxis,
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}}
        }
    },
    // U8: channel slice, per-channel quantization
    {
        LayerTransformation::createParamsU8I8(),
        sliceWithAdditionalAxis,
        {
            ov::element::u8,
            {{ov::element::f32}, { {128.f, 64.f, 32.f} }, { {0.1f, 0.2f, 0.3f} }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                { {128.f, 64.f, 32.f}, ov::element::f32, {1, 1, 3, 1, 1} },
                { {0.1f, 0.2f, 0.3f}, ov::element::f32, {1, 1, 3, 1, 1} }
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
} // namespace inputs_4d

namespace inputs_4d_spatial {
const std::vector<ov::PartialShape> inputShapes = {
    { -1, -1, -1, -1 },
    { 1, 3, 4, 4 }
};

const std::vector<StridedSliceTransformationTestValues> testValuesWithDQBySpatialDimension = {
    // I8: channel slice, quantization by special dimension
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::i8,
            {
                {ov::element::f32},
                {{32.f, 64.f, 32.f, 64.f}, ov::element::f32, {1, 1, 4, 1}},
                {{3.f, 2.f, 1.f, 3.f}, ov::element::f32, {1, 1, 4, 1}}
            }
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {
                {ov::element::f32},
                {{32.f, 64.f, 32.f, 64.f}, ov::element::f32, {1, 1, 4, 1}},
                {{3.f, 2.f, 1.f, 3.f}, ov::element::f32, {1, 1, 4, 1}}
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
} // namespace inputs_4d_spatial

namespace dynamic_inputs {
const std::vector<ov::PartialShape> inputShapesWithDynamicChannels = {
    PartialShape::dynamic()
};

const std::vector<StridedSliceTransformationTestValues> testValues = {
    // U8: channel slice, per-tensor quantization
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.1f }}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, { 128.f }, { 0.1f }}
        }
    },
    // U8: channel slice, per-channel quantization with different values
    {
        LayerTransformation::createParamsU8I8(),
        channelSlice,
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {{ 128.f, 64.f, 128.f }}, {{ 0.1f, 0.01f, 1.f }}},
            ov::element::f32,
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
} // namespace dynamic_inputs

namespace inputs_3d {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 3, 4 },
    { 1, -1, 4 }
};

StridedSliceTransformationTestValues::LayerParams slice = {
    { 0, 1 }, // begin
    { 0, 2 }, // end
    { 1, 1 }, // strided
    { 1, 0 }, // beginMask
    { 1, 0 }, // endMask
    { 0, 0 }, // newAxisMask
    { 0, 1 }, // shrinkAxisMask
    { 0, 0 }  // elipsisMask
};

StridedSliceTransformationTestValues::LayerParams slice2 = {
    { 0, 1 }, // begin
    { 0, 2 }, // end
    { 1, 1 }, // strided
    { 1, 0 }, // beginMask
    { 1, 0 }, // endMask
    { 0, 0 }, // newAxisMask
    { 0, 1 }, // shrinkAxisMask
    { 0, 0 }  // elipsisMask
};

const std::vector<StridedSliceTransformationTestValues> testValuesWithDQBySpatialDimension = {
    // U8: channel slice, quantization by spatial dimension
    {
        LayerTransformation::createParamsU8I8(),
        slice,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 1, 4}},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 1, 4}}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4}},
                {{1.f, 2.f, 3.f, 4.f}, ov::element::f32, {1, 4}}
            }
        }
    },
    // U8: channel slice, quantization by spatial dimension
    {
        LayerTransformation::createParamsU8I8(),
        slice2,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1}},
                {{1.f, 2.f, 3.f}, ov::element::f32, {1, 3, 1}}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{2.f}, ov::element::f32, {}},
                {{2.f}, ov::element::f32, {}}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    StridedSliceTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValuesWithDQBySpatialDimension)),
    StridedSliceTransformation::getTestCaseName);
} // namespace inputs_3d

} // namespace
