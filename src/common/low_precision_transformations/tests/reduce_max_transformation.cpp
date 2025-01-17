// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <utility>
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"

#include "low_precision/reduce_max.hpp"
#include "ov_lpt_models/reduce.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/constant.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class ReduceMaxTransformation : public ReduceTransformation<ov::op::v1::ReduceMax> {
    void SetUp() override {
        ReduceTransformation::SetUp();
        const auto transformationParams = std::get<1>(GetParam()).params;

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::ReduceMaxTransformation, ov::op::v1::ReduceMax>(transformationParams);
        transform.transform(actualFunction);
    }
};

TEST_P(ReduceMaxTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    {1, 3, 16, 16},
    {4, 3, 16, 16},
    {-1, -1, -1, -1}
};

const std::vector<ReduceTransformationTestValues> reduceMaxTransformationTestValues = {
    // U8: keep dims, per-channel quantization, reduction by batch
    {
        LayerTransformation::createParamsU8I8(),
        {0},
        true,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: don't keep dims, per-channel quantization with negative values, reduction by spatial dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        false,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, -1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, -1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}},
            ov::element::f32,
            {}
        }
    },
    // U8: keep dims, per-channel quantization with subtract, reduction by batch
    {
        LayerTransformation::createParamsU8I8(),
        {0},
        true,
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{64.f, 128.f, 32.f}, ov::element::f32, {1, 3, 1, 1}},
                {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}
            }
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {
                {ov::element::f32},
                {{64.f, 128.f, 32.f}, ov::element::f32, {1, 3, 1, 1}},
                {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}
            }
        }
    },
    // U8: don't keep dims, per-channel quantization, reduction by channel
    {
        LayerTransformation::createParamsU8I8(),
        {1},
        false,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}},
            ov::element::f32,
            {}
        }
    },
    // U8: don't keep dims, per-tensor quantization, reduction by channel (reduction constant with negative values)
    {
        LayerTransformation::createParamsU8I8(),
        {-2},
        false,
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
    // U8: keep dims, per-channel quantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        true,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        }
    },
    // U8: don't keep dims, per-channel quantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsU8I8(),
        {2, 3},
        false,
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::u8,
            {},
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3}}}
        }
    },
    // I8: keep dims, per-channel quantization, reduction by batch
    {
        LayerTransformation::createParamsI8I8(),
        {0},
        true,
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        }
    },
    // I8: don't keep dims, per-channel quantization with negative values, reduction by spatial dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        false,
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, -1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, -1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}},
            ov::element::f32,
            {}
        }
    },
    // I8: don't keep dims, per-channel quantization, reduction by channel
    {
        LayerTransformation::createParamsI8I8(),
        {1},
        false,
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}},
            ov::element::f32,
            {}
        }
    },
    // I8: don't keep dims, per-tensor quantization, reduction by channel (reduction constant with negative values)
    {
        LayerTransformation::createParamsI8I8(),
        {-2},
        false,
        {
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {0.1f}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {64.f}, {0.1f}}
        }
    },
    // I8: don't keep dims, per-channel quantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        false,
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3}}}
        }
    },
    // I8: keep dims, per-channel quantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        true,
        {
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::i8,
            {},
            ov::element::i8,
            {{ov::element::f32}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        }
    },
    // not update precisions, keep dims, per-channel quantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsI8I8().setUpdatePrecisions(false),
        {2, 3},
        true,
        {
            ov::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        },
        {
            ov::element::f32,
            {},
            ov::element::f32,
            {{}, {}, {{0.1f, 1.f, 10.f}, ov::element::f32, {1, 3, 1, 1}}}
        }
    },
    // I8: keep dims, no dequantization, reduction by spatial dimensions
    {
        LayerTransformation::createParamsI8I8(),
        {2, 3},
        true,
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
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReduceMaxTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(reduceMaxTransformationTestValues)),
    ReduceMaxTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {
    PartialShape::dynamic()
};

const std::vector<ReduceTransformationTestValues> reduceMaxTransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {-2},
        false,
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.1f}},
            ov::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReduceMaxTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(reduceMaxTransformationTestValues)),
    ReduceMaxTransformation::getTestCaseName);
} // namespace testValues2
} // namespace
