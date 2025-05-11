// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/mvn.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/mvn.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class MVNTransformationTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ov::AxisSet reductionAxes;
    bool normalizeVariance;
    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    MVNTransformationTestValues,
    int
> MVNTransformationParams;

class MVNTransformation : public LayerTransformation, public testing::WithParamInterface<MVNTransformationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::PartialShape inputShape = std::get<1>(GetParam());
        const MVNTransformationTestValues testValues = std::get<2>(GetParam());
        const int opset_version = std::get<3>(GetParam());

        actualFunction = ov::builder::subgraph::MVNFunction::getOriginal(
            precision,
            inputShape,
            testValues.reductionAxes,
            testValues.normalizeVariance,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            opset_version);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ov::pass::low_precision::MVNTransformation, ov::op::v0::Interpolate>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ov::builder::subgraph::MVNFunction::getReference(
            precision,
            inputShape,
            testValues.reductionAxes,
            testValues.normalizeVariance,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            opset_version);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MVNTransformationParams> obj) {
        const ov::element::Type precision = std::get<0>(obj.param);
        const ov::PartialShape inputShape = std::get<1>(obj.param);
        const MVNTransformationTestValues testValues = std::get<2>(obj.param);
        const int opset_version = std::get<3>(obj.param);

        std::ostringstream result;
        result <<
            precision << "_" <<
            toString(testValues.params) << "_" <<
            inputShape << "_" <<
            testValues.reductionAxes << "_" <<
            testValues.normalizeVariance << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            opset_version;
        return result.str();
    }
};


TEST_P(MVNTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> precisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<int> opset_version = {
    2, 6
};

namespace testValues1 {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 4, 16, 16 },
    { -1, -1, -1, -1 },
};

const std::vector<MVNTransformationTestValues> testValues = {
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::f16,
            {{ov::element::f16}, {}, {{0.45f}, ov::element::f16, {}, false, 1ul, ov::element::f16}}
        },
        {
            ov::element::f16,
            { },
            ov::element::f32,
            {{}, {}, {1.f}},
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::u8,
            {{ov::element::f32}, {-0.32f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {-0.32f}, {0.45f}},
            ov::element::f32,
            { }
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.45f}}
        },
        {
            ov::element::u8,
            { },
            ov::element::f32,
            {{}, {}, {1.f}}
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ov::element::u8,
            {{ov::element::f32}, {127.f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {127.f}, {0.45f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ov::element::u8,
            {{ov::element::f32}, {12.5f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {12.5f}, {0.45f}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::u8,
            {{ov::element::f32}, {127.f}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {127.f}, {0.45f}},
            ov::element::f32,
            {}
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {-0.5f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {-1.f}}
        }
    },
    {
        {1, 2, 3},
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.45f}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {0.45f}}
        }
    },
    {
        {1, 2, 3},
        false,
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            ov::element::f32,
            {{}, {}, {0.45f}}
        },
        {
            ov::element::f32,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {0.45f}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MVNTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opset_version)),
    MVNTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ov::PartialShape> inputShapes = {
    { 1, 2, 2, 2 },
    { -1, -1, -1, -1}
};

const std::vector<MVNTransformationTestValues> testValues = {
    {
        {1, 2, 3},
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.45f, 0.45f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {{0.45f, 0.45f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}}
        }
    },
    {
        {2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.45f, -0.45f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ov::element::u8,
            {{}, {}, {}},
            ov::element::f32,
            {{}, {}, {{1.f, -1.f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}}
        }
    },
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.45f, -0.45f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.45f, -0.45f}, ov::element::f32, ov::Shape{ 1, 2, 1, 1 }}},
            ov::element::f32,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MVNTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opset_version)),
    MVNTransformation::getTestCaseName);
} // namespace testValues2


namespace testValues3 {
const std::vector<ov::PartialShape> inputShapesWithDynamicRank = {
    PartialShape::dynamic(),
};

const std::vector<MVNTransformationTestValues> testValues = {
    {
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.45f}}
        },
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.45f}},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MVNTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(inputShapesWithDynamicRank),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opset_version)),
    MVNTransformation::getTestCaseName);
} // namespace testValues3
} // namespace
