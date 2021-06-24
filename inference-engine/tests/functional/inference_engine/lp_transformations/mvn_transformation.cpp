// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "low_precision/mvn.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/mvn_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class MVNTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::Shape inputShape;
    ngraph::AxisSet reductionAxes;
    bool normalizeVariance;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::element::Type,
    MVNTransformationTestValues,
    int
> MVNTransformationParams;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
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

class MVNTransformation : public LayerTransformation, public testing::WithParamInterface<MVNTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const MVNTransformationTestValues testValues = std::get<1>(GetParam());
        const int opset_version = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::MVNFunction::getOriginal(
            precision,
            testValues.inputShape,
            testValues.reductionAxes,
            testValues.normalizeVariance,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            opset_version);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::MVNTransformation, ngraph::opset1::Interpolate>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MVNFunction::getReference(
            precision,
            testValues.inputShape,
            testValues.reductionAxes,
            testValues.normalizeVariance,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            opset_version);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MVNTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const MVNTransformationTestValues testValues = std::get<1>(obj.param);
        const int opset_version = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            precision << "_" <<
            toString(testValues.params) << "_" <<
            testValues.inputShape << "_" <<
            testValues.reductionAxes << "_" <<
            testValues.normalizeVariance << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            opset_version;
        return result.str();
    }
};

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    ngraph::element::f16
};

const std::vector<int> opset_version = {
    2, 6
};

const std::vector<MVNTransformationTestValues> testValues = {
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.45f}},
            ngraph::element::f32,
            { }
        }
    },
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.45f}}
        },
        {
            ngraph::element::u8,
            { },
            ngraph::element::f32,
            {{}, {}, {1.f}}
        }
    },
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {127.f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {127.f}, {0.45f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {12.5f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {12.5f}, {0.45f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {127.f}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {127.f}, {0.45f}},
            ngraph::element::f32,
            {}
        }
    },

    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {-0.5f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {-1.f}}
        }
    },

    {
        ngraph::Shape{ 1, 4, 16, 16 },
        {1, 2, 3},
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.45f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {0.45f}}
        }
    },
    {
        ngraph::Shape{ 1, 2, 2, 2 },
        {1, 2, 3},
        false,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.45f, 0.45f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {{0.45f, 0.45f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}}
        }
    },
    {
        ngraph::Shape{ 1, 2, 2, 2 },
        {2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.45f, -0.45f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::f32,
            {{}, {}, {{1.f, -1.f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}}
        }
    },
    {
        ngraph::Shape{ 1, 2, 2, 2 },
        {1, 2, 3},
        true,
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.45f, -0.45f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.45f, -0.45f}, ngraph::element::f32, ngraph::Shape{ 1, 2, 1, 1 }}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
};

TEST_P(MVNTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MVNTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(opset_version)),
    MVNTransformation::getTestCaseName);
