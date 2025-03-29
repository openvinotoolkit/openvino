// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "low_precision/fake_quantize_decomposition.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/get_dequantization.hpp"
#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/network_helper.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class GetDequantizationTestValues {
public:
    FakeQuantizeOnData fakeQuantize;
    // actual dequantization to create ov::Model to run NetworkHelper::getDequantization
    DequantizationOperations actualDequantization;
    DequantizationOperations expectedDequantization;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
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

inline std::ostream& operator<<(std::ostream& out, const GetDequantizationTestValues& testValue) {
    return out << "_" << testValue.fakeQuantize << "_" << testValue.actualDequantization;
}

typedef std::tuple<
    ov::element::Type,
    ov::Shape,
    GetDequantizationTestValues> GetDequantizationParams;

class GetDequantizationTransformation : public LayerTransformation, public testing::WithParamInterface<GetDequantizationParams> {
public:
    void SetUp() override {
        const ov::element::Type precision = std::get<0>(GetParam());
        const ov::Shape shape = std::get<1>(GetParam());
        const GetDequantizationTestValues testValues = std::get<2>(GetParam());

        actualFunction = GetDequantizationFunction::get(
            precision,
            shape,
            testValues.fakeQuantize,
            testValues.actualDequantization);

        const auto output = actualFunction->get_output_op(0);
        auto dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(output);
    }

    static std::string getTestCaseName(testing::TestParamInfo<GetDequantizationParams> obj) {
        ov::element::Type precision;
        ov::Shape shape;
        GetDequantizationTestValues testValues;
        std::tie(precision, shape, testValues) = obj.param;

        std::ostringstream result;
        result << precision << "_" << shape << "_" << testValues;
        return result.str();
    }
};

TEST_P(GetDequantizationTransformation, CompareFunctions) {
    const GetDequantizationTestValues testValues = std::get<2>(GetParam());

    const auto output = actualFunction->get_output_op(0);
    const ov::pass::low_precision::FakeQuantizeDequantization dequantization = ov::pass::low_precision::NetworkHelper::getDequantization(output);
    DequantizationOperations actualDequantization = toDequantizationOperations(dequantization);
    actualDequantization.subtract.constantShapeIsDefined = testValues.expectedDequantization.subtract.constantShapeIsDefined;
    actualDequantization.subtract.outPrecision = testValues.expectedDequantization.subtract.outPrecision;
    actualDequantization.multiply.constantShapeIsDefined = testValues.expectedDequantization.multiply.constantShapeIsDefined;
    actualDequantization.multiply.outPrecision = testValues.expectedDequantization.multiply.outPrecision;
    ASSERT_TRUE(actualDequantization == testValues.expectedDequantization);
}

const element::TypeVector precisions = {
    ov::element::f32,
};

const std::vector<GetDequantizationTestValues> testValues = {
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ov::element::u8 },
        { ov::element::f32, {}, { 0.01f } },
        { ov::element::f32, {}, { 0.01f } }
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ov::element::u8 },
        { ov::element::f32, { 127.f }, { 0.01f } },
        { ov::element::f32, { 127.f }, { 0.01f } }
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ov::element::u8 },
        {
            ov::element::f32,
            {{ 127.f }, ov::element::f32, {}, false, 1, ov::element::u8, true},
            {{ 0.1f }, ov::element::f32, {}, false, 1},
        },
        {
            ov::element::f32,
            {{ 127.f }, ov::element::f32, {}, false, 1, ov::element::u8, true},
            {{ 0.1f }, ov::element::f32, {}, false, 1},
        }
    },
    {
        // unexpected Subtract shape
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ov::element::u8 },
        {
            ov::element::f32,
            {std::vector<float>(12ul, 127.0), ov::element::f32, {1, 3, 2, 2}, false, 0, ov::element::u8, true},
            {{ 0.1f }, ov::element::f32, {}, false, 1},
        },
        {
            {},
            {},
            {{ 0.1f }, ov::element::f32, {}, false, 1},
        }
    },
    {
        { 256ul, {}, { 0.f }, { 2.55f }, { 0.f }, { 2.55f }, ov::element::u8 },
        {
            ov::element::f32,
            {{ 127.f }, ov::element::f32, {}, false, 0, ov::element::u8, true},
            {{ 0.1f }, ov::element::f32, {}, false, 0},
        },
        {
            ov::element::f32,
            {{ 127.f }, ov::element::f32, {}, false, 0, ov::element::u8, true},
            {{ 0.1f }, ov::element::f32, {}, false, 0},
        }
    },
    {
        // unexpected precision (non dequantization operations)
        {},
        {
            ov::element::i32,
            DequantizationOperations::Subtract{ 32 }.setConstantPrecision(element::i32),
            DequantizationOperations::Multiply{ 2 }.setConstantPrecision(element::i32),
        },
        {}
    }
};

const std::vector<ov::Shape> shapes = {
    { 1, 1, 2, 2 },
    // TODO: 3D tensor
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    GetDequantizationTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    GetDequantizationTransformation::getTestCaseName);
