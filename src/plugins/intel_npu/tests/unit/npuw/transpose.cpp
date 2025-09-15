// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#    include "transpose.hpp"

namespace {

void TransposeTestsBase::make_input() {
    size_t nElements = shape_size(input_shape);

    ASSERT_EQ((type.bitwidth() * nElements) % 8, 0)
        << "Input len has to be byte boundary aligned, but was " << type.bitwidth() * nElements << " bits";
    const size_t nBytes = type.bitwidth() * nElements / 8;

    input.resize(nBytes);
    ref_output.resize(nBytes);
    output.resize(nBytes);

    std::fill(ref_output.begin(), ref_output.end(), 0);
    std::fill(output.begin(), output.end(), 0);

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 100);

    for (size_t i = 0; i < nBytes; ++i) {
        input[i] = static_cast<int8_t>(dist(rng));
    }

    inTensor = ov::Tensor(type, input_shape, input.data());
}

void TransposeTestsBase::SetUp(const TransposeTestsParams& getParam) {
    ShapesInitializer shapeInit;

    std::tie(type, shapeInit) = getParam;

    std::vector<int> input;
    shapeInit(input);

    input_shape = ov::Shape{input.begin(), input.end()};

    make_input();

    make_ref_output();
}

std::string TransposeTestsBase::ToString() const {
    std::ostringstream result;
    result << (isNegative() ? "NEGATIVE_" : "") << "[";

    for (size_t i = 0; i != input_shape.size(); i++) {
        result << input_shape[i] << ((i + 1 == input_shape.size()) ? "" : "x");
    }
    result << "]"
           << "_type_" << type;

    return result.str();
}

/**
 * Negative test cases has to be carefully reviewed, to still remain positive runs at some points
 * @return
 */
bool TransposeTestsBase::isNegative() const {
    return false;
}

void TransposeTestsBase::make_ref_output() {
    auto rows = input_shape[0] * input_shape[1];
    auto cols = input_shape[2];

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (type == ov::element::i4) {
                uint8_t val = details::read_4b(reinterpret_cast<const uint8_t*>(input.data()), i, j, cols);
                details::write_4b(reinterpret_cast<uint8_t*>(ref_output.data()), val, j, i, rows);
            } else if (type == ov::element::f16) {
                const uint16_t* input_ptr = reinterpret_cast<const uint16_t*>(input.data());
                uint16_t* output_ptr = reinterpret_cast<uint16_t*>(ref_output.data());
                output_ptr[j * rows + i] = input_ptr[i * cols + j];
            } else if (type == ov::element::f32) {
                const float* input_ptr = reinterpret_cast<const float*>(input.data());
                float* output_ptr = reinterpret_cast<float*>(ref_output.data());
                output_ptr[j * rows + i] = input_ptr[i * cols + j];
            }
        }
    }
}

TEST_P(TransposeTests, transpose) {
    ASSERT_NO_THROW_WITH_MESSAGE(outTensor = ov::npuw::util::transpose(inTensor));
    int8_t* dst = static_cast<int8_t*>(outTensor.data());
    output = std::vector<int8_t>(dst, dst + output.size());
    ASSERT_TRUE(details::ArraysMatch(output, ref_output));
}

const auto TestCases = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f32}),
        ::details::ShapesIn({Tensors{input={1, 10, 16};
}
, Tensors {
    input = {1, 10, 128};
}
, Tensors {
    input = {1, 16, 256};
}
, Tensors {
    input = {1, 16, 300};
}
})
);

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTests, TestCases, TransposeTests::getTestCaseName);

}  // anonymous namespace

#endif  // __AVX2__
