// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <immintrin.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "util.hpp"

namespace {

#define ASSERT_NO_THROW_WITH_MESSAGE(code)   \
    do {                                     \
        try {                                \
            code;                            \
        } catch (const std::exception& ex) { \
            FAIL() << ex.what();             \
        } catch (...) {                      \
            FAIL() << "Unknown exception";   \
        }                                    \
    } while (0)

#define ASSERT_NO_THROW_IF(condition, code)     \
    do {                                        \
        if (condition) {                        \
            ASSERT_NO_THROW_WITH_MESSAGE(code); \
        } else {                                \
            ASSERT_ANY_THROW(code);             \
        }                                       \
    } while (0);

namespace details {

/// Read a 4-bit value from packed int4 array
uint8_t read_4b(const uint8_t* data, size_t r, size_t c, size_t cols) {
    size_t idx = r * cols + c;
    uint8_t byte = data[idx / 2];
    return (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
}

// Write a 4-bit value to packed int4 array
void write_4b(uint8_t* data, uint8_t val, size_t r, size_t c, size_t cols) {
    size_t idx = r * cols + c;
    // uint8_t* byte = data + idx;
    uint8_t& byte = data[idx / 2];
    if (idx % 2 == 0) {
        byte = (byte & 0xF0) | (val & 0x0F);
    } else {
        byte = (byte & 0x0F) | ((val & 0x0F) << 4);
    }
}

// Transpose for int4-packed data (2x int4 per uint8_t)
void transpose_i4(const uint8_t* input, uint8_t* output, size_t rows, size_t cols) {
    size_t total = rows * cols;
    // Zero output buffer
    std::fill(output, output + (total + 1) / 2, 0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            uint8_t val = read_4b(input, i, j, cols);
            write_4b(output, val, j, i, rows);
        }
    }
}

template <typename T>
void transpose(const T* input, T* output, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename T>
::testing::AssertionResult ArraysMatch(const T& actual, const T& expected) {
    if (actual.size() != expected.size()) {
        return ::testing::AssertionFailure()
               << "Size mismatch: actual.size()=" << actual.size() << ", expected.size()=" << expected.size();
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        using ElemType = typename T::value_type;
        ElemType a = actual[i];
        ElemType e = expected[i];

        // For float types, use a tolerance for comparison
        if constexpr (std::is_floating_point<ElemType>::value) {
            float tol = 1e-3f;
            if (std::fabs(static_cast<float>(a) - static_cast<float>(e)) > tol) {
                return ::testing::AssertionFailure()
                       << "Mismatch at index " << i << ": actual=" << a << ", expected=" << e;
            }
        } else {
            // For integer types, direct comparison
            if (a != e) {
                return ::testing::AssertionFailure() << "Mismatch at index " << i << ": actual=" << static_cast<int>(a)
                                                     << ", expected=" << static_cast<int>(e);
            }
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace details

using ShapesInitializer = std::function<void(std::vector<int>&)>;

using TransposeTestsParams = std::tuple<ov::element::Type_t,  // Precision
                                        ShapesInitializer     // input_shape
                                        >;

class TransposeTestsBase {
protected:
    ov::element::Type type;
    ov::Tensor inTensor;
    ov::Tensor outTensor;

    std::vector<int8_t> input;
    std::vector<int8_t> output;
    std::vector<int8_t> ref_output;
    ov::Shape input_shape;

    void make_input() {
        size_t nElements = shape_size(input_shape);

        ASSERT_EQ((type.bitwidth() * nElements) % 8, 0)
            << "Input len has to be byte boundary aligned, but was " << type.bitwidth() * nElements << " bits";
        const size_t nBytes = type.bitwidth() * nElements / 8;

        input.resize(nBytes);
        ref_output.resize(nBytes);
        output.resize(nBytes);

        std::fill(ref_output.begin(), ref_output.end(), 0);
        std::fill(output.begin(), output.end(), 0);

        std::array<int8_t, 32> input_local = {0x0A, 0x0B, 0x1C, 0x1D, 0x2E, 0x2F, 0x35, 0x36, 0x4A, 0x4B, 0x5A,
                                              0x5B, 0x6A, 0x6B, 0x7A, 0x7B, 0x0C, 0x0D, 0x1C, 0x1D, 0x2C, 0x2D,
                                              0x3C, 0x3D, 0x4C, 0x4D, 0x5C, 0x5D, 0x6C, 0x6D, 0x7C, 0x7D};

        for (size_t idx = 0, k = 0; k < nBytes; k++, idx = (idx + 1) % input_local.size()) {
            input[k] = input_local[idx];
        }

        inTensor = ov::Tensor(type, input_shape, input.data());
    }

public:
    void SetUp(const TransposeTestsParams& getParam) {
        ShapesInitializer shapeInit;

        std::tie(type, shapeInit) = getParam;

        std::vector<int> input;
        shapeInit(input);

        input_shape = ov::Shape{input.begin(), input.end()};

        make_input();

        make_ref_output();
    }
    std::string ToString() const {
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
    virtual bool isNegative() const {
        return false;
    }

    virtual void make_ref_output() {
        auto rows = input_shape[0] * input_shape[1];
        auto cols = input_shape[2];

        if (type == ov::element::i4) {
            details::transpose_i4(reinterpret_cast<const uint8_t*>(input.data()),
                                  reinterpret_cast<uint8_t*>(ref_output.data()),
                                  rows,
                                  cols);
        } else if (type == ov::element::f16) {
            details::transpose<uint16_t>(reinterpret_cast<const uint16_t*>(input.data()),
                                         reinterpret_cast<uint16_t*>(ref_output.data()),
                                         rows,
                                         cols);
        } else if (type == ov::element::f32) {
            details::transpose<float>(reinterpret_cast<const float*>(input.data()),
                                      reinterpret_cast<float*>(ref_output.data()),
                                      rows,
                                      cols);
        }
    }
};

template <class T>
class TransposeTestsTmpl : public ::testing::Test,
                           public T,
                           public ::testing::WithParamInterface<TransposeTestsParams> {
protected:
    void SetUp() override {
        T::SetUp(GetParam());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeTestsParams>& obj) {
        T _bt;
        _bt.SetUp(obj.param);
        return _bt.ToString();
    }
};

using TransposeTests = TransposeTestsTmpl<TransposeTestsBase>;
class TransposeTestsRef : public TransposeTests {};

#define Tensors [](std::vector<int> & input)

namespace details {
::testing::internal::ParamGenerator<typename std::vector<ShapesInitializer>::value_type> ShapesIn(
    const std::vector<ShapesInitializer>& container) {
    return ::testing::ValuesIn(container.begin(), container.end());
}

}  // namespace details
}  // anonymous namespace
