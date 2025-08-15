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
#include "transpose.hpp"
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

template <typename T>
inline T read_value(const T* src, std::size_t r, std::size_t c, std::size_t cols) {
    std::size_t idx = r * cols + c;
    return src[idx];
}

template <typename T>
inline void write_value(T* dst, T value, std::size_t r, std::size_t c, std::size_t cols) {
    std::size_t idx = r * cols + c;
    dst[idx] = value;
}

// Permute for int4-packed data (2x int4 per uint8_t)
inline void permute_i4(const uint8_t* input, uint8_t* output, int rows, int cols) {
    int total = rows * cols;
    // Zero output buffer
    std::fill(output, output + (total + 1) / 2, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint8_t val = read_4b(input, i, j, cols);
            write_4b(output, val, j, i, rows);
        }
    }
}

template <typename T>
void permute(const T* input, T* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

}  // namespace details

using ShapesInitializer = std::function<void(std::vector<int>&)>;

using PermuteTestsParams = std::tuple<ov::element::Type_t,      // Precision
                                      ShapesInitializer,        // input_shape
                                      std::vector<std::size_t>  // axes
                                      >;

class PermuteTestsBase {
protected:
    ov::element::Type type;
    ov::Tensor inTensor;
    ov::Tensor outTensor;

    std::vector<std::size_t> axes;

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
    void SetUp(const PermuteTestsParams& getParam) {
        ShapesInitializer shapeInit;

        std::tie(type, shapeInit, axes) = getParam;

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

        result << "_axis";

        for (size_t i = 0; i < axes.size(); i++) {
            result << "_" << axes[i];
        }

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

        if (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) {
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
        } else if (axes[0] == 0 && axes[1] == 2 && axes[2] == 1) {
            for (std::size_t p = 0; p < input_shape[0]; p++) {
                for (std::size_t r = 0; r < input_shape[1]; r++) {
                    for (std::size_t c = 0; c < input_shape[2]; c++) {
                        if (type == ov::element::i4) {
                            details::write_4b(reinterpret_cast<uint8_t*>(ref_output.data()),
                                              details::read_4b(reinterpret_cast<const uint8_t*>(input.data()),
                                                               p * input_shape[1] + r,
                                                               c,
                                                               input_shape[2]),
                                              p * input_shape[2] + c,
                                              r,
                                              input_shape[1]);
                        } else if (type == ov::element::f32) {
                            details::write_value<float>(
                                reinterpret_cast<float*>(ref_output.data()),
                                details::read_value<float>(reinterpret_cast<const float*>(input.data()),
                                                           p * input_shape[1] + r,
                                                           c,
                                                           input_shape[2]),
                                p * input_shape[2] + c,
                                r,
                                input_shape[1]);
                        } else if (type == ov::element::f16) {
                            details::write_value<uint16_t>(
                                reinterpret_cast<uint16_t*>(ref_output.data()),
                                details::read_value<uint16_t>(reinterpret_cast<const uint16_t*>(input.data()),
                                                              p * input_shape[1] + r,
                                                              c,
                                                              input_shape[2]),
                                p * input_shape[2] + c,
                                r,
                                input_shape[1]);
                        }
                    }
                }
            }
        } else if (axes[0] == 1 && axes[1] == 0 && axes[2] == 2) {
            // Iterate over output tensor coordinates
            for (std::size_t p = 0; p < input_shape[1]; p++) {
                for (std::size_t r = 0; r < input_shape[0]; r++) {
                    for (std::size_t c = 0; c < input_shape[2]; c++) {
                        if (type == ov::element::i4) {
                            details::write_4b(reinterpret_cast<uint8_t*>(ref_output.data()),
                                              details::read_4b(reinterpret_cast<const uint8_t*>(input.data()),
                                                               r,
                                                               p * input_shape[2] + c,
                                                               input_shape[1] * input_shape[2]),
                                              p * input_shape[0] + r,
                                              c,
                                              input_shape[2]);
                        } else if (type == ov::element::f16) {
                            details::write_value<uint16_t>(
                                reinterpret_cast<uint16_t*>(ref_output.data()),
                                details::read_value<uint16_t>(reinterpret_cast<const uint16_t*>(input.data()),
                                                              r,
                                                              p * input_shape[2] + c,
                                                              input_shape[1] * input_shape[2]),
                                p * input_shape[0] + r,
                                c,
                                input_shape[2]);
                        }
                    }
                }
            }
        } else if (axes[0] == 1 && axes[1] == 2 && axes[2] == 0) {
            if (type == ov::element::f16) {
                details::transpose<uint16_t>(reinterpret_cast<const uint16_t*>(input.data()),
                                             reinterpret_cast<uint16_t*>(ref_output.data()),
                                             input_shape[0],
                                             input_shape[1] * input_shape[2]);
            } else if (type == ov::element::f32) {
                details::transpose<float>(reinterpret_cast<const float*>(input.data()),
                                          reinterpret_cast<float*>(ref_output.data()),
                                          input_shape[0],
                                          input_shape[1] * input_shape[2]);
            }
        }
    }
};

template <class T>
class PermuteTestsTmpl : public ::testing::Test, public T, public ::testing::WithParamInterface<PermuteTestsParams> {
protected:
    void SetUp() override {
        T::SetUp(GetParam());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<PermuteTestsParams>& obj) {
        T _bt;
        _bt.SetUp(obj.param);
        return _bt.ToString();
    }
};

using PermuteTests = PermuteTestsTmpl<PermuteTestsBase>;
class PermuteTestsRef : public PermuteTests {};

TEST_P(PermuteTests, permute) {
    ASSERT_NO_THROW_WITH_MESSAGE(outTensor = ov::npuw::util::permute(inTensor, axes));
    int8_t* dst = static_cast<int8_t*>(outTensor.data());
    output = std::vector<int8_t>(dst, dst + output.size());
    ASSERT_TRUE(details::ArraysMatch(output, ref_output));
}

#define Tensors [](std::vector<int> & input)

}  // anonymous namespace