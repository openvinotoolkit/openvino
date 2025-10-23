// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#    include "permute.hpp"

namespace details {

// Permute function.
template <typename T, bool Int4 = false>
void permute(const T* input, T* output, const std::vector<size_t>& dims, const std::vector<size_t>& order) {
    static_assert(!Int4 || (Int4 && std::is_same_v<T, uint8_t>), "For int4 data input data must be of type uint8_t");

    size_t ndim = dims.size();
    assert(order.size() == ndim);

    // Compute new dims.
    std::vector<size_t> new_dims = details::reorder(dims, order);

    // Compute strides.
    std::vector<size_t> old_strides = details::compute_strides(dims);
    std::vector<size_t> new_strides = details::compute_strides(new_dims);

    size_t total = std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());

    // For each index in output, compute corresponding input index.
    // Use a flat index and unravel it to multi-index in new order, then map to old order.
    for (size_t out_idx = 0; out_idx < total; ++out_idx) {
        // Unravel out_idx to multi-index in new_dims.
        std::vector<size_t> new_multi(ndim);
        size_t tmp = out_idx;
        for (size_t i = 0; i < ndim; ++i) {
            new_multi[i] = tmp / new_strides[i];
            tmp = tmp % new_strides[i];
        }
        // Map new_multi to old_multi.
        std::vector<size_t> old_multi(ndim);
        for (size_t i = 0; i < ndim; ++i)
            old_multi[order[i]] = new_multi[i];

        // Compute flat input index.
        size_t in_idx = 0;
        for (size_t i = 0; i < ndim; ++i)
            in_idx += old_multi[i] * old_strides[i];

        if constexpr (Int4) {
            uint8_t in_byte = input[in_idx / 2];
            uint8_t in_val = (in_idx % 2 == 0) ? (in_byte & 0x0F) : ((in_byte >> 4) & 0x0F);

            size_t out_byte_idx = out_idx / 2;
            if (out_idx % 2 == 0) {
                output[out_byte_idx] = (output[out_byte_idx] & 0xF0) | (in_val & 0x0F);
            } else {
                output[out_byte_idx] = (output[out_byte_idx] & 0x0F) | ((in_val & 0x0F) << 4);
            }
        } else {
            output[out_idx] = input[in_idx];
        }
    }
}

}  // namespace details

namespace {

void PermuteTestsBase::make_input() {
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

void PermuteTestsBase::SetUp(const PermuteTestsParams& getParam) {
    ShapesInitializer shapeInit;

    std::tie(type, shapeInit, axes) = getParam;

    std::vector<int> input;
    shapeInit(input);

    input_shape = ov::Shape{input.begin(), input.end()};

    make_input();

    make_ref_output();
}

std::string PermuteTestsBase::ToString() const {
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
bool PermuteTestsBase::isNegative() const {
    return false;
}

void PermuteTestsBase::make_ref_output() {
    std::vector<size_t> dims(input_shape.begin(), input_shape.end());
    if (type == ov::element::f16) {
        details::permute<uint16_t>(reinterpret_cast<const uint16_t*>(input.data()),
                                   reinterpret_cast<uint16_t*>(ref_output.data()),
                                   dims,
                                   axes);
    } else if (type == ov::element::f32) {
        details::permute<float>(reinterpret_cast<const float*>(input.data()),
                                reinterpret_cast<float*>(ref_output.data()),
                                dims,
                                axes);
    } else if (type == ov::element::i4) {
        details::permute<uint8_t, true>(reinterpret_cast<const uint8_t*>(input.data()),
                                        reinterpret_cast<uint8_t*>(ref_output.data()),
                                        dims,
                                        axes);
    } else {
        throw std::runtime_error("Unsupported element type in PermuteTestsBase::make_ref_output");
    }
}

TEST_P(PermuteTests, permute) {
    ASSERT_NO_THROW_WITH_MESSAGE(outTensor = ov::npuw::util::permute(inTensor, axes));
    int8_t* dst = static_cast<int8_t*>(outTensor.data());
    output = std::vector<int8_t>(dst, dst + output.size());
    ASSERT_TRUE(details::ArraysMatch(output, ref_output));
}

const auto TestCases201 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f32}),
        ::details::ShapesIn({Tensors{input={1, 2, 4};
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
}),
::testing::ValuesIn(
    {
    std::vector<std::size_t>({2, 0, 1})
}
)
);

INSTANTIATE_TEST_SUITE_P(PermuteTests201, PermuteTests, TestCases201, PermuteTests::getTestCaseName);

const auto TestCases021 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f16, ov::element::Type_t::f32}),
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
, Tensors {
    input = {32,128,1288};
}
}),
::testing::ValuesIn(
    {
    std::vector<std::size_t>({0, 2, 1})
}
)
);

INSTANTIATE_TEST_SUITE_P(PermuteTests021, PermuteTests, TestCases021, PermuteTests::getTestCaseName);

const auto TestCases102 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f16}),
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
}),
::testing::ValuesIn(
    {
    std::vector<std::size_t>({1, 0, 2})
}
)
);

INSTANTIATE_TEST_SUITE_P(PermuteTests102, PermuteTests, TestCases102, PermuteTests::getTestCaseName);

const auto TestCases120 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
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
}),
::testing::ValuesIn(
    {
    std::vector<std::size_t>({1, 2, 0})
}
)
);

INSTANTIATE_TEST_SUITE_P(PermuteTests120, PermuteTests, TestCases120, PermuteTests::getTestCaseName);

}  // anonymous namespace

#endif  // __AVX2__
