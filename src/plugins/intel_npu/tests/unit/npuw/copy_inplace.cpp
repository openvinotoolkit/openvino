// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#    include "copy_inplace.hpp"

namespace {
static ov::Strides make_padded_strides_keep_tail_default(const ov::Shape& shape,
                                                         const ov::element::Type& et,
                                                         size_t kv_dim,
                                                         size_t pad_elems) {
    ov::Strides s = copy_inplace_details::default_byte_strides(shape, et);

    const size_t rank = shape.size();
    if (rank == 0) {
        return s;
    }
    if (rank == 1) {
        if (kv_dim == 0) {
            s[0] += pad_elems * et.size();
        }
        return s;
    }

    s[rank - 1] = et.size();
    for (size_t d = rank - 1; d-- > 0;) {
        s[d] = s[d + 1] * shape[d + 1];
        if (d == kv_dim) {
            s[d] += pad_elems * s[d + 1];
        }
    }

    return s;
}

static std::vector<int8_t> to_i8(const std::vector<uint8_t>& v) {
    std::vector<int8_t> out(v.size());
    std::memcpy(out.data(), v.data(), v.size());
    return out;
}

void CopyInplaceTestsBase::make_input() {
    const auto elem_bytes = copy_inplace_details::elem_size_bytes(type);
    const auto total_elems = ov::shape_size(shape);
    ASSERT_GT(total_elems, 0u);

    auto max_offset = [&](const ov::Strides& strides) -> size_t {
        size_t off = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            off += (shape[d] - 1) * strides[d];
        }
        return off;
    };

    const size_t src_max = max_offset(src_strides);
    const size_t dst_max = max_offset(dst_strides);
    const size_t byte_size = std::max(src_max, dst_max) + elem_bytes;

    base_bytes_initial.resize(byte_size);
    ref_bytes.assign(byte_size, 0);
    out_bytes.assign(byte_size, 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < base_bytes_initial.size(); ++i) {
        base_bytes_initial[i] = static_cast<uint8_t>(dist(rng));
    }

    baseTensor = ov::Tensor(ov::element::u8, ov::Shape{byte_size}, base_bytes_initial.data());
}

bool CopyInplaceTestsBase::isNegative() const {
    if (shape.size() < 2) {
        return true;
    }
    if (kv_dim >= shape.size()) {
        return true;
    }
    if (type.bitwidth() < 8) {
        return true;
    }
    return false;
}

void CopyInplaceTestsBase::make_ref_output() {
    ref_bytes = base_bytes_initial;

    const auto elem_bytes = copy_inplace_details::elem_size_bytes(type);
    const uint8_t* base_in = base_bytes_initial.data();

    std::vector<uint8_t> tmp_out = base_bytes_initial;

    ov::Shape idx(shape.size(), 0);
    std::vector<uint8_t> elem(elem_bytes);

    for (;;) {
        copy_inplace_details::read_elem_bytes(base_in, idx, src_strides, elem_bytes, elem.data());
        copy_inplace_details::write_elem_bytes(tmp_out.data(), idx, dst_strides, elem_bytes, elem.data());

        if (!copy_inplace_details::next_index(idx, shape)) {
            break;
        }
    }

    ref_bytes = std::move(tmp_out);
}

void CopyInplaceTestsBase::SetUp(const CopyInplaceTestsParams& getParam) {
    ShapesInitializer shapeInit;
    ov::element::Type_t t;
    std::tie(t, shapeInit, kv_dim) = getParam;

    type = ov::element::Type(t);

    std::vector<int> dims;
    shapeInit(dims);
    shape = ov::Shape{dims.begin(), dims.end()};

    src_strides = copy_inplace_details::default_byte_strides(shape, type);
    const size_t pad_elems = 13;
    dst_strides = make_padded_strides_keep_tail_default(shape, type, kv_dim, pad_elems);

    make_input();

    void* base_ptr = baseTensor.data();
    ASSERT_NE(base_ptr, nullptr);
    srcView = ov::Tensor(type, shape, base_ptr, src_strides);
    dstView = ov::Tensor(type, shape, base_ptr, dst_strides);

    if (!isNegative()) {
        make_ref_output();
    }
}

TEST_P(CopyInplaceTests, copy_tensor_inplace_by_dim_correctness) {
    ASSERT_NO_THROW_IF(!isNegative(), {
        auto src_it = ov::get_tensor_impl(srcView);
        auto dst_it = ov::get_tensor_impl(dstView);

        ov::npuw::util::copy_tensor_inplace_by_dim(src_it,
                                                   dst_it,
                                                   static_cast<uint32_t>(kv_dim),
                                                   static_cast<uint32_t>(kv_dim));

        uint8_t* base_ptr = baseTensor.data<uint8_t>();
        ASSERT_NE(base_ptr, nullptr);
        out_bytes.assign(base_ptr, base_ptr + out_bytes.size());

        ASSERT_TRUE(details::ArraysMatch(to_i8(out_bytes), to_i8(ref_bytes)));
    });
}

// Test cases
const auto TestCases = ::testing::Combine(
    ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::f16, ov::element::Type_t::f32}),
    details::ShapesIn({
        Tensors{ input = {1, 2, 3, 4};
}
, Tensors {
    input = {1, 8, 16, 32};
}
, Tensors {
    input = {1, 16, 33, 64};
}
, Tensors {
    input = {1, 4, 128, 16};
}
,
}),
    ::testing::Values<std::size_t>(0, 1, 2, 3)
);

INSTANTIATE_TEST_SUITE_P(CopyInplaceTests, CopyInplaceTests, TestCases, CopyInplaceTests::getTestCaseName);

}  // namespace

#endif  // HAVE_AVX2
