// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "infer_request_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "test_utils.hpp"

namespace {

using CopyInplaceTestsParams = std::tuple<ov::element::Type_t, ShapesInitializer, std::size_t>;

namespace copy_inplace_details {

inline ov::Strides default_byte_strides(const ov::Shape& shape, const ov::element::Type& et) {
    ov::Strides strides(shape.size(), 0);
    if (!strides.empty()) {
        strides.back() = et.size();
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
    }
    return strides;
}

inline size_t elem_size_bytes(const ov::element::Type& et) {
    return et.size();
}

inline void read_elem_bytes(const uint8_t* base,
                            const ov::Shape& idx,
                            const ov::Strides& strides,
                            size_t elem_bytes,
                            uint8_t* out_elem) {
    size_t off = 0;
    for (size_t d = 0; d < idx.size(); ++d) {
        off += idx[d] * strides[d];
    }
    std::memcpy(out_elem, base + off, elem_bytes);
}

inline void write_elem_bytes(uint8_t* base,
                             const ov::Shape& idx,
                             const ov::Strides& strides,
                             size_t elem_bytes,
                             const uint8_t* elem) {
    size_t off = 0;
    for (size_t d = 0; d < idx.size(); ++d) {
        off += idx[d] * strides[d];
    }
    std::memcpy(base + off, elem, elem_bytes);
}

inline bool next_index(ov::Shape& idx, const ov::Shape& shape) {
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        const size_t ud = static_cast<size_t>(d);
        if (++idx[ud] < shape[ud]) {
            return true;
        }
        idx[ud] = 0;
    }
    return false;
}

}  // namespace copy_inplace_details

class CopyInplaceTestsBase {
protected:
    ov::element::Type type;
    ov::Tensor baseTensor;
    ov::Tensor srcView;
    ov::Tensor dstView;
    ov::Shape shape;

    std::vector<uint8_t> base_bytes_initial;
    std::vector<uint8_t> ref_bytes;
    std::vector<uint8_t> out_bytes;

    std::size_t kv_dim = 0;

    ov::Strides src_strides;
    ov::Strides dst_strides;

    void make_input();
    void make_ref_output();
    bool isNegative() const;

public:
    void SetUp(const CopyInplaceTestsParams& getParam);
};

template <class T>
class CopyInplaceTestsTmpl : public ::testing::Test,
                             public T,
                             public ::testing::WithParamInterface<CopyInplaceTestsParams> {
protected:
    void SetUp() override {
        T::SetUp(GetParam());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CopyInplaceTestsParams>& obj) {
        ov::element::Type_t t;
        ShapesInitializer shapeInit;
        std::size_t kv_dim = 0;
        std::tie(t, shapeInit, kv_dim) = obj.param;

        std::vector<int> dims;
        shapeInit(dims);

        std::ostringstream oss;
        oss << "S";
        for (size_t i = 0; i < dims.size(); ++i) {
            oss << dims[i];
            if (i + 1 != dims.size())
                oss << "x";
        }
        oss << "_T" << ov::element::Type(t) << "_KV" << kv_dim;
        return oss.str();
    }
};

using CopyInplaceTests = CopyInplaceTestsTmpl<CopyInplaceTestsBase>;

}  // anonymous namespace
