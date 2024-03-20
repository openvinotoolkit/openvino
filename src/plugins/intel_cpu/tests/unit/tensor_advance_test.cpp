// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include "common_test_utils/test_common.hpp"
#include "nodes/common/tensor_advance.h"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "common_test_utils/common_utils.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu;

namespace TensorAdvanceUT {
using TensorAdvanceTestParam = std::tuple<ov::Shape, int64_t>;

inline MemoryPtr create_memory(ov::element::Type prc, const Shape& shape, void* data = nullptr) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDescPtr desc;
    desc = std::make_shared<CpuBlockedMemoryDesc>(prc, shape);
    return std::make_shared<Memory>(eng, desc, data);
}

struct TensorAdvanceTest : public ov::test::TestsCommon, public testing::WithParamInterface<TensorAdvanceTestParam> {
    static std::string getTestCaseName(const testing::TestParamInfo<TensorAdvanceTestParam>& obj) {
        std::vector<size_t> shape;
        int64_t axis;
        std::tie(shape, axis) = obj.param;
        std::ostringstream result;
        result << "shape=";
        result << ov::test::utils::vec2str(shape);
        result << ", axis=" << axis;
        return result.str();
    }

    void SetUp() override {
        std::tie(_shape, _squashed_axis)= this->GetParam();
    }

    ov::Shape _shape;
    int64_t _squashed_axis;
};

TEST_P(TensorAdvanceTest, ForEachConstInput) {
    if (_shape.size() <= static_cast<size_t>(_squashed_axis)) GTEST_SKIP();

    std::vector<float> out(ov::shape_size(_shape), 0);
    std::vector<float> va(ov::shape_size(_shape), 0);
    std::iota(std::begin(va), std::end(va), 1);  // 1, 2, 3...
    std::copy(std::begin(va), std::end(va), std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';

    auto out_memptr = create_memory(ov::element::f32, Shape(_shape), out.data());
    auto a_memptr = create_memory(ov::element::f32, Shape(_shape), va.data());
    std::array<PlainTensor, 2> arr_memptr = {PlainTensor(out_memptr), PlainTensor(a_memptr)};

     int64_t index_dim_size = _shape[_squashed_axis];
     int64_t out_dim_stride = static_cast<int64_t>(arr_memptr[0].stride_bytes(_squashed_axis));
     int64_t in_dim_stride = static_cast<int64_t>(arr_memptr[1].stride_bytes(_squashed_axis));
     int64_t updates_rank = _shape.size();

    auto my_loop = [&](char** data, const size_t* strides, const size_t n) {
        if (_squashed_axis == updates_rank - 1) {
            auto* out_data = data[0];
            auto* in_data = data[1];
            for (size_t k = 0; k < n; k++) {
                for (int64_t i = 0; i < index_dim_size; i++) {
                    *(reinterpret_cast<float*>(out_data + i * out_dim_stride)) += *(reinterpret_cast<float*>(in_data + i * in_dim_stride));
                }
                out_data += strides[0];
                in_data += strides[1];
            }
        } else {
            for (int64_t i = 0; i < index_dim_size; i++) {
                auto* out_data = data[0];
                auto* in_data = data[1];
                for (size_t k = 0; k < n; k++) {
                    *(reinterpret_cast<float*>(out_data + i * out_dim_stride)) += *(reinterpret_cast<float*>(in_data + i * in_dim_stride));
                    out_data += strides[0];
                    in_data += strides[1];
                }
            }
        }
    };

    ov::Shape squashed_shape(_shape);
    squashed_shape[_squashed_axis] = 1;

    int num_threads = 0;
    const char *penv = std::getenv("NUM_THREADS");
    if (penv) num_threads = std::atoi(penv);
    std::cout << "=============== num_threads=" << num_threads << std::endl;

    TensorAdvance<2> iter(squashed_shape, arr_memptr);
    ov::parallel_nt(num_threads, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        ov::splitter(ov::shape_size(squashed_shape), nthr, ithr, start, end);
        if (start>=ov::shape_size(squashed_shape)) return;
        iter.run(my_loop, start, end);
    });

    std::copy(std::begin(out), std::end(out), std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';
    std::copy(std::begin(va), std::end(va), std::ostream_iterator<int>(std::cout, " "));
    std::cout << '\n';
    EXPECT_TRUE(out==va);
}

namespace {
const std::vector<ov::Shape> shapes = {
    ov::Shape({8}),
    ov::Shape({2, 4}),
    ov::Shape({2, 4, 2}),
    ov::Shape({2, 4, 2, 3}),
    ov::Shape({2, 4, 2, 3, 5}),
    ov::Shape({2, 3, 4, 5, 6, 7, 8})
};

INSTANTIATE_TEST_SUITE_P(smoke_TensorAdvanceTest,
                         TensorAdvanceTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(0, 1, 2, 3, 4, 5)),  // squashed axis
                         TensorAdvanceTest::getTestCaseName);
}  // namespace
} // namespace TensorAdvanceUT
