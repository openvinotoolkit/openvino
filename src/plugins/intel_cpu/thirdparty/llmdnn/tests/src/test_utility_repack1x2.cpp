// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>
#include "gtest/gtest.h"
#include "llm_mm.hpp"
#include "tensor2d.hpp"
#include "tensor2d_helper.hpp"
#include "mm_kernel_amx.hpp"
#include "test_common.hpp"

using namespace std;
using namespace llmdnn;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

using RepackTestParamSet = std::tuple<
        data_type_t                                // data type
        >;

class RepackTest : public TestWithParam<RepackTestParamSet> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RepackTestParamSet>& obj) {
        data_type_t types;
        std::tie(types) = obj.param;

        std::ostringstream result;
        result << dtype_to_str(types);
        return result.str();
    }

protected:
    virtual void SetUp() {
        std::tie(_types) = GetParam();
    };

    template<typename T>
    static void gen_ref(tensor2D<T>& in_t, tensor2D<T>& out) {
        int K = in_t.dims[1];
        int N = in_t.dims[0];
        int kStep = 64 / sizeof(T);
        int K_padded = (K + kStep - 1) / kStep * kStep;
        int Ktails = K % kStep;
        int Kbody = K - Ktails;

        // N_padded : round up to multiple of (2*16)
        int N_unit = 2 * 16;
        int N_padded = (N + N_unit - 1) / N_unit * N_unit;

        // Bo(ni, 0) is a vector flattened from a slice of shape [K_padded x N_unit]
        out.resize(N_padded / N_unit, K_padded * N_unit, true, true);

        tensor2D<T> in_padded;
        in_padded.resize(N_padded, K_padded, true, true);
        for (int i = 0; i < N; i++) {
            // full range
            memcpy(&in_padded(i, 0), &in_t(i, 0), (K - Ktails) * sizeof(T));
            // k tail needs to right aligned
            memcpy(&in_padded(i, K_padded - Ktails), &in_t(i, K - Ktails), Ktails * sizeof(T));
        }
        for (int n = 0; n < N_padded; n += N_unit) {
            for (int k = 0; k < K_padded; k += kStep) {
                // bf16 as example:
                //   [N, K], 2*[16(N), 32(K)] =>
                //   [K, N], 2*[16(K), 16(N)*2(K)]
                for (int m = 0; m < 2; m++) {
                    auto* src = reinterpret_cast<int*>(&in_padded(n + m * 16, k));
                    auto* dst = reinterpret_cast<int*>(&out(n / N_unit, k * N_unit));
                    dst += m * (1024 / sizeof(int));
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 16; j++) {
                            dst[i * 16 + j] = src[j * in_padded.stride / sizeof(int) + i];
                        }
                    }
                }
            }
        }
    }

    template<typename T>
    void test() {
        auto testone = [] (int k, int n, std::string prefix) {
            tensor2D<T> A(k, n, true);

            fill_rnd(A);
            tensor2D<T> AT = A.Tr(true);
            tensor2D<T> A_out, AT_out, A_ref;
            amx_kernel::repackB_1x2(A, false, A_out, false);
            amx_kernel::repackB_1x2(AT, true, AT_out, false);
            gen_ref(AT, A_ref);
            ASSERT_TRUE(A_out == A_ref) << " " << prefix << " without transform K: " << k << " N: " << n;
            ASSERT_TRUE(AT_out == A_ref) << " " << prefix << " with transform K: " << k << " N: " << n;
        };
        // n tail: transpose case needs from 1 to 31, without transpose needs one
        int k = 32;
        int n;
        for (n = 1; n < 32; n++) {
            testone(k, n, "ntail");
        }
        for (n = 32 + 1; n < 32 + 32; n++) {
            testone(k, n, "ntail");
        }
        // k tail: transpose case needs 1, without transpose needs from 1 to 31
        n = 32;
        for (k = 1; k < 32; k++) {
            testone(k, n, "ktail");
        }
        for (k = 32 + 1; k < 32 + 32; k++) {
            testone(k, n, "ktail");
        }
        // k, n normal
        testone(32, 32, "normal");
        testone(64, 128, "normal");
        // k, n tail
        testone(64, 128 + 5, "ntail");
        testone(64 + 3, 128, "ktail");
        testone(64 + 3, 128 + 5, "alltail");
        testone(64, 128 + 16 + 5, "ntail");
        testone(64 + 16 + 3, 128, "ktail");
        testone(64 + 16 + 3, 128 + 16 + 5, "alltail");
    }

    data_type_t _types;
};

TEST_P(RepackTest, Func) {
    if (_types == dnnl_s8) {
        test<int8_t>();
    } else {
        test<ov::bfloat16>();
    }
}

const std::vector<data_type_t> types = {
    dnnl_s8, dnnl_bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_Repack, RepackTest,
    ::testing::Combine(ValuesIn(types)),
    RepackTest::getTestCaseName);
