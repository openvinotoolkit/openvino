// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <functional>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include "llm_types.hpp"
#include "llm_fc.hpp"
#include "tensor2d.hpp"
#include "bf16.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#define rndup(x, n) (((x + n - 1)/n)*n)

std::string dtype_to_str(llmdnn::data_type_t type);

using func_act = std::function<float(float)>;

bool initXTILE();

template<typename TC>
void matmul(tensor2D<ov::bfloat16> & A,
            tensor2D<ov::bfloat16> & B,
            tensor2D<TC> & C,
            float * dq = nullptr,
            float * bias = nullptr,
            func_act act = func_act(),
            float * q = nullptr) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C(m,n);
            int k;
            for (k = 0; (k + 32) <= K; k += 32) {
                float psum0 = 0;
                float psum1 = 0;
                for(int p = 0; p < 32; p+=2) {
                    psum0 += static_cast<float>(A(m,k+p)) * static_cast<float>(B(k+p,n));
                    psum1 += static_cast<float>(A(m,k+p+1)) * static_cast<float>(B(k+p+1,n));
                }
                sum += (psum0 + psum1);
            }
            for(; k < K; k++) {
                sum += static_cast<float>(A(m,k)) * static_cast<float>(B(k,n));
            }
            if (bias) {
                sum += bias[n];
            }
            if (act) {
                sum = act(sum);
            }
            //std::cout << m << "," << n << std::endl;
            C(m,n) = sum;
        }
    }
}

inline void matmul(tensor2D<float> & A,
            tensor2D<float> & B,
            tensor2D<float> & C,
            float * dq = nullptr,
            float * bias = nullptr,
            func_act act = func_act(),
            float * q = nullptr) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C(m,n);
            for(int k = 0; k < K; k++) {
                sum += static_cast<float>(A(m,k)) * static_cast<float>(B(k,n));
            }
            if (bias) {
                sum += bias[n];
            }
            if (act) {
                sum = act(sum);
            }
            C(m,n) = sum;
        }
    }
}

template<typename TA, typename TC>
void matmul(tensor2D<TA> & A,
            tensor2D<int8_t> & B,
            tensor2D<TC> & C,
            float * dq = nullptr,
            float * bias = nullptr,
            func_act act = func_act(),
            float * q = nullptr) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C(m,n);
            for(int k = 0; k < K; k++) {
                sum += static_cast<float>(A(m,k)) * static_cast<float>(B(k,n));
            }
            if (dq) {
                sum *= dq[n];
            }
            if (bias) {
                sum += bias[n];
            }
            if (act) {
                sum = act(sum);
            }
            if (q) {
                sum *= q[n];
                sum = std::min(static_cast<float>(std::numeric_limits<TC>::max()), sum);
                sum = std::max(static_cast<float>(std::numeric_limits<TC>::min()), sum);
            }
            C(m,n) = sum;
        }
    }
}

struct ANSIcolor {
    const char * code;
    ANSIcolor(const char * code = "0") : code(code){
    }
    friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
        out << "\033[" << obj.code << "m";
        return out;
    }
};

struct pretty_size {
    double sz;
    std::string txt;
    pretty_size(double sz, const char * unit = "") : sz(sz) {
        std::stringstream ss;
        ss << std::setprecision(5) << std::setw(5);
        if (sz < 1024)
            ss << sz;
        else if (sz < 1024 * 1024)
            ss << (sz / 1024) << " K";
        else if (sz < 1024 * 1024 * 1024)
            ss << (sz / 1024/1024) << " M";
        else
            ss << (sz / 1024 / 1024/1024) << " G";
        ss << unit;
        txt = ss.str();
    }
    friend std::ostream& operator<<(std::ostream& os, const pretty_size& ps) {
        os << ps.txt;
        return os;
    }
};

inline int readenv(const char * name) {
    int v = 0;
    auto * p = std::getenv(name);
    if (p)
        v = std::atoi(p);
    std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v << std::endl << ANSIcolor();
    return v;
}

template <typename T>
struct TypeName {
    static const char* get(){return typeid(T).name();}
};

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
template <>
struct TypeName<int32_t>{
    static const char* get(){return "int32_t";}
};
template <>
struct TypeName<float>{
    static const char* get(){return "foat";}
};
template <>
struct TypeName<ov::bfloat16>{
    static const char* get(){return "bfloat16";}
};
template <>
struct TypeName<int8_t>{
    static const char* get(){return "int8_t";}
};

inline std::ostream & operator<<(std::ostream & os, const llmdnn::postops_types & steps) {
    if (steps == llmdnn::NONE)
        os << "NONE";
    if (steps & llmdnn::DEQUANT)
        os << "_DEQUANT";
    if (steps & llmdnn::BIAS)
        os << "_BIAS";
    if (steps & llmdnn::GELU)
        os << "_GELU";
    if (steps & llmdnn::QUANT)
        os << "_QUANT";
    return os;
}
