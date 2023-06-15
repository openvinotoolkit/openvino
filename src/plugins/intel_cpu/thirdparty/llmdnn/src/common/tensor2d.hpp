// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <memory>
#include <iostream>
#include <functional>
#include <assert.h>
#ifdef ENABLE_NUMA
#include "numa.h"
#endif
#include "bf16.hpp"

#define rndup(x, n) (((x + n - 1)/n)*n)

template<typename T>
struct tensor2D {
    int dims[2] = {0};
    T* data = nullptr;
    int64_t capacity = 0;
    int stride = 0;
    bool force_compact = false;
    bool own;
    int padded_dim1 = 0;

    tensor2D() = default;
    tensor2D(const tensor2D&) = delete;
    ~tensor2D() {
        if (own && data) ::free(data);
    }

    operator bool() {
        return dims[0] * dims[1] > 0;
    }

    tensor2D(int d0, int d1, bool _force_compact = false) {
        capacity = 0;
        resize(d0, d1, _force_compact);
    }

    tensor2D(int d0, int d1, T * ext, int _stride) {
        capacity = 1;
        data = ext;
        own = false;
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        padded_dim1 = stride / sizeof(T);
    }

    tensor2D<T> Tr(bool _force_compact = false) {
        tensor2D<T> ret(dims[1], dims[0], _force_compact);
        for(int c0=0; c0 < dims[0]; ++c0) {
            for(int c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }
    tensor2D<T> clone() {
        tensor2D<T> ret;
        ret.resize(dims[0], dims[1], force_compact);
        if (ret.stride == stride) {
            memcpy(ret.data, data, dims[0] * stride);
        }else{
            for(int i=0;i<dims[0];i++) {
                memcpy(&ret(i,0), &(*this)(i,0), ret.stride);
            }
        }
        return ret;
    }
    void resize(int d0, int d1, bool _force_compact = false, bool is_const=false) {
        own = true;
        force_compact = _force_compact;
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        if ((stride % 64) && (!force_compact)) {
            auto stride_fix = rndup(stride, 64);
            std::cout << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
                      << " (" << stride_fix/64 << " cache lines)\n";
            stride = stride_fix;
        }
        padded_dim1 = stride / sizeof(T);

        // resize method never shrink capacity, and extra T is added to put nan as test
        auto need_capacity = dims[0] * stride + 4096;
        if (capacity < need_capacity) {
            if (!is_const)
                need_capacity *= 2;
            capacity = need_capacity;
            // align begin address to cache line is vital, so tile load can
            // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)

#ifdef ENABLE_NUMA
            if (USE_NUMA) {
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(numa_alloc_local(capacity)),
                            [need_capacity](void * p){ numa_free(p, need_capacity); });
            } else {
#else
            {
#endif
                if (data) ::free(data);
                data = reinterpret_cast<T*>(aligned_alloc(64, capacity));
            }
            if (is_const)
                memset(data, 0, need_capacity);
            if (reinterpret_cast<uintptr_t>(data) % 64)
                std::cout << "WARNING: resize(), data is not cache-line aligned!" << std::endl;
        }
        // put a NaN at the end to test over-read
        // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
        // #define INF 0xff80 
        // #define NAN1 (INF + 1)
        // if (sizeof(T) == 2) {
        //     *reinterpret_cast<uint16_t*>(data.get() + dims[0] * padded_dim1) = NAN1;
        // }
    }

    T & operator[](int i) {
        return data[i];
    }

    const T & operator[](int i) const {
        return data[i];
    }

    //https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments
    T & operator()(int i0, int i1) {
        return (*this)[i0 * padded_dim1 + i1];
    }

    const T & operator()(int i0, int i1) const {
        return (*this)[i0 * padded_dim1 + i1];
    }


    void operator=(const T & v) {
        for(int k = 0; k < dims[0] * padded_dim1; k++)
            (*this)[k] = v;
    }

    tensor2D<T>& operator=(const tensor2D<T> & t2) = delete;

    // move semantics
    tensor2D(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        if (own && data) ::free(data);
        data = t2.data;
        own = t2.own;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data = nullptr;
    }

    tensor2D<T>&  operator=(tensor2D<T> && t2) {
        dims[0] = t2.dims[0];
        dims[1] = t2.dims[1];
        if (own && data) ::free(data);
        own = t2.own;
        data = t2.data;
        capacity = t2.capacity;
        stride = t2.stride;
        padded_dim1 = t2.padded_dim1;
        force_compact = t2.force_compact;
        t2.capacity = 0;
        t2.data = nullptr;
        return *this;
    }
};
