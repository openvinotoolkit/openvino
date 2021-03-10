// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "base.hpp"
#include "caseless.hpp"
#include "ie_parallel.hpp"

#include <string>
#include <vector>
#include <set>
#include <cmath>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct jit_eximpat_params {
    int OB, IC, IH, IW; // from input shape
    int OH, OW; //from out shape
    int KH, KW; // kernel sizes
    int SH, SW; // strides
    int RH, RW; // rates
    int PL, PT; // determine padding
    int dtype_size; // bit size of the datatype
};

struct jit_eximpat_args {
    const void* src;
    const void* dst;
};

struct jit_uni_eximpat_kernel {
    void (*ker_)(const jit_eximpat_args *);

    void operator()(const jit_eximpat_args *args) { assert(ker_); ker_(args); }

    jit_eximpat_params jpp;

    virtual void create_ker() = 0;

    explicit jit_uni_eximpat_kernel(jit_eximpat_params jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_eximpat_kernel() {}
};

using details::CaselessEq;

class ExtractImagePatchesImpl : public ExtLayerBase {
public:
    explicit ExtractImagePatchesImpl(const CNNLayer*);
    StatusCode execute(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&, ResponseDesc*) noexcept override;

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept {
        const T* src_data = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dst_data = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[0]->getTensorDesc().getDims();
        const size_t inDimsSize = inDims.size(); // Must always be 4 according to the specs.

        const size_t BATCH = 0, CHANNEL = 1, HIGHT = 0, WIDTH = 1;

        const int64_t IC = inDims[CHANNEL];
        const int64_t IH = inDims[inDimsSize - 2];
        const int64_t IW = inDims[inDimsSize - 1];

        const auto& outDims = outputs[0]->getTensorDesc().getDims();
        const size_t outDimsSize = outDims.size(); // Must always be 4 according to the specs.

        const int64_t OB = outDims[BATCH];
        //const int64_t OC = outDims[CHANNEL]; // Must always be KH * KW * IC according to the specs.
        const int64_t OH = outDims[outDimsSize - 2];
        const int64_t OW = outDims[outDimsSize - 1];

        const int64_t KH = _ksizes[HIGHT];
        const int64_t KW = _ksizes[WIDTH];
        const int64_t SH = _strides[HIGHT];
        const int64_t SW = _strides[WIDTH];
        const int64_t RH = _rates[HIGHT];
        const int64_t RW = _rates[WIDTH];
        const int64_t PL = _pads[HIGHT];
        const int64_t PT = _pads[WIDTH];

        const std::vector<int64_t> ostrides = {KH * KW * IC * OH * OW, KW * IC * OH * OW, IC * OH * OW, OH * OW};
        const std::vector<int64_t> istrides = {IC * IH * IW, IH * IW, IW};
        auto thread_body = [&](const int64_t ob, const int64_t kh, const int64_t kw, const int64_t ic) {
            const int64_t iw_start = kw * RW - PL;
            const int64_t iw_stop = iw_start + OW * SW;
            const int64_t ih_start = kh * RH - PT;
            const int64_t ih_stop = ih_start + OH * SH;
            int64_t dst_idx = ob * ostrides[0]  + kh * ostrides[1] + kw * ostrides[2] + ic * ostrides[3];
            int64_t ishift = ob * istrides[0] + ic * istrides[1] + ih_start * istrides[2];
            for (int64_t ih = ih_start; ih < ih_stop; ih += SH, ishift += SH * IW) {
                for (int64_t iw = iw_start; iw < iw_stop; iw += SW, dst_idx++) {
                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                        dst_data[dst_idx] = T(0);
                    } else {
                        dst_data[dst_idx] = src_data[ishift + iw];
                    }
                }
            }
        };
        parallel_for4d(OB, KH, KW, IC, thread_body);
    }

private:
    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    std::vector<int64_t> _pads;
    std::string _auto_pad;

    std::shared_ptr<jit_uni_eximpat_kernel> eximpat_kernel;
    static const std::set<size_t> _supported_precisions_sizes;
    inline void set_pads(const std::string & pad_str, const std::vector<int64_t> & params);
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
