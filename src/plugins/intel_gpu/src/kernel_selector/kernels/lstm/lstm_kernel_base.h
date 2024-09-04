// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>
#include <map>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lstm_cell_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lstm_params : public base_params {
    enum order_type : int32_t {
        offset_iofz,  // ONNX default
        offset_ifoz,  // caffe
        offset_izof,  // pyTorch
        offset_fizo   // OV default
    };

    lstm_params() : base_params(KernelType::LSTM_SEQ_CELL) {}
    order_type gate_order = offset_iofz;
    float clip = 0;
    bool input_forget = false;
    uint32_t direction = 0;

    size_t GetOffsetIndex(order_type type, size_t idx) const {
        static const std::map<order_type, std::vector<size_t>> offset_map{{offset_iofz, {0, 1, 2, 3}},
                                                                          {offset_ifoz, {0, 2, 1, 3}},
                                                                          {offset_izof, {0, 3, 1, 2}},
                                                                          {offset_fizo, {1, 3, 0, 2}}};
        return offset_map.at(type)[idx];
    }

    size_t GetOffsetIndexI() const { return GetOffsetIndex(gate_order, 0); }
    size_t GetOffsetIndexO() const { return GetOffsetIndex(gate_order, 1); }
    size_t GetOffsetIndexF() const { return GetOffsetIndex(gate_order, 2); }
    size_t GetOffsetIndexZ() const { return GetOffsetIndex(gate_order, 3); }

    void SetOffsetOrder(int32_t t) { gate_order = static_cast<order_type>(t); }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LSTMSeqKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LSTMKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LSTMKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const lstm_params& params, bool, size_t) const;
    KernelsData GetCommonKernelsData(const Params& params, bool, bool) const;

    bool Validate(const Params& p) const override {
        if (p.GetType() != KernelType::LSTM_SEQ_CELL) {
            return false;
        }

        return true;
    }

    static int vec_size_selector(int hidden_size) {
        if (hidden_size%4 == 0) {
            return 4;
        } else if (hidden_size%2 == 0) {
            return 2;
        } else {
            return 1;
        }
    }

    static int gcd(int a, int b) {
        while (a != b) {
            if (a > b) {
                a -= b;
            } else {
                b -= a;
            }
        }
        return a;
    }

    static size_t get_num_hidden_kernels(int hidden_size, int max_work_group_size) {
        if (max_work_group_size >= hidden_size) {
            return static_cast<size_t>(hidden_size);
        } else {
            int gcd_result = gcd(max_work_group_size, hidden_size);
            int minimal_number_when_gcd_low = 16;
            if (gcd_result < minimal_number_when_gcd_low) {
                if (minimal_number_when_gcd_low < max_work_group_size && minimal_number_when_gcd_low < hidden_size) {
                    return static_cast<size_t>(minimal_number_when_gcd_low);
                }
            }
            return static_cast<size_t>(minimal_number_when_gcd_low);
        }
    }
};
}  // namespace kernel_selector
