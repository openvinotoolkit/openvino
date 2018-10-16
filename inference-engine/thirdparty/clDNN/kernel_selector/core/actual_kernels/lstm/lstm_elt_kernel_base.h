/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lstm_elt_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct lstm_elt_params : public base_params
    {
        enum order_type : int32_t {
            offset_iofz, // ONNX default
            offset_ifoz, // caffe
        };

        lstm_elt_params()
        : base_params(KernelType::LSTM_ELT)
        {}

        DataTensor cell;
        bool has_cell = false;
        order_type gate_order = offset_iofz;
        float clip = 0;
        bool input_forget = false;

        size_t GetOffsetIndex(order_type type, size_t idx) const {
            static const std::map<order_type, std::vector<size_t>> offset_map {
                {offset_iofz, {0, 1, 2, 3}},
                {offset_ifoz, {0, 2, 1, 3}}
            };
            return offset_map.at(type)[idx];
        }

        size_t GetOffsetIndexI() const { return GetOffsetIndex(gate_order, 0); }
        size_t GetOffsetIndexO() const { return GetOffsetIndex(gate_order, 1); }
        size_t GetOffsetIndexF() const { return GetOffsetIndex(gate_order, 2); }
        size_t GetOffsetIndexZ() const { return GetOffsetIndex(gate_order, 3); }

        void SetOffsetOrder(int32_t t) {
            gate_order = static_cast<order_type>(t);
        }

        void SetCell(const DataTensor& v) {
            cell = v;
            has_cell = true;
        }

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = base_params::GetParamsKey();
            if (has_cell)
            {
                k.EnableLSTMEltCell();
            }
            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // lstm_elt_optional_params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct lstm_elt_optional_params : optional_params
    {
        lstm_elt_optional_params() : optional_params(KernelType::LSTM_ELT) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LSTMEltKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class LSTMEltKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~LSTMEltKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {};

    protected:
        virtual JitConstants GetJitConstants(const lstm_elt_params& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;

        bool Validate(const Params& p, const optional_params&) const override
        {
            if (p.GetType() != KernelType::LSTM_ELT)
            {
                return false;
            }

            return true;
        }
    };
}