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

#include "kernel_selector_params.h"

#include <cfloat>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <cfloat>
#include <string>
#include <vector>

#define AGE_BASED "-cl-no-subgroup-ifp"
#define DEFAULT ""
#define NO_PRERA_SCH "-cl-intel-no-prera-scheduling"

namespace kernel_selector {

#ifndef UNUSED
#define UNUSED(a) (void)a
#endif


// TODO: current solution until we will have kernel selection time based
#define FORCE_PRIORITY_1 (0.0000001f)
#define FORCE_PRIORITY_2 (0.0000002f)
#define FORCE_PRIORITY_3 (0.0000003f)
#define FORCE_PRIORITY_4 (0.0000004f)
#define FORCE_PRIORITY_5 (0.0000005f)
#define FORCE_PRIORITY_6 (0.0000006f)
#define FORCE_PRIORITY_7 (0.0000007f)
#define FORCE_PRIORITY_8 (0.0000008f)
#define FORCE_PRIORITY_9 (0.0000009f)
#define DONT_USE_IF_HAVE_SOMETHING_ELSE (1000000.f)
#define TUTORIAL_PRIORITY (DONT_USE_IF_HAVE_SOMETHING_ELSE + 1.f)
#define NOT_SUPPORTED (FLT_MAX)

    std::string GetStringEnv(const char* varName);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelString
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct KernelString
    {
        std::string str;
        std::string jit;
        std::string options;
        std::string entry_point;
        bool        batch_compilation;

        KernelString() :
            str(""), jit(""),
            options(""), entry_point(""),
            batch_compilation(false)
        {};

        std::string get_hash()
        {
            return str + jit + options + entry_point;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WorkGroupSizes
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WorkGroupSizes
    {
        std::vector<size_t> global;
        std::vector<size_t> local;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Scalar
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ScalarDescriptor
    {
        union ValueT
        {
            uint8_t  u8;
            uint16_t u16;
            uint32_t u32;
            uint64_t u64;
            int8_t   s8;
            int16_t  s16;
            int32_t  s32;
            int64_t  s64;
            float    f32;
            double   f64;
        };

        enum class Types
        {
            UINT8,
            UINT16,
            UINT32,
            UINT64,
            INT8,
            INT16,
            INT32,
            INT64,
            FLOAT32,
            FLOAT64,
        };

        Types t;
        ValueT v;
    };

    using Scalars = std::vector<ScalarDescriptor>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ArgumentDescpirtor
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ArgumentDescriptor
    {
        enum class Types
        {
            INPUT,
            OUTPUT,
            WEIGHTS,
            BIAS,
            PREV_WEIGHTS_GRADIENT,
            PREV_BIAS_GRADIENT,
            SCALE_TABLE,
            SLOPE,
            SPLIT,
            INTERNAL_BUFFER,
            SCALAR,
            WEIGHTS_QUANTIZATION_FACTORS,
            OUTPUT_CALIBRATION_FACTORS,
            RECURRENT, // RNN/LSTM/GRU recurrent weights
            HIDDEN,    // RNN/LSTM/GRU hidden input
            CELL,      // LSTM cell input
            LSTM_PACK, // LSTM packed output
            LEARNING_RATE
        };

        enum class ScalarTypes
        {
            UINT8,
            UINT16,
            UINT32,
            UINT64,
            INT8,
            INT16,
            INT32,
            INT64,
            FLOAT32,
            FLOAT64,
        };

        Types t;
        uint32_t index;
    };

    using Arguments = std::vector<ArgumentDescriptor>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // clKernelData
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct clKernelData
    {
        std::shared_ptr<KernelString>   kernelString;
        WorkGroupSizes                  workGroups;
        Arguments                       arguments;
        Scalars                         scalars;
        std::string                     layerID;            // TODO: in order to support run single layer. think about more appropriate place
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CPUKernel
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct CPUKernel
    {
        virtual WeightsType   GetExpectedInputType() = 0;
        virtual WeightsLayout GetExpectedInputLayout() const { return WeightsLayout::oiyx; }
        virtual void Execute(void* input, size_t input_size, void* output, size_t output_size) const = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GenericKernelParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct GenericKernelParams
    {
        enum class Engine
        {
            NONE,
            CPU,
            GPU
        };

        Engine engine = Engine::NONE;
        std::shared_ptr<clKernelData> clKernel;
        std::shared_ptr<CPUKernel> cpuKernel;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WeightsReorderParams : public GenericKernelParams
    {
        size_t newBufferSize = 0;
        WeightsType dtype = WeightsType::F16;
        WeightsLayout destLayout = WeightsLayout::oiyx;
        bool toImageType = false;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelData
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct KernelData
    {
        std::shared_ptr<Params> params;
        std::vector<clKernelData> kernels;
        std::vector<size_t> internalBufferSizes;
        float estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        uint64_t runTime = std::numeric_limits<uint64_t>::max(); // kernel run time in nanoseconds

        bool reorderInput = false;
        WeightsReorderParams weightsReorderParams;
        std::string kernelName;

        int autoTuneIndex = -1;

        template <typename T>
        inline static KernelData Default(const Params& _params, size_t kernel_nums = 1)
        {
            KernelData kd;
            const T& orgParams = static_cast<const T&>(_params);
            kd.params = std::make_shared<T>(orgParams);
            kd.kernels.resize(kernel_nums);
            kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE; // for KW
            kd.runTime = std::numeric_limits<uint64_t>::max();
            kd.reorderInput = false; // for KW
            kd.autoTuneIndex = -1;
            return kd;
        }
    };

    using KernelsData = std::vector<KernelData>;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // to string functions
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string toString(ActivationFunction activation);
    std::string toString(DataLayout l);
    std::string toString(Datatype dType);
    std::string toString(WeightsType wType);
    std::string toString(KernelType kt);
    std::string toString(EltwiseMode b_mode);
    std::string toString(ReorderMode mode);
    std::string toString(MeanSubtractMode mode);
    std::string toString(ArgMaxMinOut mode);
    std::string toString(ArgMaxMinAxis mode);
    std::string toString(LookUpTableAxis mode);
    std::string toString(PoolType mode);
    std::string toString(LRNMode mode);
    std::string toString(KernelDividerMode mode);
    std::string toString(SoftmaxDim d);
    std::string toString(NormalizeMode mode);
    std::string toString(MVNMode mode);
    std::string toString(WeightsLayout layout);
    std::string toString(ConcatAxis a);
    std::string toString(TileAxis a);
    std::string toString(GatherAxis a);
    std::string toString(SampleType type);
    std::string toString(const BorderType type);
    std::string toString(const Tensor::Dim& dim);
    std::string toString(const DataTensor& tensor);
    std::string toString(const IndexSelectAxis& axis);
    inline std::uint64_t create_hash(const unsigned char* begin, const unsigned char* end)
    {
        // Compatible with VS std::hash.
        constexpr auto start_acc  = static_cast<std::uint64_t>(UINT64_C(14695981039346656037));
        constexpr auto mul_factor = static_cast<std::uint64_t>(UINT64_C(1099511628211));

        std::uint64_t acc = start_acc;
        for (auto elem_it = begin; elem_it != end; ++elem_it)
        {
            acc ^= static_cast<std::uint64_t>(*elem_it);
            acc *= mul_factor;
        }

        return acc;
    }

    template <typename ElemTy>
    std::uint64_t create_hash(const ElemTy* begin, const std::size_t size)
    {
        return create_hash(reinterpret_cast<const unsigned char*>(begin), reinterpret_cast<const unsigned char*>(begin + size));
    }

    template <typename CharTy, typename CharTraits, typename AllocatorTy>
    std::uint64_t create_hash(const std::basic_string<CharTy, CharTraits, AllocatorTy>& value)
    {
        return create_hash<CharTy>(value.data(), value.size());
    }
}
