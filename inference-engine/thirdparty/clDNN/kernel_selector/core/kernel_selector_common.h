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
#include <limits>
#include <string>
#include <memory>
#include "primitive_db.h"
#include "kernel_selector_params.h"
#include <float.h>
#include <sstream>

#define AGE_BASED "-cl-no-subgroup-ifp"
#define ROUND_ROBIN ""

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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // usings
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    using primitive_db = kernel_selector::gpu::cache::primitive_db;

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
    inline std::string toString(ActivationFunction activation)
    {
        std::string method("LINEAR");
        switch (activation)
        {
        case ActivationFunction::LOGISTIC:              method = "LOGISTIC"; break;
        case ActivationFunction::HYPERBOLIC_TAN:        method = "HYPERBOLIC_TAN"; break;
        case ActivationFunction::RELU:                  method = "RELU"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:   method = "RELU_NEGATIVE_SLOPE"; break;
        case ActivationFunction::CLAMP:                 method = "CLAMP"; break;
        case ActivationFunction::SOFTRELU:              method = "SOFTRELU"; break;
        case ActivationFunction::ABS:                   method = "ABS"; break;
        case ActivationFunction::SQUARE:                method = "SQUARE"; break;
        case ActivationFunction::SQRT:                  method = "SQRT"; break;
        case ActivationFunction::LINEAR:                method = "LINEAR"; break;
        case ActivationFunction::ELU:                   method = "ELU"; break;
        case ActivationFunction::RELU_GRAD:             method = "RELU_GRAD"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD: method = "RELU_NEGATIVE_SLOPE_GRAD"; break;
        case ActivationFunction::NONE:                  method = "NONE"; break;
        case ActivationFunction::NONE_GRAD:             method = "NONE_GRAD"; break;
        default: break;
        }
        return method;
    }

    inline std::string toString(DataLayout l)
    {
        switch (l)
        {
        case kernel_selector::DataLayout::bf:                return "BF";
        case kernel_selector::DataLayout::fb:                return "FB";
        case kernel_selector::DataLayout::bfyx:              return "BFYX";
        case kernel_selector::DataLayout::yxfb:              return "YXFB";
        case kernel_selector::DataLayout::byxf:              return "BYXF";
        case kernel_selector::DataLayout::fyxb:              return "FYXB";
        case kernel_selector::DataLayout::bs_f_bsv8__af8:    return "BS_F_BSV8__AF8";
        case kernel_selector::DataLayout::bs_f_bsv16__af8:   return "BS_F_BSV16__AF8";
        case kernel_selector::DataLayout::bf8_xy16:          return "BF8_XY16";
        case kernel_selector::DataLayout::brfyx:             return "BRFYX";
        case kernel_selector::DataLayout::winograd_2x3_s1_data: return "WINOGRAD_2x3_S1_DATA";
        case kernel_selector::DataLayout::byxf_af32: return "BYXF_AF32";
        default: return "";
        }
    }

    inline std::string toString(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::INT8:    return "INT8";
        case Datatype::UINT8:   return "UINT8";
        case Datatype::INT16:   return "INT16";
        case Datatype::UINT16:  return "UINT16";
        case Datatype::INT32:   return "INT32";
        case Datatype::UINT32:  return "UINT32";
        case Datatype::F16:     return "F16";
        case Datatype::F32:     return "F32";
        default: return "";
        }
    }

    inline std::string toString(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::F16:  return "F16";
        case WeightsType::F32:  return "F32";
        case WeightsType::INT8: return "INT8";
        default: return "";
        }
    }

    inline std::string toString(KernelType kt)
    {
        switch (kt)
        {
        case KernelType::UNKNOWN:           return "UNKNOWN";
        case KernelType::CONVOLUTION:       return "CONVOLUTION";
        case KernelType::LRN:               return "LRN";
        case KernelType::POOLING:           return "POOLING";
        case KernelType::ROI_POOLING:       return "ROI_POOLING";
        case KernelType::FULLY_CONNECTED:   return "FULLY_CONNECTED";
        case KernelType::ACTIVATION:        return "ACTIVATION";
        case KernelType::SOFT_MAX:          return "SOFT_MAX";
        case KernelType::REGION_YOLO:       return "REGION_YOLO";
        case KernelType::REORG_YOLO:        return "REORG_YOLO";
        case KernelType::ELTWISE:           return "ELTWISE";
        case KernelType::REORDER:           return "REORDER";
        default:
            return "";
        }
    }

    inline std::string toString(EltwiseMode b_mode)
    {
        switch (b_mode)
        {
        case EltwiseMode::ADD:      return "ADD";
        case EltwiseMode::SUB:      return "SUB";
        case EltwiseMode::MUL:      return "MUL";
        case EltwiseMode::DIV:      return "DIV";
        case EltwiseMode::MIN:      return "MIN";
        case EltwiseMode::MAX:      return "MAX";
        case EltwiseMode::POW:      return "POW";
        case EltwiseMode::MODULU:   return "MODULU";
        case EltwiseMode::SQRT:     return "SQRT";
        case EltwiseMode::RSQRT:    return "RSQRT";
        case EltwiseMode::ASSIGN:   return "ASSIGN";
        default:
            return "";
        }
    }

    inline std::string toString(ReorderMode mode)
    {
        switch (mode)
        {
        case ReorderMode::xyzw: return "XYZW";
        case ReorderMode::xywz: return "XYWZ";
        case ReorderMode::xwyz: return "XWYZ";
        case ReorderMode::wxyz: return "WXYZ";
        case ReorderMode::xzyw: return "XZYW";
        case ReorderMode::zyxw: return "ZYXW";
        case ReorderMode::yxzw: return "YXZW";
        default: return "XYZW";
        }
    }

    inline std::string toString(MeanSubtractMode mode)
    {
        switch (mode)
        {
        case MeanSubtractMode::NONE:            return "NONE";
        case MeanSubtractMode::INSIDE_PARAMS:   return "INSIDE_PARAMS";
        case MeanSubtractMode::IN_BUFFER:       return "IN_BUFFER";
        default: return "";
        }
    }

    inline std::string toString(ArgMaxMinOut mode)
    {
        switch (mode)
        {
        case ArgMaxMinOut::MAX: return "MAX";
        case ArgMaxMinOut::MIN: return "MIN";
        default: return "";
        }
    }

	inline std::string toString(ArgMaxMinAxis mode) 
	{
		switch (mode)
		{
		case ArgMaxMinAxis::BATCH: return "BATCH";
		case ArgMaxMinAxis::FEATURE: return "FEATURE";
		case ArgMaxMinAxis::X: return "X";
		case ArgMaxMinAxis::Y: return "Y";
		case ArgMaxMinAxis::XYF: return "XYF";
		default: return "";
		}
	}

    inline std::string toString(LookUpTableAxis mode)
    {
        switch (mode)
        {
        case LookUpTableAxis::BATCH: return "BATCH";
        case LookUpTableAxis::FEATURE: return "FEATURE";
        case LookUpTableAxis::X: return "X";
        case LookUpTableAxis::Y: return "Y";
        case LookUpTableAxis::XYF: return "XYF";
        default: return "";
        }
    }

    inline std::string toString(PoolType mode)
    {
        switch (mode)
        {
        case PoolType::MAX: return "MAX";
        case PoolType::AVG: return "AVG";
        case PoolType::MAX_WITH_ARGMAX: return "MAX_WITH_ARGMAX";
        default: return "";
        }
    }

    inline std::string toString(LRNMode mode)
    {
        switch (mode)
        {
        case LRNMode::ACROSS_CHANNEL: return "ACROSS_CHANNEL";
        case LRNMode::WITHIN_CHANNEL: return "WITHIN_CHANNEL";
        default: return "";
        }
    }

    inline std::string toString(KernelDividerMode mode)
    {
        switch (mode)
        {
        case KernelDividerMode::DONT_CARE:  return "DONT_CARE";
        case KernelDividerMode::FIXED:      return "FIXED";
        case KernelDividerMode::DYNAMIC:    return "DYNAMIC";
        case KernelDividerMode::DYNAMIC_WITH_PADDING:    return "DYNAMIC_WITH_PADDING";
        default: return "";
        }
    }

    inline std::string toString(SoftmaxDim d)
    {
        switch (d)
        {
        case SoftmaxDim::X:         return "X";
        case SoftmaxDim::Y:         return "Y";
        case SoftmaxDim::FEATURE:   return "FEATURE";
        default: return "";
        }
    }

    inline std::string toString(NormalizeMode mode)
    {
        switch (mode)
        {
        case NormalizeMode::ACROSS_SPATIAL:         return "ACROSS_SPATIAL";
        case NormalizeMode::WITHIN_SPATIAL:         return "WITHIN_SPATIAL";
        default: return "";
        }
    }

    inline std::string toString(MVNMode mode)
    {
        switch (mode)
        {
        case MVNMode::ACROSS_CHANNELS:         return "ACROSS_CHANNELS";
        case MVNMode::WITHIN_CHANNELS:         return "WITHIN_CHANNELS";
        default: return "";
        }
    }

    inline std::string toString(WeightsLayout layout)
    {
        switch (layout)
        {
        case WeightsLayout::oi:                         return "OI";
        case WeightsLayout::io:                         return "IO";
        case WeightsLayout::oiyx:                       return "OIYX";
        case WeightsLayout::oyxi:                       return "OYXI";
        case WeightsLayout::iyxo:                       return "IYXO";
        case WeightsLayout::yxio:                       return "YXIO";
        case WeightsLayout::os_iyx_osv16:               return "OS_IYX_OSV16";
        case WeightsLayout::os_iyx_osv16_rotate_180:    return "OS_IYX_OSV16_ROTATE_180";
        case WeightsLayout::os_i_osv16:                 return "OS_I_OSV16";
        case WeightsLayout::os_i_osv8__ai8:             return "OS_I_OSV8__AI8";
        case WeightsLayout::os_i_osv16__ai8:            return "OS_I_OSV16__AI8";
        case WeightsLayout::i_yxs_os_yxsv2_osv16:       return "I_YXS_OS_YXSV2_OSV16";
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:  return "IY_XS_OS_XSV2_OSV16__AO32";
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:   return "IY_XS_OS_XSV2_OSV8__AO32";
        case WeightsLayout::image_2d_weights_c4_fyx_b:  return "IMAGE_2D_WEIGHTS_C4_FYX_B";
        case WeightsLayout::image_2d_weights_c1_b_fyx:  return "IMAGE_2D_WEIGHTS_C1_B_FYX";
        case WeightsLayout::winograd_2x3_s1_weights:    return "WINOGRAD_2x3_S1_WEIGHTS";
        case WeightsLayout::winograd_2x3_s1_fused_weights:    return "WINOGRAD_2x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::winograd_6x3_s1_fused_weights:    return "WINOGRAD_6x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb: return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_FBXYB";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb: return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_XFBYB";
        case WeightsLayout::os_is_yx_isa8_osv8_isv4: return "OS_IS_YX_ISA8_OSV8_ISV4";
        default:
            return "";
            break;
        }
    }

    inline std::string toString(ConcatAxis a)
    {
        switch (a)
        {
        case ConcatAxis::X:         return "X";
        case ConcatAxis::Y:         return "Y";
        case ConcatAxis::FEATURE:   return "FEATURE";
        case ConcatAxis::BATCH:     return "BATCH";
        default: return "";
        }
    }

    inline std::string toString(SampleType type)
    {
        switch (type)
        {
        case SampleType::NEAREST: return "SAMPLE_TYPE_NEAREST";
        case SampleType::BILINEAR: return "SAMPLE_TYPE_BILINEAR";
        default: return "";
        }
    }

    inline std::string toString(NonLinearParams params)
    {
        std::stringstream s;
        s << "m" << params.m << "_n" << params.n;
        return s.str();
    }

    inline std::string toString(Tensor::Dim dim)
    {
        std::stringstream s;
        s << "v" << dim.v << "_p" << dim.pitch << "_" << dim.pad.before << "_" << dim.pad.after;
        return s.str();
    }

    inline std::string toString(DataTensor tensor)
    {
        std::stringstream s;
        s << toString(tensor.GetDType()) << "_";
        s << toString(tensor.GetLayout()) << "_";
        int i = 0;
        for (auto dim : tensor.GetDims())
        {
            s << "d" << i << "_" << toString(dim) << "_";
            i++;
        }
        return s.str();
    }

}
