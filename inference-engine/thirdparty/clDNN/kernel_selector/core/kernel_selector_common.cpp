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

#include "kernel_selector_common.h"
#include <sstream>

namespace kernel_selector
{
    std::string GetStringEnv(const char* varName)
    {
        std::string str;
#ifdef _WIN32
        char* env = nullptr;
        size_t len = 0;
        errno_t err = _dupenv_s(&env, &len, varName);
        if (err == 0)
        {
            if (env != nullptr)
            {
                str = std::string(env);
            }
            free(env);
        }
#else
        const char *env = getenv(varName);
        if (env)
        {
            str = std::string(env);
        }
#endif

        return str;
    }

    std::string toString(ActivationFunction activation)
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
        case ActivationFunction::SIN:                   method = "SIN"; break;
        case ActivationFunction::ASIN:                  method = "ASIN"; break;
        case ActivationFunction::SINH:                  method = "SINH"; break;
        case ActivationFunction::COS:                   method = "COS"; break;
        case ActivationFunction::ACOS:                  method = "ACOS"; break;
        case ActivationFunction::COSH:                  method = "COSH"; break;
        case ActivationFunction::LOG:                   method = "LOG"; break;
		case ActivationFunction::LOG2:                  method = "LOG2"; break;
        case ActivationFunction::EXP:                   method = "EXP"; break;
        case ActivationFunction::NONE:                  method = "NONE"; break;
        case ActivationFunction::NONE_GRAD:             method = "NONE_GRAD"; break;
        default: break;
        }
        return method;
    }

    std::string toString(DataLayout l)
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
        case kernel_selector::DataLayout::fs_bs_yx_bsv4_fsv32: return "FS_BS_YX_BSV4_FSV32";
        default: return "";
        }
    }

    std::string toString(Datatype dType)
    {
        switch (dType)
        {
        case Datatype::INT8:    return "INT8";
        case Datatype::UINT8:   return "UINT8";
        case Datatype::INT16:   return "INT16";
        case Datatype::UINT16:  return "UINT16";
        case Datatype::INT32:   return "INT32";
        case Datatype::UINT32:  return "UINT32";
        case Datatype::INT64:   return "INT64";
        case Datatype::F16:     return "F16";
        case Datatype::F32:     return "F32";
        default: return "";
        }
    }

    std::string toString(WeightsType wType)
    {
        switch (wType)
        {
        case WeightsType::F16:  return "F16";
        case WeightsType::F32:  return "F32";
        case WeightsType::INT8: return "INT8";
        default: return "";
        }
    }

    std::string toString(KernelType kt)
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
        case KernelType::SELECT:            return "SELECT";
        default:
            return "";
        }
    }

    std::string toString(EltwiseMode b_mode)
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

    std::string toString(ReorderMode mode)
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

    std::string toString(MeanSubtractMode mode)
    {
        switch (mode)
        {
        case MeanSubtractMode::NONE:            return "NONE";
        case MeanSubtractMode::INSIDE_PARAMS:   return "INSIDE_PARAMS";
        case MeanSubtractMode::IN_BUFFER:       return "IN_BUFFER";
        default: return "";
        }
    }

    std::string toString(ArgMaxMinOut mode)
    {
        switch (mode)
        {
        case ArgMaxMinOut::MAX: return "MAX";
        case ArgMaxMinOut::MIN: return "MIN";
        default: return "";
        }
    }

    std::string toString(ArgMaxMinAxis mode)
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

    std::string toString(LookUpTableAxis mode)
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

    std::string toString(PoolType mode)
    {
        switch (mode)
        {
        case PoolType::MAX: return "MAX";
        case PoolType::AVG: return "AVG";
        case PoolType::MAX_WITH_ARGMAX: return "MAX_WITH_ARGMAX";
        case PoolType::BILINEAR: return "BILINEAR";
        default: return "";
        }
    }

    std::string toString(LRNMode mode)
    {
        switch (mode)
        {
        case LRNMode::ACROSS_CHANNEL: return "ACROSS_CHANNEL";
        case LRNMode::WITHIN_CHANNEL: return "WITHIN_CHANNEL";
        default: return "";
        }
    }

    std::string toString(KernelDividerMode mode)
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

    std::string toString(SoftmaxDim d)
    {
        switch (d)
        {
        case SoftmaxDim::X:         return "X";
        case SoftmaxDim::Y:         return "Y";
        case SoftmaxDim::FEATURE:   return "FEATURE";
        default: return "";
        }
    }

    std::string toString(NormalizeMode mode)
    {
        switch (mode)
        {
        case NormalizeMode::ACROSS_SPATIAL:         return "ACROSS_SPATIAL";
        case NormalizeMode::WITHIN_SPATIAL:         return "WITHIN_SPATIAL";
        default: return "";
        }
    }

    std::string toString(MVNMode mode)
    {
        switch (mode)
        {
        case MVNMode::ACROSS_CHANNELS:         return "ACROSS_CHANNELS";
        case MVNMode::WITHIN_CHANNELS:         return "WITHIN_CHANNELS";
        default: return "";
        }
    }

    std::string toString(WeightsLayout layout)
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
        case WeightsLayout::is_o_yx_isv32: return "IS_O_YX_ISV32";
        default:
            return "";
            break;
        }
    }

    std::string toString(ConcatAxis a)
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

    std::string toString(TileAxis a)
    {
        switch (a)
        {
        case TileAxis::X:         return "X";
        case TileAxis::Y:         return "Y";
        case TileAxis::FEATURE:   return "FEATURE";
        case TileAxis::BATCH:     return "BATCH";
        default: return "";
        }
    }

    std::string toString(SampleType type)
    {
        switch (type)
        {
        case SampleType::NEAREST: return "SAMPLE_TYPE_NEAREST";
        case SampleType::BILINEAR: return "SAMPLE_TYPE_BILINEAR";
        default: return "";
        }
    }

    std::string toString(const BorderType type)
    {
        switch (type)
        {
        case BorderType::ZERO:       return "BORDER_TYPE_ZERO";
        case BorderType::MIRROR:     return "BORDER_TYPE_MIRROR";
        case BorderType::MIRROR_101: return "BORDER_TYPE_MIRROR_101";
        default:                     return "";
        }
    }

    std::string toString(const IndexSelectAxis& axis)
    {
        switch (axis)
        {
        case IndexSelectAxis::BATCH:       return "INDEX_SELECT_AXIS_BATCH";
        case IndexSelectAxis::FEATURE:     return "INDEX_SELECT_AXIS_FEATURE";
        case IndexSelectAxis::X:           return "INDEX_SELECT_AXIS_X";
        case IndexSelectAxis::Y:           return "INDEX_SELECT_AXIS_Y";
        default:                           return "";
        }
    }

    std::string toString(NonLinearParams params)
    {
        std::stringstream s;
        s << "m" << params.m << "_n" << params.n;
        return s.str();
    }

    std::string toString(const Tensor::Dim& dim)
    {
        std::stringstream s;
        s << "v" << dim.v << "_p" << dim.pitch << "_" << dim.pad.before << "_" << dim.pad.after;
        return s.str();
    }

    std::string toString(const DataTensor& tensor)
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
