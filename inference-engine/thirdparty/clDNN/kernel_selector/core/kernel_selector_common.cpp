// Copyright (c) 2016-2019 Intel Corporation
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


#include "kernel_selector_common.h"
#include <sstream>
#include <string>

namespace kernel_selector {
std::string GetStringEnv(const char* varName) {
    std::string str;
#ifdef _WIN32
    char* env = nullptr;
    size_t len = 0;
    errno_t err = _dupenv_s(&env, &len, varName);
    if (err == 0) {
        if (env != nullptr) {
            str = std::string(env);
        }
        free(env);
    }
#else
    const char* env = getenv(varName);
    if (env) {
        str = std::string(env);
    }
#endif

    return str;
}

std::string toString(ActivationFunction activation) {
    std::string method("LINEAR");
    switch (activation) {
        case ActivationFunction::LOGISTIC:                 method = "LOGISTIC"; break;
        case ActivationFunction::HYPERBOLIC_TAN:           method = "HYPERBOLIC_TAN"; break;
        case ActivationFunction::RELU:                     method = "RELU"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:      method = "RELU_NEGATIVE_SLOPE"; break;
        case ActivationFunction::CLAMP:                    method = "CLAMP"; break;
        case ActivationFunction::SOFTRELU:                 method = "SOFTRELU"; break;
        case ActivationFunction::ABS:                      method = "ABS"; break;
        case ActivationFunction::SQUARE:                   method = "SQUARE"; break;
        case ActivationFunction::SQRT:                     method = "SQRT"; break;
        case ActivationFunction::LINEAR:                   method = "LINEAR"; break;
        case ActivationFunction::ELU:                      method = "ELU"; break;
        case ActivationFunction::RELU_GRAD:                method = "RELU_GRAD"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD: method = "RELU_NEGATIVE_SLOPE_GRAD"; break;
        case ActivationFunction::SIN:                      method = "SIN"; break;
        case ActivationFunction::ASIN:                     method = "ASIN"; break;
        case ActivationFunction::SINH:                     method = "SINH"; break;
        case ActivationFunction::ASINH:                    method = "ASINH"; break;
        case ActivationFunction::COS:                      method = "COS"; break;
        case ActivationFunction::ACOS:                     method = "ACOS"; break;
        case ActivationFunction::COSH:                     method = "COSH"; break;
        case ActivationFunction::ACOSH:                    method = "ACOSH"; break;
        case ActivationFunction::LOG:                      method = "LOG"; break;
        case ActivationFunction::LOG2:                     method = "LOG2"; break;
        case ActivationFunction::EXP:                      method = "EXP"; break;
        case ActivationFunction::NOT:                      method = "NOT"; break;
        case ActivationFunction::POW:                      method = "POW"; break;
        case ActivationFunction::NONE:                     method = "NONE"; break;
        case ActivationFunction::NONE_GRAD:                method = "NONE_GRAD"; break;
        case ActivationFunction::TAN:                      method = "TAN"; break;
        case ActivationFunction::ATAN:                     method = "ATAN"; break;
        case ActivationFunction::ATANH:                    method = "ATANH"; break;
        case ActivationFunction::FLOOR:                    method = "FLOOR"; break;
        case ActivationFunction::CEIL:                     method = "CEIL"; break;
        case ActivationFunction::NEGATIVE:                 method = "NEGATIVE"; break;
        case ActivationFunction::ERF:                      method = "ERF"; break;
        case ActivationFunction::HARD_SIGMOID:             method = "HARD_SIGMOID"; break;
        case ActivationFunction::RECIPROCAL:               method = "RECIPROCAL"; break;
        case ActivationFunction::SELU:                     method = "SELU"; break;
        case ActivationFunction::SIGN:                     method = "SIGN"; break;
        case ActivationFunction::SOFTPLUS:                 method = "SOFTPLUS"; break;
        case ActivationFunction::SOFTSIGN:                 method = "SOFTSIGN"; break;
        default: break;
    }
    return method;
}

std::string toString(DataLayout l) {
    switch (l) {
        case kernel_selector::DataLayout::bf:                   return "BF";
        case kernel_selector::DataLayout::fb:                   return "FB";
        case kernel_selector::DataLayout::bfyx:                 return "BFYX";
        case kernel_selector::DataLayout::yxfb:                 return "YXFB";
        case kernel_selector::DataLayout::byxf:                 return "BYXF";
        case kernel_selector::DataLayout::fyxb:                 return "FYXB";
        case kernel_selector::DataLayout::bfyx_f16:             return "BFYX_F16";
        case kernel_selector::DataLayout::bs_f_bsv8__af8:       return "BS_F_BSV8__AF8";
        case kernel_selector::DataLayout::bs_f_bsv16__af8:      return "BS_F_BSV16__AF8";
        case kernel_selector::DataLayout::bf8_xy16:             return "BF8_XY16";
        case kernel_selector::DataLayout::winograd_2x3_s1_data: return "WINOGRAD_2x3_S1_DATA";
        case kernel_selector::DataLayout::byxf_af32:            return "BYXF_AF32";
        case kernel_selector::DataLayout::byx8_f4:              return "BYX8_F4";
        case kernel_selector::DataLayout::fs_bs_yx_bsv4_fsv32:  return "FS_BS_YX_BSV4_FSV32";
        case kernel_selector::DataLayout::b_fs_yx_fsv4:         return "B_FS_YX_FSV4";
        case kernel_selector::DataLayout::b_fs_yx_32fp:         return "B_FS_YX_32FP";
        case kernel_selector::DataLayout::bfzyx:                return "BFZYX";
        case kernel_selector::DataLayout::fs_b_yx_fsv32:        return "FS_B_YX_FSV32";
        case kernel_selector::DataLayout::bfwzyx:               return "BFWZYX";
        case kernel_selector::DataLayout::bfzyx_f16:            return "BFZYX_F16";
        default:
            return "";
    }
}

std::string toString(Datatype dType) {
    switch (dType) {
        case Datatype::BINARY: return "BINARY";
        case Datatype::INT8:   return "INT8";
        case Datatype::UINT8:  return "UINT8";
        case Datatype::INT16:  return "INT16";
        case Datatype::UINT16: return "UINT16";
        case Datatype::INT32:  return "INT32";
        case Datatype::UINT32: return "UINT32";
        case Datatype::INT64:  return "INT64";
        case Datatype::F16:    return "F16";
        case Datatype::F32:    return "F32";
        default: return "";
    }
}

std::string toString(WeightsType wType) {
    switch (wType) {
        case WeightsType::BINARY: return "BINARY";
        case WeightsType::F16:    return "F16";
        case WeightsType::F32:    return "F32";
        case WeightsType::INT8:   return "INT8";
        default: return "";
    }
}

std::string toString(KernelType kt) {
    switch (kt) {
        case KernelType::UNKNOWN:         return "UNKNOWN";
        case KernelType::CONVOLUTION:     return "CONVOLUTION";
        case KernelType::LRN:             return "LRN";
        case KernelType::POOLING:         return "POOLING";
        case KernelType::ROI_POOLING:     return "ROI_POOLING";
        case KernelType::FULLY_CONNECTED: return "FULLY_CONNECTED";
        case KernelType::ACTIVATION:      return "ACTIVATION";
        case KernelType::SOFT_MAX:        return "SOFT_MAX";
        case KernelType::REGION_YOLO:     return "REGION_YOLO";
        case KernelType::REORG_YOLO:      return "REORG_YOLO";
        case KernelType::ELTWISE:         return "ELTWISE";
        case KernelType::REORDER:         return "REORDER";
        case KernelType::SELECT:          return "SELECT";
        default: return "";
    }
}

std::string toString(EltwiseMode b_mode) {
    switch (b_mode) {
        case EltwiseMode::ADD:    return "ADD";
        case EltwiseMode::SUB:    return "SUB";
        case EltwiseMode::MUL:    return "MUL";
        case EltwiseMode::DIV:    return "DIV";
        case EltwiseMode::MIN:    return "MIN";
        case EltwiseMode::MAX:    return "MAX";
        case EltwiseMode::POW:    return "POW";
        case EltwiseMode::MODULU: return "MODULU";
        case EltwiseMode::SQRT:   return "SQRT";
        case EltwiseMode::RSQRT:  return "RSQRT";
        case EltwiseMode::ASSIGN: return "ASSIGN";
        default: return "";
    }
}

std::string toString(ReorderMode mode) {
    switch (mode) {
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

std::string toString(MeanSubtractMode mode) {
    switch (mode) {
        case MeanSubtractMode::NONE:          return "NONE";
        case MeanSubtractMode::INSIDE_PARAMS: return "INSIDE_PARAMS";
        case MeanSubtractMode::IN_BUFFER:     return "IN_BUFFER";
        default: return "";
    }
}

std::string toString(ArgMaxMinOut mode) {
    switch (mode) {
        case ArgMaxMinOut::MAX: return "MAX";
        case ArgMaxMinOut::MIN: return "MIN";
        default: return "";
    }
}

std::string toString(ArgMaxMinAxis mode) {
    switch (mode) {
        case ArgMaxMinAxis::BATCH:   return "BATCH";
        case ArgMaxMinAxis::FEATURE: return "FEATURE";
        case ArgMaxMinAxis::X:       return "X";
        case ArgMaxMinAxis::Y:       return "Y";
        case ArgMaxMinAxis::Z:       return "Z";
        case ArgMaxMinAxis::XYF:     return "XYF";
        default: return "";
    }
}

std::string toString(LookUpTableAxis mode) {
    switch (mode) {
        case LookUpTableAxis::BATCH:   return "BATCH";
        case LookUpTableAxis::FEATURE: return "FEATURE";
        case LookUpTableAxis::X:       return "X";
        case LookUpTableAxis::Y:       return "Y";
        case LookUpTableAxis::XYF:     return "XYF";
        default: return "";
    }
}

std::string toString(PoolType mode) {
    switch (mode) {
        case PoolType::MAX:                 return "MAX";
        case PoolType::AVG:                 return "AVG";
        case PoolType::MAX_WITH_ARGMAX:     return "MAX_WITH_ARGMAX";
        case PoolType::BILINEAR:            return "BILINEAR";
        case PoolType::DEFORMABLE_BILINEAR: return "DEFORMABLE_BILINEAR";
        default: return "";
    }
}

std::string toString(LRNMode mode) {
    switch (mode) {
        case LRNMode::ACROSS_CHANNEL: return "ACROSS_CHANNEL";
        case LRNMode::WITHIN_CHANNEL: return "WITHIN_CHANNEL";
        default: return "";
    }
}

std::string toString(KernelDividerMode mode) {
    switch (mode) {
        case KernelDividerMode::DONT_CARE:            return "DONT_CARE";
        case KernelDividerMode::FIXED:                return "FIXED";
        case KernelDividerMode::DYNAMIC:              return "DYNAMIC";
        case KernelDividerMode::DYNAMIC_WITH_PADDING: return "DYNAMIC_WITH_PADDING";
        default: return "";
    }
}

std::string toString(SoftmaxDim d) {
    switch (d) {
        case SoftmaxDim::X:       return "X";
        case SoftmaxDim::Y:       return "Y";
        case SoftmaxDim::FEATURE: return "FEATURE";
        default: return "";
    }
}

std::string toString(NormalizeMode mode) {
    switch (mode) {
        case NormalizeMode::ACROSS_SPATIAL: return "ACROSS_SPATIAL";
        case NormalizeMode::WITHIN_SPATIAL: return "WITHIN_SPATIAL";
        default: return "";
    }
}

std::string toString(MVNMode mode) {
    switch (mode) {
        case MVNMode::ACROSS_CHANNELS: return "ACROSS_CHANNELS";
        case MVNMode::WITHIN_CHANNELS: return "WITHIN_CHANNELS";
        default: return "";
    }
}

std::string toString(WeightsLayout layout) {
    switch (layout) {
        case WeightsLayout::oi:                                     return "OI";
        case WeightsLayout::io:                                     return "IO";
        case WeightsLayout::oiyx:                                   return "OIYX";
        case WeightsLayout::oyxi:                                   return "OYXI";
        case WeightsLayout::iyxo:                                   return "IYXO";
        case WeightsLayout::yxio:                                   return "YXIO";
        case WeightsLayout::oiyx_o16:                               return "OIYX_O16";
        case WeightsLayout::o_i_yx_i16_o16:                         return "O_I_YX_I16_O16";
        case WeightsLayout::os_iyx_osv16:                           return "OS_IYX_OSV16";
        case WeightsLayout::os_iyx_osv32:                           return "OS_IYX_OSV32";
        case WeightsLayout::os_iyx_osv64:                           return "OS_IYX_OSV64";
        case WeightsLayout::os_iyx_osv16_rotate_180:                return "OS_IYX_OSV16_ROTATE_180";
        case WeightsLayout::os_i_osv16:                             return "OS_I_OSV16";
        case WeightsLayout::os_i_osv8__ai8:                         return "OS_I_OSV8__AI8";
        case WeightsLayout::os_i_osv16__ai8:                        return "OS_I_OSV16__AI8";
        case WeightsLayout::i_yxs_os_yxsv2_osv16:                   return "I_YXS_OS_YXSV2_OSV16";
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:              return "IY_XS_OS_XSV2_OSV16__AO32";
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:               return "IY_XS_OS_XSV2_OSV8__AO32";
        case WeightsLayout::image_2d_weights_c4_fyx_b:              return "IMAGE_2D_WEIGHTS_C4_FYX_B";
        case WeightsLayout::image_2d_weights_c1_b_fyx:              return "IMAGE_2D_WEIGHTS_C1_B_FYX";
        case WeightsLayout::winograd_2x3_s1_weights:                return "WINOGRAD_2x3_S1_WEIGHTS";
        case WeightsLayout::winograd_2x3_s1_fused_weights:          return "WINOGRAD_2x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::winograd_6x3_s1_fused_weights:          return "WINOGRAD_6x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb: return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_FBXYB";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb: return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_XFBYB";
        case WeightsLayout::dlstm_dir_io:                           return "DLSTM_DIR_IO";
        case WeightsLayout::os_is_yx_isa8_osv8_isv4:                return "OS_IS_YX_ISA8_OSV8_ISV4";
        case WeightsLayout::os_is_yx_isa8_osv8_isv4_swizzled_by_4:  return "OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4";
        case WeightsLayout::is_o_yx_isv32:                          return "IS_O_YX_ISV32";
        case WeightsLayout::is_o32_yx_isv32_swizzled_by_4:          return "IS_O32_YX_ISV32_SWIZZLED_BY_4";
        case WeightsLayout::os_is_y_x8_osv8_isv4:                   return "OS_IS_Y_X8_OSV8_ISV4";
        case WeightsLayout::os_is_yx_osv16_isv4:                    return "OS_IS_YX_OSV16_ISV4";
        case WeightsLayout::os_is_y_x8_osv8_isv4_swizzled_by_4:     return "OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4";
        case WeightsLayout::os_is_yx_osv32_isv32p:                  return "OS_IS_YX_OSV32_ISV32P";
        case WeightsLayout::oizyx:                                  return "OIZYX";
        case WeightsLayout::bf_lyx_yx:                              return "BF_LYX_YX";
        case WeightsLayout::o_i_zyx_i16_o16:                        return "O_I_ZYX_I16_O16";
        case WeightsLayout::i_o_zyx_o16_i16:                        return "I_O_ZYX_O16_I16";
        case WeightsLayout::os_is_osv32_isv32_swizzled_by_4:        return "OS_IS_OSV32_ISV32_SWIZZLED_BY_4";
        default: throw std::invalid_argument("Failed to convert WeightsLayout " + std::to_string(layout) + " to string");
    }
}

std::string toString(ConcatAxis a) {
    switch (a) {
        case ConcatAxis::X:       return "X";
        case ConcatAxis::Y:       return "Y";
        case ConcatAxis::Z:       return "Z";
        case ConcatAxis::W:       return "W";
        case ConcatAxis::FEATURE: return "FEATURE";
        case ConcatAxis::BATCH:   return "BATCH";
        default: return "";
    }
}

std::string toString(TileAxis a) {
    switch (a) {
        case TileAxis::X:       return "X";
        case TileAxis::Y:       return "Y";
        case TileAxis::FEATURE: return "FEATURE";
        case TileAxis::BATCH:   return "BATCH";
        default: return "";
    }
}

std::string toString(GatherAxis a) {
    switch (a) {
        case GatherAxis::X:       return "X";
        case GatherAxis::Y:       return "Y";
        case GatherAxis::FEATURE: return "FEATURE";
        case GatherAxis::BATCH:   return "BATCH";
        default: return "";
    }
}

std::string toString(SampleType type) {
    switch (type) {
        case SampleType::NEAREST:  return "SAMPLE_TYPE_NEAREST";
        case SampleType::BILINEAR: return "SAMPLE_TYPE_BILINEAR";
        default: return "";
    }
}

std::string toString(const BorderType type) {
    switch (type) {
        case BorderType::CONSTANT:   return "BORDER_TYPE_CONSTANT";
        case BorderType::EDGE:       return "BORDER_TYPE_EDGE";
        case BorderType::MIRROR:     return "BORDER_TYPE_MIRROR";
        case BorderType::MIRROR_101: return "BORDER_TYPE_MIRROR_101";
        default:                     return "";
    }
}

std::string toString(const IndexSelectAxis& axis) {
    switch (axis) {
        case IndexSelectAxis::BATCH:   return "INDEX_SELECT_AXIS_BATCH";
        case IndexSelectAxis::FEATURE: return "INDEX_SELECT_AXIS_FEATURE";
        case IndexSelectAxis::X:       return "INDEX_SELECT_AXIS_X";
        case IndexSelectAxis::Y:       return "INDEX_SELECT_AXIS_Y";
        default:                       return "";
    }
}

std::string toString(const Tensor::Dim& dim) {
    std::stringstream s;
    s << "v" << dim.v << "_p" << dim.pitch << "_" << dim.pad.before << "_" << dim.pad.after;
    return s.str();
}

template <typename DType, typename Layout>
std::string toStringTensor(const Tensor::TensorBaseT<DType, Layout>& tensor) {
    std::stringstream s;
    s << toString(tensor.GetDType()) << "_";
    s << toString(tensor.GetLayout()) << "_";
    int i = 0;
    for (auto dim : tensor.GetDims()) {
        s << "d" << i << "_" << toString(dim) << "_";
        i++;
    }
    return s.str();
}

std::string toString(const DataTensor& tensor) {
    return toStringTensor(tensor);
}

std::string toString(const WeightsTensor& tensor) {
    return toStringTensor(tensor);
}

std::string toString(ReduceMode mode) {
    switch (mode) {
        case ReduceMode::MAX:
            return "MAX";
        case ReduceMode::MIN:
            return "MIN";
        case ReduceMode::MEAN:
            return "MEAN";
        case ReduceMode::PROD:
            return "PROD";
        case ReduceMode::SUM:
            return "SUM";
        case ReduceMode::AND:
            return "AND";
        case ReduceMode::OR:
            return "OR";
        case ReduceMode::SUM_SQUARE:
            return "SUM_SQUARE";
        case ReduceMode::L1:
            return "L1";
        case ReduceMode::L2:
            return "L2";
        case ReduceMode::LOG_SUM:
            return "LOG_SUM";
        case ReduceMode::LOG_SUM_EXP:
            return "LOG_SUM_EXP";
        default:
            return "";
    }
}

}  // namespace kernel_selector
