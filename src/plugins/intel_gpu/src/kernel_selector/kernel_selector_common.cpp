// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_common.h"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include <sstream>
#include <string>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "micro_utils.hpp"
#endif

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
        case ActivationFunction::LOGISTIC:                  method = "LOGISTIC"; break;
        case ActivationFunction::HYPERBOLIC_TAN:            method = "HYPERBOLIC_TAN"; break;
        case ActivationFunction::RELU:                      method = "RELU"; break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:       method = "RELU_NEGATIVE_SLOPE"; break;
        case ActivationFunction::CLAMP:                     method = "CLAMP"; break;
        case ActivationFunction::SOFTRELU:                  method = "SOFTRELU"; break;
        case ActivationFunction::ABS:                       method = "ABS"; break;
        case ActivationFunction::SQUARE:                    method = "SQUARE"; break;
        case ActivationFunction::SQRT:                      method = "SQRT"; break;
        case ActivationFunction::LINEAR:                    method = "LINEAR"; break;
        case ActivationFunction::ELU:                       method = "ELU"; break;
        case ActivationFunction::SIN:                       method = "SIN"; break;
        case ActivationFunction::ASIN:                      method = "ASIN"; break;
        case ActivationFunction::SINH:                      method = "SINH"; break;
        case ActivationFunction::ASINH:                     method = "ASINH"; break;
        case ActivationFunction::COS:                       method = "COS"; break;
        case ActivationFunction::ACOS:                      method = "ACOS"; break;
        case ActivationFunction::COSH:                      method = "COSH"; break;
        case ActivationFunction::ACOSH:                     method = "ACOSH"; break;
        case ActivationFunction::LOG:                       method = "LOG"; break;
        case ActivationFunction::LOG2:                      method = "LOG2"; break;
        case ActivationFunction::EXP:                       method = "EXP"; break;
        case ActivationFunction::NOT:                       method = "NOT"; break;
        case ActivationFunction::POW:                       method = "POW"; break;
        case ActivationFunction::NONE:                      method = "NONE"; break;
        case ActivationFunction::TAN:                       method = "TAN"; break;
        case ActivationFunction::ATAN:                      method = "ATAN"; break;
        case ActivationFunction::ATANH:                     method = "ATANH"; break;
        case ActivationFunction::FLOOR:                     method = "FLOOR"; break;
        case ActivationFunction::CEIL:                      method = "CEIL"; break;
        case ActivationFunction::NEGATIVE:                  method = "NEGATIVE"; break;
        case ActivationFunction::ERF:                       method = "ERF"; break;
        case ActivationFunction::HARD_SIGMOID:              method = "HARD_SIGMOID"; break;
        case ActivationFunction::HSIGMOID:                  method = "HSIGMOID"; break;
        case ActivationFunction::RECIPROCAL:                method = "RECIPROCAL"; break;
        case ActivationFunction::SELU:                      method = "SELU"; break;
        case ActivationFunction::SIGN:                      method = "SIGN"; break;
        case ActivationFunction::SOFTPLUS:                  method = "SOFTPLUS"; break;
        case ActivationFunction::SOFTSIGN:                  method = "SOFTSIGN"; break;
        case ActivationFunction::SWISH:                     method = "SWISH"; break;
        case ActivationFunction::HSWISH:                    method = "HSWISH"; break;
        case ActivationFunction::MISH:                      method = "MISH"; break;
        case ActivationFunction::GELU:                      method = "GELU"; break;
        case ActivationFunction::GELU_TANH:                 method = "GELU_TANH"; break;
        case ActivationFunction::ROUND_HALF_TO_EVEN:        method = "ROUND_HALF_TO_EVEN"; break;
        case ActivationFunction::ROUND_HALF_AWAY_FROM_ZERO: method = "ROUND_HALF_AWAY_FROM_ZERO"; break;
        default: break;
    }
    return method;
}

std::string toString(DataLayout l) {
    switch (l) {
        case kernel_selector::DataLayout::bf:                    return "BF";
        case kernel_selector::DataLayout::fb:                    return "FB";
        case kernel_selector::DataLayout::bfyx:                  return "BFYX";
        case kernel_selector::DataLayout::yxfb:                  return "YXFB";
        case kernel_selector::DataLayout::byxf:                  return "BYXF";
        case kernel_selector::DataLayout::byfx:                  return "BYFX";
        case kernel_selector::DataLayout::bxfy:                  return "BXFY";
        case kernel_selector::DataLayout::fbyx:                  return "FBYX";
        case kernel_selector::DataLayout::fyxb:                  return "FYXB";
        case kernel_selector::DataLayout::b_fs_yx_fsv2:          return "B_FS_YX_FSV2";
        case kernel_selector::DataLayout::b_fs_yx_fsv4:          return "B_FS_YX_FSV4";
        case kernel_selector::DataLayout::b_fs_yx_fsv8:          return "B_FS_YX_FSV8";
        case kernel_selector::DataLayout::b_fs_yx_fsv16:         return "B_FS_YX_FSV16";
        case kernel_selector::DataLayout::b_fs_yx_fsv32:         return "B_FS_YX_FSV32";
        case kernel_selector::DataLayout::b_fs_zyx_fsv32:        return "B_FS_ZYX_FSV32";
        case kernel_selector::DataLayout::bs_f_bsv8__af8:        return "BS_F_BSV8__AF8";
        case kernel_selector::DataLayout::bs_f_bsv16__af8:       return "BS_F_BSV16__AF8";
        case kernel_selector::DataLayout::winograd_2x3_s1_data:  return "WINOGRAD_2x3_S1_DATA";
        case kernel_selector::DataLayout::bfzyx:                 return "BFZYX";
        case kernel_selector::DataLayout::bzyxf:                 return "BZYXF";
        case kernel_selector::DataLayout::fs_b_yx_fsv32:         return "FS_B_YX_FSV32";
        case kernel_selector::DataLayout::bfwzyx:                return "BFWZYX";
        case kernel_selector::DataLayout::bfuwzyx:               return "BFUWZYX";
        case kernel_selector::DataLayout::bfvuwzyx:              return "BFVUWZYX";
        case kernel_selector::DataLayout::b_fs_zyx_fsv8:         return "B_FS_ZYX_FSV8";
        case kernel_selector::DataLayout::b_fs_zyx_fsv16:        return "B_FS_ZYX_FSV16";
        case kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv16:  return "BS_FS_YX_BSV16_FSV16";
        case kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv32:  return "BS_FS_YX_BSV16_FSV32";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv32: return "BS_FS_ZYX_BSV16_FSV32";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv16: return "BS_FS_ZYX_BSV16_FSV16";
        case kernel_selector::DataLayout::bs_fs_yx_bsv4_fsv4:    return "BS_FS_YX_BSV4_FSV4";
        case kernel_selector::DataLayout::bs_fs_yx_bsv8_fsv4:    return "BS_FS_YX_BSV8_FSV4";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv8_fsv4:   return "BS_FS_ZYX_BSV8_FSV4";
        case kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv8:   return "BS_FS_YX_BSV16_FSV8";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv8:  return "BS_FS_ZYX_BSV16_FSV8";
        case kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv4:   return "BS_FS_YX_BSV16_FSV4";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv4:  return "BS_FS_ZYX_BSV16_FSV4";
        case kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv2:   return "BS_FS_YX_BSV16_FSV2";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv2:  return "BS_FS_ZYX_BSV16_FSV2";
        case kernel_selector::DataLayout::bs_fs_yx_bsv8_fsv2:    return "BS_FS_YX_BSV8_FSV2";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv8_fsv2:   return "BS_FS_ZYX_BSV8_FSV2";
        case kernel_selector::DataLayout::bs_fs_yx_bsv4_fsv2:    return "BS_FS_YX_BSV4_FSV2";
        case kernel_selector::DataLayout::bs_fs_yx_bsv32_fsv32:  return "BS_FS_YX_BSV32_FSV32";
        case kernel_selector::DataLayout::bs_fs_yx_bsv32_fsv16:  return "BS_FS_YX_BSV32_FSV16";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv32_fsv32: return "BS_FS_ZYX_BSV32_FSV32";
        case kernel_selector::DataLayout::bs_fs_zyx_bsv32_fsv16: return "BS_FS_ZYX_BSV32_FSV16";
        case kernel_selector::DataLayout::nv12:                  return "NV12";
        case kernel_selector::DataLayout::image_2d_rgba:         return "IMAGE_2D_RGBA";
        default:
            return std::to_string(l);
    }
}

std::string toString(Datatype dType) {
    switch (dType) {
        case Datatype::UINT4:  return "UINT4";
        case Datatype::INT4:   return "INT4";
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
        case WeightsType::F16:    return "F16";
        case WeightsType::F32:    return "F32";
        case WeightsType::UINT4:  return "UINT4";
        case WeightsType::INT4:   return "INT4";
        case WeightsType::INT8:   return "INT8";
        case WeightsType::UINT8:  return "UINT8";
        case WeightsType::INT32:  return "INT32";
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
        case KernelType::NON_MAX_SUPPRESSION:
                                          return "NON_MAX_SUPPRESSION";
        case KernelType::MATRIX_NMS:      return "MATRIX_NMS";
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
        default: return "";
    }
}

std::string toString(PoolType mode) {
    switch (mode) {
        case PoolType::MAX:                 return "MAX";
        case PoolType::AVG:                 return "AVG";
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
        case SoftmaxDim::Z:       return "Z";
        case SoftmaxDim::FEATURE: return "FEATURE";
        case SoftmaxDim::BATCH:   return "BATCH";
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

std::string toString(MVNEpsMode mode) {
    switch (mode) {
        case MVNEpsMode::INSIDE_SQRT : return "INSIDE_SQRT";
        case MVNEpsMode::OUTSIDE_SQRT : return "OUTSIDE_SQRT";
        default: return "";
    }
}

std::string toString(WeightsLayout layout) {
   switch (layout) {
        case WeightsLayout::oi:                                          return "OI";
        case WeightsLayout::io:                                          return "IO";
        case WeightsLayout::oiyx:                                        return "OIYX";
        case WeightsLayout::ioyx:                                        return "IOYX";
        case WeightsLayout::oyxi:                                        return "OYXI";
        case WeightsLayout::oyix:                                        return "OYIX";
        case WeightsLayout::oxiy:                                        return "OXIY";
        case WeightsLayout::iyxo:                                        return "IYXO";
        case WeightsLayout::yxio:                                        return "YXIO";
        case WeightsLayout::os_is_yx_isv16_osv16:                        return "OS_IS_YX_ISV16_OSV16";
        case WeightsLayout::os_is_yx_osv16_isv16:                        return "OS_IS_YX_OSV16_ISV16";
        case WeightsLayout::os_is_zyx_osv16_isv16:                       return "OS_IS_ZYX_OSV16_ISV16";
        case WeightsLayout::os_is_zyx_osv32_isv16:                       return "OS_IS_ZYX_OSV32_ISV16";
        case WeightsLayout::os_is_zyx_osv64_isv16:                       return "OS_IS_ZYX_OSV64_ISV16";
        case WeightsLayout::o_is_yx_isv4:                                return "O_IS_YX_ISV4";
        case WeightsLayout::o_is_yx_isv16:                               return "O_IS_YX_ISV16";
        case WeightsLayout::os_iyx_osv16:                                return "OS_IYX_OSV16";
        case WeightsLayout::os_iyx_osv32:                                return "OS_IYX_OSV32";
        case WeightsLayout::os_iyx_osv8:                                 return "OS_IYX_OSV8";
        case WeightsLayout::os_iyx_osv32__ai32:                          return "OS_IYX_OSV32__AI32";
        case WeightsLayout::os_iyx_osv64:                                return "OS_IYX_OSV64";
        case WeightsLayout::os_iyx_osv16_rotate_180:                     return "OS_IYX_OSV16_ROTATE_180";
        case WeightsLayout::g_os_iyx_osv16_rotate_180:                   return "G_OS_IYX_OSV16_ROTATE_180";
        case WeightsLayout::os_i_osv16:                                  return "OS_I_OSV16";
        case WeightsLayout::os_i_osv8__ai8:                              return "OS_I_OSV8__AI8";
        case WeightsLayout::os_i_osv16__ai8:                             return "OS_I_OSV16__AI8";
        case WeightsLayout::i_yxs_os_yxsv2_osv16:                        return "I_YXS_OS_YXSV2_OSV16";
        case WeightsLayout::iy_xs_os_xsv2_osv16__ao32:                   return "IY_XS_OS_XSV2_OSV16__AO32";
        case WeightsLayout::iy_xs_os_xsv2_osv8__ao32:                    return "IY_XS_OS_XSV2_OSV8__AO32";
        case WeightsLayout::image_2d_weights_c4_fyx_b:                   return "IMAGE_2D_WEIGHTS_C4_FYX_B";
        case WeightsLayout::image_2d_weights_c1_b_fyx:                   return "IMAGE_2D_WEIGHTS_C1_B_FYX";
        case WeightsLayout::winograd_2x3_s1_weights:                     return "WINOGRAD_2x3_S1_WEIGHTS";
        case WeightsLayout::winograd_2x3_s1_fused_weights:               return "WINOGRAD_2x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::winograd_6x3_s1_fused_weights:               return "WINOGRAD_6x3_S1_FUSED_WEIGHTS";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb:      return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_FBXYB";
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb:      return "IMAGE_2D_WEIGHTS_WINOGRAD_6x3_S1_XFBYB";
        case WeightsLayout::os_is_yx_isa8_osv8_isv4:                     return "OS_IS_YX_ISA8_OSV8_ISV4";
        case WeightsLayout::os_is_yx_isa8_osv16_isv4:                    return "OS_IS_YX_ISA8_OSV16_ISV4";
        case WeightsLayout::os_is_zyx_isa8_osv8_isv4:                    return "OS_IS_ZYX_ISA8_OSV8_ISV4";
        case WeightsLayout::os_is_zyx_isa8_osv16_isv4:                   return "OS_IS_ZYX_ISA8_OSV16_ISV4";
        case WeightsLayout::os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4:  return "OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4";
        case WeightsLayout::os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4: return "OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4";
        case WeightsLayout::os_is_yx_osv16_isv4:                         return "OS_IS_YX_OSV16_ISV4";
        case WeightsLayout::os_is_yx_osv32_isv4_swizzled_by_2:           return "OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2";
        case WeightsLayout::os_is_yx_osv32_isv4:                         return "OS_IS_YX_OSV32_ISV4";
        case WeightsLayout::os_is_yx_osv32_isv2:                         return "OS_IS_YX_OSV32_ISV2";
        case WeightsLayout::os_is_yx_osv64_isv2:                         return "OS_IS_YX_OSV64_ISV2";
        case WeightsLayout::os_is_zyx_osv32_isv4:                        return "OS_IS_ZYX_OSV32_ISV4";
        case WeightsLayout::oizyx:                                       return "OIZYX";
        case WeightsLayout::iozyx:                                       return "IOZYX";
        case WeightsLayout::os_is_zyx_isv16_osv16:                       return "OS_IS_ZYX_ISV16_OSV16";
        case WeightsLayout::is_os_zyx_isv16_osv16:                       return "IS_OS_ZYX_ISV16_OSV16";
        case WeightsLayout::is_os_yx_isv16_osv16:                        return "IS_OS_YX_ISV16_OSV16";
        case WeightsLayout::os_is_zyx_isv8_osv16_isv2:                   return "OS_IS_ZYX_ISV8_OSV16_ISV2";
        case WeightsLayout::os_zyxi_osv16:                               return "OS_ZYXI_OSV16";
        case WeightsLayout::os_is_yx_isv8_osv16_isv2:                    return "OS_IS_YX_ISV8_OSV16_ISV2";
        case WeightsLayout::os_is_yx_osv8_isv4:                          return "OS_IS_YX_OSV8_ISV4";
        case WeightsLayout::goiyx:                                       return "GOIYX";
        case WeightsLayout::gioyx:                                       return "GIOYX";
        case WeightsLayout::gyxio:                                       return "GYXIO";
        case WeightsLayout::goizyx:                                      return "GOIZYX";
        case WeightsLayout::giozyx:                                      return "GIOZYX";
        case WeightsLayout::g_os_iyx_osv8:                               return "G_OS_IYX_OSV8";
        case WeightsLayout::g_os_iyx_osv16:                              return "G_OS_IYX_OSV16";
        case WeightsLayout::g_os_iyx_osv32:                              return "G_OS_IYX_OSV32";
        case WeightsLayout::gs_oiyx_gsv16:                               return "GS_OIYX_GSV16";
        case WeightsLayout::gs_oizyx_gsv16:                              return "GS_OIZYX_GSV16";
        case WeightsLayout::gs_oiyx_gsv32:                               return "GS_OIYX_GSV32";
        case WeightsLayout::gi_yxs_os_yxsv2_osv16:                       return "GI_YXS_OS_YXSV2_OSV16";
        case WeightsLayout::g_is_os_zyx_isv16_osv16:                     return "G_IS_OS_ZYX_ISV16_OSV16";
        case WeightsLayout::g_is_os_yx_isv16_osv16:                      return "G_IS_OS_YX_ISV16_OSV16";
        case WeightsLayout::g_os_is_zyx_isv8_osv16_isv2:                 return "G_OS_IS_ZYX_ISV8_OSV16_ISV2";
        case WeightsLayout::g_os_is_yx_isv8_osv16_isv2:                  return "G_OS_IS_YX_ISV8_OSV16_ISV2";
        case WeightsLayout::g_os_is_zyx_isv16_osv16:                     return "G_OS_IS_ZYX_ISV16_OSV16";
        case WeightsLayout::giy_xs_os_xsv2_osv16__ao32:                  return "GIY_XS_OS_XSV2_OSV16__AO32";
        case WeightsLayout::giy_xs_os_xsv2_osv8__ao32:                   return "GIY_XS_OS_XSV2_OSV8__AO32";
        case WeightsLayout::gs_oi_yxs_gsv4_yxsv4:                        return "GS_OI_YXS_GSV4_YXSV4";
        case WeightsLayout::gs_oi_yxs_gsv16_yxsv4:                       return "GS_OI_YXS_GSV16_YXSV4";
        case WeightsLayout::gs_oi_yxs_gsv32_yxsv4:                       return "GS_OI_YXS_GSV32_YXSV4";
        case WeightsLayout::os_is_yx_osa4_isa8_osv8_isv4:                return "OS_IS_YX_OSA4_ISA8_OSV8_ISV4";
        case WeightsLayout::os_is_zyx_osa4_isa8_osv8_isv4:               return "OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4";
        case WeightsLayout::g_os_is_yx_isv16_osv16:                      return "G_OS_IS_YX_ISV16_OSV16";
        case WeightsLayout::g_os_is_yx_osv16_isv4:                       return "G_OS_IS_YX_OSV16_ISV4";
        case WeightsLayout::g_os_is_zyx_osv16_isv16:                     return "G_OS_IS_ZYX_OSV16_ISV16";
        case WeightsLayout::g_os_zyx_is_osv16_isv4:                      return "G_OS_ZYX_IS_OSV16_ISV4";
        case WeightsLayout::g_os_zyx_is_osv16_isv16:                     return "G_OS_ZYX_IS_OSV16_ISV16";
        case WeightsLayout::g_os_zyx_is_osv16_isv32:                     return "G_OS_ZYX_IS_OSV16_ISV32";
        case WeightsLayout::g_os_zyx_is_osv32_isv4:                      return "G_OS_ZYX_IS_OSV32_ISV4";
        case WeightsLayout::g_os_zyx_is_osv32_isv16:                     return "G_OS_ZYX_IS_OSV32_ISV16";
        case WeightsLayout::g_os_zyx_is_osv32_isv32:                     return "G_OS_ZYX_IS_OSV32_ISV32";

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

std::string toString(GatherAxis a) {
    switch (a) {
        case GatherAxis::X:       return "X";
        case GatherAxis::Y:       return "Y";
        case GatherAxis::Z:       return "Z";
        case GatherAxis::W:       return "W";
        case GatherAxis::FEATURE: return "FEATURE";
        case GatherAxis::BATCH:   return "BATCH";
        default: return "";
    }
}

std::string toString(ScatterUpdateAxis a) {
    switch (a) {
        case ScatterUpdateAxis::X:       return "X";
        case ScatterUpdateAxis::Y:       return "Y";
        case ScatterUpdateAxis::Z:       return "Z";
        case ScatterUpdateAxis::W:       return "W";
        case ScatterUpdateAxis::FEATURE: return "FEATURE";
        case ScatterUpdateAxis::BATCH:   return "BATCH";
        default: return "";
    }
}

std::string toString(ResampleType type) {
    switch (type) {
        case ResampleType::NEAREST_NEIGHBOR:  return "SAMPLE_TYPE_NEAREST";
        case ResampleType::BILINEAR_INTERP: return "SAMPLE_TYPE_INTERP";
        case ResampleType::CAFFE_BILINEAR_INTERP: return "SAMPLE_TYPE_CAFFE_INTERP";
        case ResampleType::CUBIC: return "SAMPLE_TYPE_CUBIC";
        case ResampleType::LINEAR_ONNX: return "SAMPLE_TYPE_LINEAR_ONNX";
        default: return "";
    }
}

std::string toString(CoordinateTransformationMode mode) {
    switch (mode) {
        case CoordinateTransformationMode::HALF_PIXEL:  return "COORD_TRANS_MODE_HALF_PIXEL";
        case CoordinateTransformationMode::PYTORCH_HALF_PIXEL: return "COORD_TRANS_MODE_PYTORCH_HALF_PIXEL";
        case CoordinateTransformationMode::ASYMMETRIC: return "COORD_TRANS_MODE_ASYMMETRIC";
        case CoordinateTransformationMode::TF_HALF_PIXEL_FOR_NN: return "COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN";
        case CoordinateTransformationMode::ALIGN_CORNERS: return "COORD_TRANS_MODE_ALIGN_CORNERS";
        default: return "";
    }
}

std::string toString(NearestMode mode) {
    switch (mode) {
        case NearestMode::ROUND_PREFER_FLOOR:  return "NEAREST_ROUND_PREFER_FLOOR";
        case NearestMode::ROUND_PREFER_CEIL: return "NEAREST_ROUND_PREFER_CEIL";
        case NearestMode::FLOOR: return "NEAREST_FLOOR";
        case NearestMode::CEIL: return "NEAREST_CEIL";
        case NearestMode::SIMPLE: return "NEAREST_SIMPLE";
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
    // WA to reuse old tuning cache. Code below must be replace with the following line once new cache file is merged.
    // return toStringTensor(tensor);
    if (tensor.GetLayout() != DataLayout::b_fs_yx_fsv16 &&
        tensor.GetLayout() != DataLayout::b_fs_zyx_fsv16) {
        return toStringTensor(tensor);
    } else {
        std::stringstream s;
        s << toString(tensor.GetDType()) << "_";
        std::string layoutStr;
        switch (tensor.GetLayout()) {
            case DataLayout::b_fs_yx_fsv16: layoutStr = "BFYX_F16"; break;
            case DataLayout::b_fs_zyx_fsv16: layoutStr = "BFZYX_F16"; break;
            default: layoutStr = toString(tensor.GetLayout()); break;
        }
        s << layoutStr << "_";
        int i = 0;
        for (auto dim : tensor.GetDims()) {
            s << "d" << i << "_" << toString(dim) << "_";
            i++;
        }
        return s.str();
    }
}

std::string toString(const WeightsTensor& tensor) {
    return toStringTensor(tensor);
}

std::string toString_v2(const DataTensor& tensor) {
    std::stringstream s;
    s << toString(tensor.GetDType()) << "_";
    s << toString(tensor.GetLayout());
    for (auto dim : tensor.GetDims()) {
        s << "_v" << dim.v << "_p" << dim.pad.before << "_" << dim.pad.after;
    }
    return s.str();
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

void clKernelData::save(cldnn::BinaryOutputBuffer& ob) const {
    ob(params.workGroups.global, params.workGroups.local);
    ob << params.arguments.size();
    for (const auto& arg : params.arguments) {
        ob << make_data(&arg.t, sizeof(cldnn::argument_desc::Types)) << arg.index;
    }
    ob << params.scalars.size();
    for (const auto& scalar : params.scalars) {
        ob << make_data(&scalar.t, sizeof(cldnn::scalar_desc::Types)) << make_data(&scalar.v, sizeof(cldnn::scalar_desc::ValueT));
    }
    ob << params.layerID;
#ifdef ENABLE_ONEDNN_FOR_GPU
    ob << micro_kernels.size();
    for (const auto& microkernel : micro_kernels) {
        microkernel->save(ob);
    }
#endif
}

void clKernelData::load(cldnn::BinaryInputBuffer& ib) {
    ib(params.workGroups.global, params.workGroups.local);

    typename cldnn::arguments_desc::size_type arguments_desc_size = 0UL;
    ib >> arguments_desc_size;
    params.arguments.resize(arguments_desc_size);
    for (auto& arg : params.arguments) {
        ib >> make_data(&arg.t, sizeof(cldnn::argument_desc::Types)) >> arg.index;
    }

    typename cldnn::scalars_desc::size_type scalars_desc_size = 0UL;
    ib >> scalars_desc_size;
    params.scalars.resize(scalars_desc_size);
    for (auto& scalar : params.scalars) {
        ib >> make_data(&scalar.t, sizeof(cldnn::scalar_desc::Types)) >> make_data(&scalar.v, sizeof(cldnn::scalar_desc::ValueT));
    }

    ib >> params.layerID;

#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t n_microkernels;
    ib >> n_microkernels;
    micro_kernels.clear();
    for (size_t i = 0; i < n_microkernels; i++) {
        auto microkernel = std::make_shared<micro::MicroKernelPackage>();
        microkernel->load(ib);
        micro_kernels.push_back(microkernel);
    }
#endif
}

}  // namespace kernel_selector
