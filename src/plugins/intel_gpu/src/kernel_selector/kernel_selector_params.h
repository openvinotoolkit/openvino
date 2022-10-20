// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <cstddef>
#include <limits>
#include "common_types.h"
#include "tensor_type.h"
#include "document.h"
#include <vector>
#include <utility>
#include <bitset>

namespace kernel_selector {
using DataTensor = Tensor::DataTensor;
using WeightsTensor = Tensor::WeightsTensor;
using DataLayout = Tensor::DataLayout;
using WeightsLayout = Tensor::WeightsLayout;
using MultiDataTensor = std::vector<DataTensor>;
using DataBitField = std::bitset<DataLayout::DataLayoutCount>;
using WightsBitField = std::bitset<WeightsLayout::WeightsLayoutCount>;

class JitConstants;
class TuningCache;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fuse_params {
    virtual ~fuse_params() {}

    KernelType GetType() const { return kType; }
protected:
    explicit fuse_params(KernelType kt) : kType(kt) {}
    KernelType kType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ParamsKey
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ParamsKey {
public:
    ParamsKey() {
        key.restrict.raw = 0;
        key.enableTuning = 1;
        key.machineInfo.raw = 0;
        key.inputType.raw = 0;
        key.outputType.raw = 0;
        key.inputWeightsType.raw = 0;
        key.outputWeightsType.raw = 0;
        key.inputLayout = 0;
        key.outputLayout = 0;
        key.weightsInputLayout = 0;
        key.weightsOutputLayout = 0;
    }

    struct Key {
        union restrict_t {
            struct val_t {
                uint32_t different_types : 1;
                uint32_t different_input_weights_types : 1;
                uint32_t offset : 1;
                uint32_t pitches : 1;
                uint32_t batching : 1;
                uint32_t biasPerFeatureMap : 1;
                uint32_t biasPerOutput : 1;
                uint32_t nonBias : 1;
                uint32_t activationAdditionalParamsAsInput : 1;
                uint32_t FP16Emulation : 1;
                uint32_t momentum : 1;
                uint32_t quantization : 1;
                uint32_t sym_quantization : 1;
                uint32_t asym_w_quantization : 1;
                uint32_t asym_d_quantization : 1;

                union dedicated_t {
                    struct argm_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisZ : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                    } argm;
                    struct idxsel_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                    } idxsel;
                    struct norm_t {
                        uint32_t across : 1;
                        uint32_t within : 1;
                        uint32_t fixedKenrelDivider : 1;
                        uint32_t dynamicKenrelDivider : 1;
                    } norm;
                    struct mvn_t {
                        uint32_t across : 1;
                        uint32_t within : 1;
                        uint32_t normalize_variance : 1;
                    } mvn;
                    struct pooling_t {
                        uint32_t max : 1;
                        uint32_t avg : 1;
                        uint32_t floor : 1;
                        uint32_t max_with_argmax : 1;
                        uint32_t ceil : 1;
                        uint32_t bilinear : 1;
                        uint32_t deformable_bilinear : 1;
                        uint32_t fixedKenrelDivider : 1;
                        uint32_t dynamicKenrelDivider : 1;
                        uint32_t dynamicKenrelDividerWithPadding : 1;
                        uint32_t position_sensitive : 1;
                        uint32_t dilation : 1;
                        uint32_t indices_output : 1;
                    } pooling;
                    struct conv_t {
                        uint32_t split : 1;
                        uint32_t dilation : 1;
                        uint32_t depthwise_separable_opt : 1;
                        uint32_t grouped : 1;
                        uint32_t deformable : 1;
                        uint32_t bilinear_interpolation_pad : 1;
                        uint32_t deformable_mask_enabled : 1;
                    } conv;
                    struct fc_t {
                    } fc;
                    struct softmax_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimZ : 1;
                        uint32_t dimFeature : 1;
                        uint32_t dimBatch : 1;
                    } softmax;
                    struct region_yolo_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimFeature : 1;
                        uint32_t coords : 1;
                        uint32_t classes : 1;
                        uint32_t num : 1;
                    } region_yolo;
                    struct reorg_yolo_t {
                        uint32_t dimX : 1;
                        uint32_t dimY : 1;
                        uint32_t dimFeature : 1;
                        uint32_t stride : 1;
                    } reorg_yolo;
                    struct concat_t {
                        uint32_t axisX : 1;
                        uint32_t axisY : 1;
                        uint32_t axisZ : 1;
                        uint32_t axisW : 1;
                        uint32_t axisFeature : 1;
                        uint32_t axisBatch : 1;
                        uint32_t kernelPerInput : 1;
                        uint32_t oneKernel : 1;
                    } concat;
                    struct upsample_t {
                        uint32_t nearest_neighbor : 1;
                        uint32_t caffe_bilinear_interp : 1;
                        uint32_t bilinear_interp : 1;
                        uint32_t cubic : 1;
                        uint32_t linear_onnx : 1;
                    } resample;
                    struct reorder_t {
                        uint32_t winograd : 1;
                        uint32_t rotate : 1;
                    } reorder;
                    struct eltwise_t {
                        uint32_t stride : 1;
                        uint32_t broadcast : 1;
                    } eltwise;
                    struct lstm_gemm_t {
                        uint32_t bias : 1;
                        uint32_t hidden : 1;
                    } lstm_gemm;
                    struct lstm_dynamic_t {
                        uint32_t last_hidden : 1;
                        uint32_t last_cell : 1;
                    } lstm_dynamic;
                    struct lstm_elt_t {
                        uint32_t cell : 1;
                    } lstm_elt;
                    struct quantize_t {
                        uint32_t packed_binary_output : 1;
                        uint32_t scale_shift_opt : 1;
                    } quantize;
                } dedicated;
            } val;
            uint64_t raw;
        } restrict;

        union machine_info_t {
            struct val_t {
                uint32_t subgroup : 1;
                uint32_t subgroupShort : 1;
                uint32_t subgroupChar : 1;
            } val;
            uint32_t raw;
        } machineInfo;

        static_assert(sizeof(restrict_t) == sizeof(uint64_t), "problem with union");

        typedef union DataTypesKey_t {
            struct val_t {
                uint32_t int8 : 1;
                uint32_t uint8 : 1;
                uint32_t int16 : 1;
                uint32_t uint16 : 1;
                uint32_t int32 : 1;
                uint32_t uint32 : 1;
                uint32_t int64 : 1;
                uint32_t F16 : 1;
                uint32_t F32 : 1;
                uint32_t binary : 1;
            } val;
            uint32_t raw;
        } DataTypesKey;

        uint32_t enableTuning;
        DataTypesKey inputType;
        DataTypesKey outputType;
        DataTypesKey inputWeightsType;
        DataTypesKey outputWeightsType;
        DataBitField inputLayout;
        DataBitField outputLayout;
        WightsBitField weightsInputLayout;
        WightsBitField weightsOutputLayout;
    };

    void EnableInputDataType(Datatype dt);
    void EnableAllInputDataType();
    void EnableOutputDataType(Datatype dt);
    void EnableAllOutputDataType();
    void EnableInputWeightsType(WeightsType wt);
    void EnableAllInputWeightsType();
    void EnableOutputWeightsType(WeightsType wt);
    void EnableAllOutputWeightsType();
    void EnableFP16Emulation() { key.restrict.val.FP16Emulation = 1; }
    void EnableDifferentTypes() { key.restrict.val.different_types = 1; }
    void EnableDifferentInputWeightsTypes() { key.restrict.val.different_input_weights_types = 1; }
    void EnableInputLayout(DataLayout l) { key.inputLayout.set(static_cast<size_t>(l)); }
    void EnableAllInputLayout() { key.inputLayout.set(); }
    void EnableOutputLayout(DataLayout l) { key.outputLayout.set(static_cast<size_t>(l)); }
    void EnableAllOutputLayout() { key.outputLayout.set(); }
    void EnableInputWeightsLayout(WeightsLayout l) {
        key.weightsInputLayout.set(static_cast<size_t>(l));
    }
    void EnableAllInputWeightsLayout() { key.weightsInputLayout.set(); }
    void EnableOutputWeightsLayout(WeightsLayout l) {
        key.weightsOutputLayout.set(static_cast<size_t>(l));
    }
    void EnableAllOutputWeightsLayout() { key.weightsOutputLayout.set(); }
    void EnableTensorOffset() { key.restrict.val.offset = 1; }
    void EnableTensorPitches() { key.restrict.val.pitches = 1; }
    void EnableBatching() { key.restrict.val.batching = 1; }
    void EnableSubGroup() { key.machineInfo.val.subgroup = 1; }
    void EnableSubGroupShort() { key.machineInfo.val.subgroupShort = 1; }
    void EnableSubGroupChar() { key.machineInfo.val.subgroupChar = 1; }
    void EnableNonBiasTerm() { key.restrict.val.nonBias = 1; }
    void EnableBiasPerFeature() { key.restrict.val.biasPerFeatureMap = 1; }
    void EnableBiasPerOutput() { key.restrict.val.biasPerOutput = 1; }
    void EnableActivationAdditionalParamsAsInput() { key.restrict.val.activationAdditionalParamsAsInput = 1; }
    void EnableMomentum() { key.restrict.val.momentum = 1; }
    void EnableLRNMode(LRNMode m);
    void EnableNormalizeMode(NormalizeMode m);
    void EnableMVNMode(MVNMode m);
    void EnableMVNNormalizeVariance();
    void EnableLRNKernelDividerMode(KernelDividerMode m);
    void EnablePoolKernelDividerMode(KernelDividerMode m);
    void EnablePoolType(PoolType t);
    void EnablePoolRemainder(PoolRemainder r);
    void EnablePoolDilation() { key.restrict.val.dedicated.pooling.dilation = 1; }
    void EnablePoolIndicesOutput() { key.restrict.val.dedicated.pooling.indices_output = 1; }
    void EnableQuantization(QuantizationType q);
    void EnablePositionSensitivePooling() { key.restrict.val.dedicated.pooling.position_sensitive = 1; }
    void EnableSplitSupport() { key.restrict.val.dedicated.conv.split = 1; }
    void EnableDilation() { key.restrict.val.dedicated.conv.dilation = 1; }
    void EnableDepthwiseSeparableOpt() { key.restrict.val.dedicated.conv.depthwise_separable_opt = 1; }
    void EnableGroupedConvolution() { key.restrict.val.dedicated.conv.grouped = 1; }
    void EnableDeformableMode() { key.restrict.val.dedicated.conv.deformable = 1; }
    void EnableBilinearInterpolationPad() { key.restrict.val.dedicated.conv.bilinear_interpolation_pad = 1; }
    void EnableDeformableMask() { key.restrict.val.dedicated.conv.deformable_mask_enabled = 1; }

    void EnableQuantizePackedBinaryOutput() { key.restrict.val.dedicated.quantize.packed_binary_output = 1; }
    void EnableQuantizeScaleShiftOpt() { key.restrict.val.dedicated.quantize.scale_shift_opt = 1; }

    void EnableWinogradReorder() { key.restrict.val.dedicated.reorder.winograd = 1; }
    void EnableRotateReorder() { key.restrict.val.dedicated.reorder.rotate = 1; }
    void EnableSoftmaxDim(SoftmaxDim d);
    void EnableConcatAxis(ConcatAxis a);
    void EnableReampleType(ResampleType a);
    void EnableEltwiseStride();
    void EnableEltwiseBroadcast() { key.restrict.val.dedicated.eltwise.broadcast = 1; }

    void EnableLSTMGEMMBias() { key.restrict.val.dedicated.lstm_gemm.bias = 1; }
    void EnableLSTMGEMMHidden() { key.restrict.val.dedicated.lstm_gemm.hidden = 1; }
    void EnableLSTMEltCell() { key.restrict.val.dedicated.lstm_elt.cell = 1; }
    void EnableLSTMDyanmicOptionalHiddenOutput() { key.restrict.val.dedicated.lstm_dynamic.last_hidden = 1; }
    void EnableLSTMDyanmicOptionalCellOutput() { key.restrict.val.dedicated.lstm_dynamic.last_cell = 1; }
    void EnableConcatKernelPerInput() { key.restrict.val.dedicated.concat.kernelPerInput = 1; }
    void DisableTuning() { key.enableTuning = 0; }
    void EnableConcatOneKernel() { key.restrict.val.dedicated.concat.oneKernel = 1; }
    void EnableArgMaxMinAxis(ArgMaxMinAxis a);
    void EnableIndexSelectAxis(IndexSelectAxis a);
    void EnableFusedConvEltwiseRWOutOpt();
    bool Support(const ParamsKey& k) const;
    bool TuningSupport() const {
        if (key.enableTuning == 1)
            return true;
        return false;
    }
    bool isEnabledDifferentInputWeightsTypes() const {
        return key.restrict.val.different_input_weights_types ? true : false;
    }
    ParamsKey Merge(const ParamsKey& k) const;

private:
    Key key;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device type
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class dev_type {
    integrated_gpu = 0,
    discrete_gpu = 1
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EngineInfo
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EngineInfo {
    bool bSubGroupSupport = false;
    bool bSubGroupShortSupport = false;
    bool bSubGroupCharSupport = false;
    bool bFP16Support = false;
    bool bFP64Support = false;
    bool bImageSupport = false;
    bool bIMADSupport = false;
    bool bIMMADSupport = false;
    bool bOptHintsSupport = false;
    bool bLocalBlockIOSupport = false;
    dev_type deviceType = dev_type::integrated_gpu;
    uint32_t computeUnitsCount = 0;
    uint32_t maxThreadsPerExecutionUnit = 0;
    uint32_t maxThreadsPerDevice = 0;
    uint64_t maxWorkGroupSize = 0;
    uint64_t maxLocalMemSize = 0;
    uint64_t maxImage2dWidth = 0;
    uint64_t maxImage2dHeight = 0;
    std::string deviceId = "";
    std::string driverVersion = "";
    std::vector<size_t> supportedSimdSizes = {};
    std::shared_ptr<TuningCache> deviceCache;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct Params {
    virtual ~Params() {}

    KernelType GetType() const { return kType; }
    virtual ParamsKey GetParamsKey() const;

protected:
    Params(KernelType kt, const std::string& id) : kType(kt), layerID(id) {}
    KernelType kType;

public:
    std::string layerID;
    std::string forceImplementation;
    EngineInfo engineInfo;
    std::string uniqueID;

    virtual std::string to_string() const;
    virtual std::string to_cache_string_v2() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_activation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct base_activation_params {
    ActivationFunction function = ActivationFunction::NONE;
    float m = 1.f;
    float n = 0.f;

    base_activation_params() = default;
    base_activation_params(const float m, const float n) : m(m), n(n) {}
    base_activation_params(const ActivationFunction f, const float m, const float n) : function(f),
                                                                                       m(m),
                                                                                       n(n) {}

    virtual std::string to_string() const;
};

struct FusedOpsConfiguration {
    enum class LoadType {
        LT_UNALIGNED = 0,
        LT_ALIGNED_READ = 1,
        FEATURE_SHUFFLE = 2
    };

    enum class BoundaryCheck {
        DISABLED = 0,
        ENABLED = 1
    };

    enum class IndexType {
        TENSOR_COORD = 0,
        LINEAR_OFFSET = 1
    };

    // Optional suffix that is added to each macro in the configuration.
    std::string suffix;
    // Indices to load additional data for a fused op.
    std::vector<std::string> bfzyx_idx_order;
    // Name of the input variable for the first fused op.
    std::string input_var_name;
    // Data type of the input
    Datatype input_dt;
    // Data type vector size of the input
    size_t vec_size;
    // Represents a channel in the input tensor that is loaded to the input variable
    Tensor::DataChannelName vec_axis;
    // Sets used load type - aligned or unaligned. Aligned load requires specific extensions and adjusted indices.
    LoadType load_type;
    // Defines if safe index function should be used for offset calculation
    BoundaryCheck boundary_check;
    // Defines how to treat indices array
    IndexType index_type;
    // Defines outer loops channels where fused op is called.
    std::vector<Tensor::DataChannelName> loop_axes;
    // If allow_for_partial_preload is false, then it's required that all fused_ops can be preloaded.
    // If allow_for_partial_preload is true, then not preloaded fused_ops will be loaded in FUSED_OPS_CALC.
    bool allow_for_partial_preload;
    // Load index for shuffle fused op
    std::string shuffle_var_name;
    // Record original output layout before reorder is fused
    DataLayout orig_output_layout;

    FusedOpsConfiguration(std::string suffix,
                          std::vector<std::string> bfzyx_idx_order,
                          std::string input_var_name,
                          Datatype input_dt,
                          size_t vec_size = 1,
                          LoadType load_type = LoadType::LT_UNALIGNED,
                          BoundaryCheck boundary_check = BoundaryCheck::ENABLED,
                          IndexType index_type = IndexType::TENSOR_COORD,
                          Tensor::DataChannelName vec_axis = Tensor::DataChannelName::COUNT,
                          std::vector<Tensor::DataChannelName> loop_axes = {},
                          bool allow_for_partial_preload = false,
                          std::string shuffle_var_name = "",
                          DataLayout orig_output_layout = DataLayout::DataLayoutCount)
      : suffix(suffix)
      , bfzyx_idx_order(bfzyx_idx_order)
      , input_var_name(input_var_name)
      , input_dt(input_dt)
      , vec_size(vec_size)
      , vec_axis(vec_axis)
      , load_type(load_type)
      , boundary_check(boundary_check)
      , index_type(index_type)
      , loop_axes(loop_axes)
      , allow_for_partial_preload(allow_for_partial_preload)
      , shuffle_var_name(shuffle_var_name)
      , orig_output_layout(orig_output_layout) { }

    FusedOpsConfiguration& SetVectorSize(size_t val) { vec_size = val; return *this; }
    FusedOpsConfiguration& SetLoadType(LoadType val) { load_type = val; return *this; }
    FusedOpsConfiguration& SetBoundaryCheck(BoundaryCheck val) { boundary_check = val; return *this; }
    FusedOpsConfiguration& SetIndexType(IndexType val) { index_type = val; return *this; }
    FusedOpsConfiguration& SetVectorAxis(Tensor::DataChannelName val) { vec_axis = val; return *this; }
    FusedOpsConfiguration& SetLoopAxes(std::vector<Tensor::DataChannelName> val, bool partial_preload = false) {
        loop_axes = std::move(val);
        allow_for_partial_preload = partial_preload;
        return *this; }
    FusedOpsConfiguration& SetShuffleVarName(std::string val) { shuffle_var_name = val; return *this; }
    bool IsPostReorderFused(void) const { return orig_output_layout != DataLayout::DataLayoutCount; }
};

// Dependency(Input) type of fusing operation in fused node.
// There are different ways to generate input var name and type by the dependency(input) type in MakeOpJitConstants in jitter
// - ORIGINAL: The input of the operation is the fused node such as Conv
// - EXTERNAL: The input of the operation is the external node outside the fused node
// - INTERNAL: The input of the operation is the another fused operation in the fused node
enum class DepType {
    UNDEFINED  = -1,
    ORIGINAL   = 0,
    EXTERNAL   = 1,
    INTERNAL   = 2
};

// Dependency(Input) information of fusing operation which is used to generate input var name and type
// in MakeOpJitConstants in jitter
struct dep_info {
    DepType     dep_type = DepType::UNDEFINED;
    size_t      op_id;
    Datatype    data_type;
};

// Instance of fused_operation_desc is added to fused_ops vector if a node has been fused to current one using program::fuse_nodes
// method. In order to process fused ops following modifications should be done in a kernel:
// option 1 - using common generator:
//     - create FusedOpsConfiguration object that contains configuration for common code generator.
//       Multiple objects can be created if a kernel uses different data types at the same time. E.g. kernels that contains scalar and
//       vector branches that are chosen in runtime. To handle this case, create 2 configurations with different suffixes, like
//       "_SCALAR" and "_VEC" and then use generated macros accordingly.
//     - add jit constants returned by KernelBase::MakeFusedOpsJitConstants method to the kernel's constants.
//     - insert generated macros in the ocl code:
//       in kernel declaration:
//         #if HAS_FUSED_OPS_DECLS
//           FUSED_OPS_DECLS,
//         #endif
//       in kernel body:
//         #if HAS_FUSED_OPS
//           FUSED_OPS<OPTIONAL_SUFFIX>;
//           <SOME_VARIABLE> = FUSED_OPS_RESULT<OPTIONAL_SUFFIX>;
//         #endif
//   In this case common generator creates set of definitions for each op which are called sequentially in FUSED_OP<OPTIONAL_SUFFIX>
//   macro. Example:
//     #define FUSED_OPS
//       FUSED_OP0_LOAD_VEC
//       FUSED_OP0_ACTION_VEC
//       FUSED_OP1_LOAD_VEC
//       FUSED_OP1_ACTION_VEC
//     #define FUSED_OP0_LOAD_VEC
//       MAKE_VECTOR_TYPE(FUSED_OP_0_INPUT0_TYPE,2) activation0_data0 = UNIT_BLOCK_READ(activation0_input0,
//                                                                      FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(f_block*16),0,0));
//     #define FUSED_OP0_ACTION_VEC
//       float2 dst_0 = dst;
//       dst_0 = ACTIVATION_FUSED_OP0_VEC(dst_0, ACTIVATION_PARAMS_FUSED_OP0_VEC);
//     #define FUSED_OP1_LOAD_VEC
//       MAKE_VECTOR_TYPE(FUSED_OP_1_INPUT0_TYPE,2) eltwise1_data0 = UNIT_BLOCK_READ2(eltwise1_input0,
//                                                                   FUSED_OP_1_INPUT0_GET_INDEX_SAFE(0,(f_block*16),y,x));
//     #define FUSED_OP1_ACTION_VEC
//       float2 dst_0_2 = convert_float2(eltwise1_data0) + convert_float2(dst_0);
//     #define FUSED_OPS_RESULT_VEC dst_0_2
// option 2 - using custom generator in a kernel. It can be used if performance is not optimal in the common one or to handle
//            some difficult cases that can't be unified. Custom processing of fused ops can be written absolutely independently
//            in a kernel, but to make it easier set of helper functions exist:
//     - KernelBase::MakeFusedOpsDeclsJitConstants that creates arguments for kernel declaration and macro for all tensors used in
//       a fused op (requires FusedOpsConfiguration instance).
//     - fused_operation_desc contains a bunch of methods to generate variable/pointer names, type conversions, data loads
//  If you need an example of custom code generation for fused ops, check BinaryConvolutionKernelGeneric::GetFusedPrimitivesJitConstants
//  method in binary_convolution_kernel_generic.cpp.
struct fused_operation_desc {
    std::shared_ptr<fuse_params> op_params;
    size_t dep_idx_start;
    size_t dep_size;
    MultiDataTensor tensors;
    DataTensor output_tensor;
    size_t op_id;
    std::vector<dep_info> dep_data = {};

    // Helper functions for operation generation
    KernelType GetType() const { return op_params->GetType(); }
    template<typename T>
    std::shared_ptr<T> GetOpParams() const {
        auto p = std::dynamic_pointer_cast<T>(op_params);
        if (!p)
            throw std::runtime_error("Invalid dynamic cast of fused operation parameters");

        return p;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct base_params : public Params {
    virtual ~base_params() {}

    std::vector<base_activation_params> activations;
    std::vector<fused_operation_desc> fused_ops = {};
    MultiDataTensor inputs;
    MultiDataTensor outputs;
    std::string to_string() const override;
    std::string to_cache_string_v2() const override;
    ParamsKey GetParamsKey() const override;

protected:
    explicit base_params(KernelType kt) : Params(kt, ""), inputs(1), outputs(1) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Auto tuner parameters
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class KernelRunnerInterface;
struct TuningParams {
    TuningMode mode;
    std::string cacheFilePath;
    std::shared_ptr<KernelRunnerInterface> runner;

    TuningParams() : mode(TuningMode::TUNING_DISABLED), cacheFilePath(""), runner(nullptr) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct optional_params {
    virtual ~optional_params() {}

    KernelType GetType() const { return kType; }

    std::vector<DataLayout> inputLayouts;
    std::vector<DataLayout> outputLayouts;

    bool meaningfulKernelsNames = false;  // use layer name instead of internal kernel name
    bool allowStaticInputReordering =
        true;  // allow kernel to provide a kernel which reorder static data like weights/bias/tables...
    bool allowInputReordering =
        false;  // allow kernel to ask graph compiler to reorder the input data before executing its
    bool allowOutputReordering =
        false;  // allow kernel to ask graph compiler to reorder the output data before executing the next kernel

    TuningParams tuningParams;

    virtual ParamsKey GetSupportedKey() const;

protected:
    explicit optional_params(KernelType kt) : kType(kt) {}
    KernelType kType;
};

}  // namespace kernel_selector
