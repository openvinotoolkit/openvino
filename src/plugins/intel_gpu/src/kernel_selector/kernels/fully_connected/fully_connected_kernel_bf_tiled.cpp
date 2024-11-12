// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bf_tiled.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <functional>
#include "common_types.h"

static constexpr size_t lws_batches = 8;
static constexpr size_t simd = 16;
static constexpr size_t min_quantize_grp_size = 32;
static constexpr size_t min_slm_size = 256;
static std::vector<size_t> available_quantize_grp_size = {128, 64, 32};

namespace kernel_selector {

static std::pair<size_t, size_t> get_input_bf_size(const fully_connected_params& params) {
    auto& input = params.inputs[0];
    size_t input_f = input.Feature().v;
    size_t input_batch = input.Batch().v;

    // 3D input
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        input_f = input.Y().v;
        input_batch = input.Batch().v * input.Feature().v;
    }

    // In Some model, input_f could be dynamic in input0. It refers to IFM value of weight.
    if (input.is_dynamic() && input_f == 0 && params.weights.IFM().v != 0)
        input_f = params.weights.IFM().v;

    return {input_batch, input_f};
}

static std::pair<size_t, size_t> get_output_aligned_bf_size(const fully_connected_params& params,
                                                            bool needs_align,
                                                            uint32_t align_b = 1,
                                                            int32_t align_f = 1) {
    size_t output_f = (needs_align == true) ? CeilDiv(params.outputs[0].Feature().v, align_f) : params.outputs[0].Feature().v;
    size_t output_b = params.outputs[0].Batch().v;
    // 3D output
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        output_f = (needs_align == true) ? CeilDiv(params.outputs[0].Y().v, align_f) : params.outputs[0].Y().v;
        output_b = params.outputs[0].Batch().v * params.outputs[0].Feature().v;
    }

    output_b = (needs_align == true) ? CeilDiv(output_b, align_b) : output_b;

    return {output_b, output_f};
}

static bool is_weight_dyn_quantizable(const fully_connected_params& params) {
    auto weight_type = params.weights.GetDType();
    if (weight_type == WeightsType::INT4 || weight_type == WeightsType::UINT4)
        return true;
    // UINT8 weight type is supported by FC dyn-quantize(with SLM).
    if (weight_type == WeightsType::UINT8)
        return true;

    return false;
}

// DYNAMIC_QUANTIZE
static size_t get_dynamic_quantize_group_size(const fully_connected_params& params) {
    auto dynamic_quantization_group_size = params.dynamic_quantization_group_size;

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->dynamic_quantize_group_size != debug_config->DYNAMIC_QUANTIZE_GROUP_SIZE_NOT_SET) {
        dynamic_quantization_group_size = debug_config->dynamic_quantize_group_size;

        // Specify which Fully-connected layer would be dynamic-quantized
        GPU_DEBUG_IF(!debug_config->dynamic_quantize_layers_without_onednn.empty()) {
            auto layers = debug_config->dynamic_quantize_layers_without_onednn;
            auto iter = std::find_if(layers.begin(), layers.end(), [&](const std::string& pattern){
                return debug_config->is_layer_name_matched(params.layerID, pattern);
            });

            if (iter != layers.end()) {
                dynamic_quantization_group_size = debug_config->dynamic_quantize_group_size;
                GPU_DEBUG_COUT << "Found specified Fully-connected layer [" << params.layerID << "]. Enable Dynamic-quantize." << std::endl;
            } else {
                dynamic_quantization_group_size = 0;
            }
        }
    }

    const size_t scale_group_size = params.weights.IFM().v / params.decompression_scale.Feature().v;
    for (auto group_size : available_quantize_grp_size) {
        if (dynamic_quantization_group_size >= group_size) {
            dynamic_quantization_group_size = group_size;

            if (dynamic_quantization_group_size > scale_group_size) {
                GPU_DEBUG_TRACE_DETAIL << " Scale group size " << scale_group_size << " is smaller than FC dyn-quan group size "
                                        << dynamic_quantization_group_size << ". Reduce FC dyn-quan group size to scale size." << std::endl;
                dynamic_quantization_group_size = scale_group_size;
            }
            return (size_t)dynamic_quantization_group_size;
        }
    }

    return 0;
}

static bool should_dynamic_quantize(const fully_connected_params& params, bool print_log = false) {
    size_t dynamic_quantization_group_size = get_dynamic_quantize_group_size(params);

    if (params.inputs[0].GetFirstElementOffset() != 0)
        return false;

    if (dynamic_quantization_group_size < min_quantize_grp_size) {
            GPU_DEBUG_TRACE_DETAIL << "Set dynamic_quantize_group_size " << dynamic_quantization_group_size
                            << " is smaller than minimum supported size 32" << std::endl;
            return false;
    }

    auto threads = get_input_bf_size(params);
    auto input_b = threads.first;
    auto input_f = threads.second;

    const size_t scale_group_size = params.weights.IFM().v / params.decompression_scale.Feature().v;
    if ((scale_group_size % simd == 0) && (input_f % dynamic_quantization_group_size == 0) &&
        (params.is_shape_agnostic || (params.inputs[0].Batch().v > 1 && input_b > min_slm_size)) &&
        params.inputs[0].GetDType() == Datatype::F16 && is_weight_dyn_quantizable(params)) {
            if (print_log) {
                GPU_DEBUG_TRACE_DETAIL << " Dynamic quantizing for FC : scale_group_size: " << scale_group_size <<
                    ", Dyn-quan group size: " << dynamic_quantization_group_size <<
                    ", Type(I:" << kernel_selector::toString(params.inputs[0].GetDType()) <<
                    ", O:" << kernel_selector::toString(params.outputs[0].GetDType()) <<
                    ", W:" << kernel_selector::toString(params.weights.GetDType()) <<
                    "), Format(W:" << kernel_selector::toString(params.weights.GetLayout()) <<
                    ") B: " << params.inputs[0].Batch().v << ", F: " << params.inputs[0].Feature().v <<
                    ", Y: " << params.inputs[0].Y().v << std ::endl;
            }
        return true;
    }

    return false;
}

static bool is_weight_vertical(const fully_connected_params& params, size_t output_f) {
    size_t min_num_threads = params.engineInfo.computeUnitsCount * simd;
    GPU_DEBUG_TRACE_DETAIL << "out_ofm (== weight N dim) size " << output_f << " is small compared to the available threads. "
                           << "(computeUnitsCount : " << params.engineInfo.computeUnitsCount
                           << " min_num_threads : " << min_num_threads << ")" << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "Use ofm_tile size 1 if the batch size is 1." << std::endl;
    return (params.weights.IFM().v >= params.weights.OFM().v * 3
            && output_f / 2 /*most frequently used tile_ofm*/ <= min_num_threads);
}

static bool is_weight_horizontal(const fully_connected_params& params, size_t output_f) {
    size_t min_num_threads = params.engineInfo.computeUnitsCount * simd;
    GPU_DEBUG_TRACE_DETAIL << "out_ofm (== weight N dim) size " << output_f << " is large compared to the available threads. "
                           << "(computeUnitsCount : " << params.engineInfo.computeUnitsCount
                           << " min_num_threads : " << min_num_threads << ")" << std::endl;
    return (params.weights.OFM().v > params.weights.IFM().v * 3
            && output_f / 4 /* tile_ofm=4 */ > min_num_threads * 1.5);
}

static bool is_suitable_outer_ofm(const fully_connected_params& params, size_t output_f) {
    size_t min_num_threads = params.engineInfo.computeUnitsCount * simd;
    return (params.weights.OFM().v > params.weights.IFM().v * 6
            && output_f / 8 /* tile_ofm=4 and outer_ofm=2 */ > min_num_threads * 1.5);
}

FullyConnected_bf_tiled::FullyConnected_bf_tiled() : FullyConnectedKernelBase("fully_connected_gpu_bf_tiled") {
    for (unsigned tile_b = 1; tile_b <= 32; ++tile_b)
    for (unsigned tile_ofm = 1; tile_ofm <= 4; tile_ofm *= 2)
    for (unsigned tile_ifm = 1; tile_ifm <= 2; tile_ifm *= 2)
    for (unsigned tile_k = 1; tile_k <= 8; tile_k *= 2)
    for (unsigned outer_ofm = 1; outer_ofm <= 2; ++outer_ofm)
    for (unsigned dispatch_bsv = 1; dispatch_bsv <= 16; ++dispatch_bsv)
    for (unsigned dispatch_fsv = 1; dispatch_fsv <= 16; ++dispatch_fsv)
    for (auto exec : Parent::autoTuneOptions) {
        // Block reads support at most vector size of 8.
        if (tile_k * tile_ofm > 8)
            continue;
        // For bsv == 1 dispatch order reduces to b_fsv, so anything other than fsv == 1 is redundant.
        if (dispatch_bsv == 1 && dispatch_fsv != 1)
            continue;

        auto_tune_params.emplace_back(tile_b, tile_ofm, tile_ifm, tile_k, outer_ofm, dispatch_bsv, dispatch_fsv, exec);
    }
}

ParamsKey FullyConnected_bf_tiled::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::UINT4);
    k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

DeviceFeaturesKey FullyConnected_bf_tiled::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

bool FullyConnected_bf_tiled::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];
    auto& output = fc_params.outputs[0];
    auto& weights = fc_params.weights;

    // Block reads must be aligned to 4 bytes, for fp16 we can correct for offset misalignment,
    // but we need to ensure that batch pitch preserves alignment.
    if (input.GetDType() == Datatype::F16) {
        if (input.Batch().pitch % 2 != 0 && (input.Batch().v > 1 || fc_params.is_shape_agnostic))
            return false;
        // for 3d case we have to check feature alignment as well
        if (output.GetLayout() == DataLayout::bfyx && input.Feature().pitch % 2 != 0 && (input.Feature().v > 1 || fc_params.is_shape_agnostic))
            return false;
    }

    // Dynamic kernel doesn't support dynamic weights yet
    if (fc_params.is_shape_agnostic && input.is_dynamic()) {
        if (get_input_bf_size(fc_params).second == 0)
            return false;
    }

    if (input.GetLayout() == DataLayout::bfyx) {
        // Padding on input is not supported.
        // TODO: Enable by mirroring the padding in weights.
        if (input.X().pad.Total() != 0)
            return false;
        if (input.Y().pad.Total() != 0)
            return false;
    }

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx) {
        if (input.X().v > 1)
            return false;
    }

    auto wt = weights.GetDType();
    if ((wt == WeightsType::UINT4 || wt == WeightsType::INT4) && (weights.IFM().v % 2 != 0 || weights.OFM().v % 2 != 0)) {
        return false;
    }

    return true;
}

namespace {

struct TuneParamsSelector {
    using tune_params = FullyConnected_bf_tiled::tune_params;
    using functional_case = std::function<tune_params(const fully_connected_params&)>;

    TuneParamsSelector(const fully_connected_params& params) : params(params), selected(false) {}

    TuneParamsSelector& Case(const tune_params& tparams) {
        if (!selected && VerifyTuneParams(params, tparams)) {
            result = tparams;
            selected = true;
        }
        return *this;
    }

    TuneParamsSelector& Case(functional_case fun) {
        return Case(fun(params));
    }

    tune_params Default(const tune_params& tparams) {
        if (!selected) {
            selected = true;
            result = tparams;
        }
        return result;
    }

    static bool VerifyTuneParams(const fully_connected_params& params, const tune_params& tparams);

    const fully_connected_params& params;
    bool selected;
    tune_params result;
};

bool TuneParamsSelector::VerifyTuneParams(const fully_connected_params& params, const tune_params& tparams) {
    // Check divisibility by dispatch tile sizes.
    auto bf_size = get_output_aligned_bf_size(params, false);
    size_t output_b = bf_size.first;
    size_t output_f = bf_size.second;

    auto batch_size = params.is_shape_agnostic ? Align(output_b, tparams.tile_b) : output_b;
    // If batch size is prime number, still can apply tile execution to avoid poor performance.
    if (batch_size % (tparams.tile_b * tparams.dispatch_bsv) != 0) {
        if ((tparams.dispatch_bsv != 1) || batch_size == 1)
            return false;
        size_t tile = simd;
        while (batch_size % tile != 0)
            tile--;
        if (tile > 1)
            return false;
    }

    if (CeilDiv(output_f, tparams.tile_ofm * simd) % tparams.dispatch_fsv != 0)
        return false;

    // Same result can be achieved with smaller tile_ofm.
    if (output_f <= (tparams.tile_ofm / 2) * simd)
        return false;
    // No weights layout for such huge tile ofm.
    if (tparams.tile_ofm * simd > 64)
        return false;

    bool is_dyn_quantable_type = is_weight_dyn_quantizable(params);
    if (tparams.kernel_type == FullyConnected_bf_tiled::KernelType::SLM) {
        const auto required_batch_alignment = 64;
        if (!params.is_shape_agnostic && (!IsAligned(output_b, required_batch_alignment) || output_b < min_slm_size))
            return false;

        const auto required_tile_b = 8;
        if ((tparams.tile_b != required_tile_b) && !is_dyn_quantable_type)
            return false;

        const auto required_tile_ofm = 2;
        if (tparams.tile_ofm != required_tile_ofm)
            return false;

        if (!is_dyn_quantable_type)
            return false;

        if (params.engineInfo.deviceType != dev_type::integrated_gpu)
            return false;

        const auto required_slm_size = tparams.tile_ofm * simd * tparams.tile_ifm * simd * 2; // 2 bytes per value (FP16 data type)
        if (params.engineInfo.maxLocalMemSize < required_slm_size)
            return false;

        return true;
    }
    if (params.compressed && is_dyn_quantable_type) {
        if (!(tparams.tile_ofm == 2 || tparams.tile_ofm == 4))
            return false;
        if (tparams.tile_ofm == 4 && tparams.outer_ofm == 2 && !is_suitable_outer_ofm(params, output_f))
            return false;
        return true;
    }

    // Reject tile sizes that are guaranteed to spill out of registers.
    unsigned acc_register_bytes = tparams.tile_b * tparams.tile_ofm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned in_register_bytes = tparams.tile_b * tparams.tile_ifm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned wei_register_bytes = tparams.tile_ofm * tparams.tile_k * simd * BytesPerElement(params.weights.GetDType());

    unsigned total_register_bytes = acc_register_bytes + in_register_bytes + wei_register_bytes;
    unsigned max_register_bytes = 128 * 32;

    if (total_register_bytes > max_register_bytes)
        return false;

    return true;
}

}  // namespace

FullyConnected_bf_tiled::tune_params
FullyConnected_bf_tiled::GetAutoTuneParams(const fully_connected_params& params, KernelType preferred_kernel_type, int idx) const {
    if (idx >= 0 && idx < static_cast<int>(auto_tune_params.size())
        && TuneParamsSelector::VerifyTuneParams(params, auto_tune_params[idx]))
        return auto_tune_params[idx];

    auto bf_size = get_output_aligned_bf_size(params, false);
    size_t batch = bf_size.first;
    size_t output_f = bf_size.second;

    Datatype dtype = params.inputs[0].GetDType();

    auto selector = TuneParamsSelector(params);

    unsigned max_tile_ofm = 1;
    while (max_tile_ofm * 2 * simd <= output_f && max_tile_ofm < 4)
        max_tile_ofm *= 2;

    if (params.weights.GetDType() == WeightsType::UINT4 || params.weights.GetDType() == WeightsType::INT4 ||
        (is_weight_dyn_quantizable(params) && should_dynamic_quantize(params))) {
        // Only 4bit weight type is fully optimized to use SLM. In default kernel, SLM is not applied to 8bit weight.
        if (!params.is_shape_agnostic && batch == 1) {
            // Tuning for Meteor Lake
            if (is_weight_vertical(params, output_f)) {
                if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
                    return selector.Default(tune_params(1, 1, 4, 2, 1, 1, 1, EXE_MODE_DEFAULT));
                } else if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
                    return selector.Default(tune_params(1, 1, 4, 4, 1, 1, 1, EXE_MODE_DEFAULT));
                }
            } else {
                if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
                    return selector.Default(tune_params(1, 1, 4, 4, 1, 1, 1, EXE_MODE_DEFAULT));
                } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
                    selector.Case(tune_params(1, 4, 4, 2, 2, 1, 1, EXE_MODE_DEFAULT))
                            .Case(tune_params(1, 4, 4, 2, 1, 1, 1, EXE_MODE_DEFAULT));
                } else {
                    return selector.Default(tune_params(1, 2, 4, 2, 1, 1, 1, EXE_MODE_DEFAULT));
                }
            }
        } else {
            // Try to use SLM kernels if possible
            if (preferred_kernel_type != KernelType::DEFAULT) {
                if (params.is_shape_agnostic && !should_dynamic_quantize(params)) {
                    selector.Case(tune_params(16, 2, 2, 4, 1, 1, 1, EXE_MODE_DEFAULT, KernelType::SLM))
                            .Case(tune_params(16, 2, 1, 4, 1, 1, 1, EXE_MODE_DEFAULT, KernelType::SLM));
                }

                selector.Case(tune_params(8, 2, 2, 4, 1, 1, 1, EXE_MODE_DEFAULT, KernelType::SLM))
                        .Case(tune_params(8, 2, 1, 4, 1, 1, 1, EXE_MODE_DEFAULT, KernelType::SLM));
            }

            if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16)
                return selector.Default(tune_params(8, 1, 1, 4, 1, 1, 1, EXE_MODE_DEFAULT));
            else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2)
                return selector.Default(tune_params(8, 4, 1, 2, 1, 1, 1, EXE_MODE_DEFAULT));
            else
                return selector.Default(tune_params(8, 2, 1, 4, 1, 1, 1, EXE_MODE_DEFAULT));
        }
    } else if (params.compressed && params.engineInfo.supports_immad) {
        return selector.Default(tune_params(1, 1, 1, 4, 1, 1, 1, EXE_MODE_DEFAULT));
    } else if (params.is_shape_agnostic) {
        // Use special tuning params for Gen12HP dGPUs, since these parameters demonstrate higher performance
        // due to better HW utilization (reduced TILE_OFM parameter) and better assembler kernel's code
        // generation (extended TILE_K parameter) for both FP16 and FP32 data types
        if (dtype == Datatype::F16) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, outer_ofm, dispatch_bsv, dispatch_fsv, exec_options)
            if (params.engineInfo.supports_immad)
                selector.Case(tune_params(8, 1, 1, 4, 1, 1, 1, EXE_MODE_AGE_BASED));
            else
                selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 1, 1, EXE_MODE_AGE_BASED));
        } else if (dtype == Datatype::F32) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, outer_ofm, dispatch_bsv, dispatch_fsv, exec_options)
            if (params.engineInfo.supports_immad)
                selector.Case(tune_params(8, 1, 1, 4, 1, 1, 1, EXE_MODE_AGE_BASED));
            else
                selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 1, 1, EXE_MODE_AGE_BASED));
        }
    } else {
        if (dtype == Datatype::F16) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, outer_ofm, dispatch_bsv, dispatch_fsv, exec_options)
            selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 16, 2, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 16, 1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 1, 4,  2, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 8,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 1, 2,  2, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 4,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 1, 1,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1, 1,  1, EXE_MODE_AGE_BASED));
        } else if (dtype == Datatype::F32) {
            // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, outer_ofm, dispatch_bsv, dispatch_fsv, exec_options)
            selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 16, 2, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 16, 1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 8,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 4,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 2,  1, EXE_MODE_AGE_BASED))
                    .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1, 1,  1, EXE_MODE_AGE_BASED));
        }

        if (params.compressed && batch == 1)
            selector.Case(tune_params(1,  std::min(max_tile_ofm, 2u), 4, 2, 1, 1, 1, EXE_MODE_AGE_BASED));

        selector.Case([&](const fully_connected_params&) -> tune_params {
            tune_params result(8, std::min(max_tile_ofm, 2u), 1, 2, 1, 1, 1, EXE_MODE_DEFAULT);

            while (batch % result.tile_b != 0)
                result.tile_b--;

            result.dispatch_bsv = 16;
            while (batch % (result.tile_b * result.dispatch_bsv) != 0)
                result.dispatch_bsv--;

            if (result.tile_b >= 8)
                result.exec_options = EXE_MODE_AGE_BASED;

            return result;
        });
    }

    return selector.Default(tune_params(1, 1, 1, 1, 1, 1, 1, EXE_MODE_DEFAULT));
}

FullyConnected_bf_tiled::DispatchData
FullyConnected_bf_tiled::SetDefault(const fully_connected_params& params, int autoTuneIndex, int kernel_number) const {
    auto dispatchData = Parent::SetDefault(params);

    // Use KernelType::ANY by default, in case of shape-agnostic kernel, choose kernel type based
    // on `kernel_number` (this implementation allows to have 2 shape-agnostic kernels at the same time
    // for small batches and large batches and change them during inference on the fly)
    auto kernel_type = KernelType::ANY;
    if (params.is_shape_agnostic)
        kernel_type = kernel_number == 0 ? KernelType::DEFAULT : KernelType::SLM;

    auto tparams = GetAutoTuneParams(params, kernel_type, autoTuneIndex);

    auto threads = get_output_aligned_bf_size(params, true, tparams.tile_b, tparams.tile_ofm * tparams.outer_ofm * simd);
    auto batch_threads = threads.first;
    auto feature_threads = threads.second;

    const size_t aligned_batch = Align(batch_threads, lws_batches); // Each WG calculates 8x8 batches (TILE_B x LWS[2] size)
    const bool can_use_slm = tparams.kernel_type == KernelType::SLM;

    dispatchData.gws[0] = can_use_slm ? feature_threads * simd
                                      : feature_threads * batch_threads * simd;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = can_use_slm ? aligned_batch : 1;

    dispatchData.lws[0] = simd;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = can_use_slm ? lws_batches : 1;

    dispatchData.tile_m = tparams.tile_b;
    dispatchData.tile_n = tparams.tile_ofm;
    dispatchData.tile_mk = tparams.tile_ifm;
    dispatchData.tile_nk = tparams.tile_k;
    dispatchData.outer_n = tparams.outer_ofm;
    dispatchData.tile_ms = tparams.dispatch_bsv;
    dispatchData.tile_ns = tparams.dispatch_fsv;
    dispatchData.use_slm = can_use_slm;

    return dispatchData;
}

KernelsPriority FullyConnected_bf_tiled::GetKernelsPriority(const Params& params) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    size_t output_b = get_output_aligned_bf_size(fc_params, false).first;

    float estimated_time = FORCE_PRIORITY_9;
    if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F32)
        estimated_time = FORCE_PRIORITY_3;
    else if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F16)
        estimated_time = FORCE_PRIORITY_4;

    return estimated_time;
}

JitConstants FullyConnected_bf_tiled::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    size_t tile_k_ofm = dispatchData.tile_nk * dispatchData.tile_n;
    size_t tile_k_ofm_packed = tile_k_ofm;
    size_t quantize_grp_size = get_dynamic_quantize_group_size(params);

    bool add_decompress_scale_post_op = false;
    WeightsType weights_dt = params.weights.GetDType();
    if (weights_dt == WeightsType::UINT4 || weights_dt == WeightsType::INT4) {
        tile_k_ofm_packed /= 2;
        jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE", weights_dt, tile_k_ofm));
        const size_t scale_group_size = params.weights.IFM().v / params.decompression_scale.Feature().v;
        // Do not use SCALE_POST_OP for SLM kernel, since it demonstrates worse performance
        if (scale_group_size % simd == 0 && !dispatchData.use_slm)
            add_decompress_scale_post_op = true;
    }
    if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else {
        jit.AddConstant(MakeJitConstant("W_IDX", "kii * TILE_OFM + fi"));
    }

    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16 && dispatchData.tile_n == 2) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 0.5f));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2 && dispatchData.tile_n == 1) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 2));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2 && dispatchData.tile_n == 2) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 2));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 1));
    }

    jit.AddConstant(MakeJitConstant("W_DYN_QUAN_IDX", "fi * TILE_K + kii"));

    if (dispatchData.use_slm) {
        OPENVINO_ASSERT(dispatchData.tile_n == 2, "[GPU] Unsupported TILE_OFM size for SLM kernel configuration");
        OPENVINO_ASSERT(is_weight_dyn_quantizable(params), "[GPU] Unsupported FC weights type for SLM kernel configuration");

        auto lws_batches = dispatchData.lws[2];
        auto total_weights_elements = simd * dispatchData.tile_n * simd * dispatchData.tile_mk; // SIMD * TILE_OFM * SIMD * TILE_IFM
        auto weights_elements_per_sg = total_weights_elements / lws_batches;
        auto weights_elements_per_wi = weights_elements_per_sg / simd;
        auto weights_elements_per_wi_8bit = weights_elements_per_wi / 2; // number of pairs of int4 weights per WI

        auto block_read_size = 0;
        auto preferred_block_sizes = { 8, 4, 2 };

        for (auto block_size : preferred_block_sizes) {
            if (weights_elements_per_wi_8bit % block_size == 0) {
                block_read_size = block_size;
                break;
            }
        }

        OPENVINO_ASSERT(block_read_size != 0, "[GPU] Can't configure proper block size");

        auto weights_load_iters = weights_elements_per_wi_8bit / block_read_size;
        auto weights_elements_per_load = block_read_size * 2;

        jit.AddConstant(MakeJitConstant("USE_SLM", 1));
        jit.AddConstant(MakeJitConstant("LWS_BATCHES", lws_batches));
        jit.AddConstant(MakeJitConstant("FILTER_LOAD_ITERS", weights_load_iters));

        if (weights_dt == WeightsType::INT4 || weights_dt == WeightsType::UINT4) {
            if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
                jit.AddConstant(MakeJitConstant("FILTER_ACTUAL_LOAD_BLOCK_SIZE", block_read_size / 2));
                jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE_PRELOAD", params.weights.GetDType(), weights_elements_per_load / 2));
            } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
                jit.AddConstant(MakeJitConstant("FILTER_ACTUAL_LOAD_BLOCK_SIZE", block_read_size / 2));
                jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE_PRELOAD", params.weights.GetDType(), weights_elements_per_load / 2));
            } else {
                jit.AddConstant(MakeJitConstant("FILTER_ACTUAL_LOAD_BLOCK_SIZE", block_read_size));
                jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE_PRELOAD", params.weights.GetDType(), weights_elements_per_load));
            }
        } else {
            jit.AddConstant(MakeJitConstant("FILTER_ACTUAL_LOAD_BLOCK_SIZE", block_read_size));
        }

        jit.AddConstant(MakeJitConstant("FILTER_LOAD_BLOCK_SIZE", block_read_size));
        jit.AddConstant(MakeJitConstant("FILTER_ELEMENTS_PER_LOAD", weights_elements_per_load));
    } else {
        jit.AddConstant(MakeJitConstant("USE_SLM", 0));
    }

    // Validated perf gain, Dynamic quantize force enable SCALE_POST_OP for char type multiplication
    if (should_dynamic_quantize(params)) {
        jit.AddConstant(MakeJitConstant("DYNAMIC_QUANTIZE", 1));
        jit.AddConstant(MakeJitConstant("DQ_DECOMPRESSION_SCALE_POST_OP", 1));
        jit.AddConstant(MakeJitConstant("QUANTIZE_GROUP_SIZE", quantize_grp_size));
    } else {
        if (add_decompress_scale_post_op)
            jit.AddConstant(MakeJitConstant("DECOMPRESSION_SCALE_POST_OP", 1));
        jit.AddConstant(MakeJitConstant("DYNAMIC_QUANTIZE", 0));
        jit.AddConstant(MakeJitConstant("QUANTIZE_GROUP_SIZE", min_quantize_grp_size));
    }
    jit.AddConstant(MakeJitConstant("DQ_TYPE", "char"));

    jit.AddConstant(MakeJitConstant("IFM_SIZE", get_input_bf_size(params).second));
    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("TILE_B", dispatchData.tile_m));
    jit.AddConstant(MakeJitConstant("HALF_TILE_B", dispatchData.tile_m/2));
    jit.AddConstant(MakeJitConstant("TILE_OFM", dispatchData.tile_n));
    jit.AddConstant(MakeJitConstant("TILE_IFM", dispatchData.tile_mk));
    jit.AddConstant(MakeJitConstant("TILE_K", dispatchData.tile_nk));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM", tile_k_ofm));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM_PACKED", tile_k_ofm_packed));
    jit.AddConstant(MakeJitConstant("OUTER_OFM", dispatchData.outer_n));
    jit.AddConstant(MakeJitConstant("DISPATCH_BSV", dispatchData.tile_ms));
    jit.AddConstant(MakeJitConstant("DISPATCH_FSV", dispatchData.tile_ns));
    jit.AddConstant(MakeJitConstant("TILE_IFM_ELEMENTS_SIZE", (dispatchData.tile_mk * simd)));

    if (quantize_grp_size / (dispatchData.tile_mk * simd) > 1 && quantize_grp_size % (dispatchData.tile_mk * simd) == 0) {
        jit.AddConstant(MakeJitConstant("NUM_LOOP_IN_DYN_QUAN_GROUP", quantize_grp_size / (dispatchData.tile_mk * simd)));
    } else {
        jit.AddConstant(MakeJitConstant("NUM_LOOP_IN_DYN_QUAN_GROUP", 1));
    }

    auto max_tile_b_size = dispatchData.tile_m;
    if (params.compressed && params.is_shape_agnostic && is_weight_dyn_quantizable(params))
        max_tile_b_size = std::max(max_tile_b_size, (uint32_t)8);

    jit.Merge(MakeConstantLoopUnrollJitConstants(max_tile_b_size));

    bool realign_fp16_offset = params.inputs[0].GetDType() == Datatype::F16 && params.inputs[0].GetFirstElementOffset() % 2 != 0;
    jit.AddConstant(MakeJitConstant("REALIGN_FP16_OFFSET", realign_fp16_offset));

    auto activation_dt = GetActivationType(params);
    auto accumulator_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));

    // for 3d output we are treating spatial as features
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        auto tile_in_b_pitch = (params.inputs[0].Feature().pitch == 0) ? get_input_bf_size(params).second : params.inputs[0].Feature().pitch;
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", tile_in_b_pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        auto tile_in_b_pitch = (params.inputs[0].Batch().pitch == 0) ? get_input_bf_size(params).second : params.inputs[0].Batch().pitch;
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", tile_in_b_pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order_scalar = { "(out_b + bi)", "(out_f + sglid)", "0", "0" };
        std::vector<std::string> idx_order_vec = { "(out_b + bi)", "(out_f + sglid + fi * SIMD)", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
            idx_order_scalar = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "(out_f + sglid)", "0" };
            idx_order_vec = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "(out_f + sglid + fi * SIMD)", "0" };
        }

        // Simplify fused ops configuration to prevent mixed layout exception in jitter
        // for common cases with bfyx -> bf layouts and eltwise fusing (such scenarios currently don't work for vectors)
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              idx_order_scalar,
                                              "activated[bi]",
                                              activation_dt,
                                              1 };
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           idx_order_vec,
                                           "activated[bi][fi]",
                                           activation_dt,
                                           1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_scalar, conf_vec }));
    }

    return jit;
}

void FullyConnected_bf_tiled::GetUpdateDispatchDataFunc(KernelData& kd) const {
    if (kd.kernels.size() == 1) {
        Parent::GetUpdateDispatchDataFunc(kd);
    } else {
        kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
            const auto& prim_params = static_cast<const fully_connected_params&>(params);

            size_t quantize_grp_size = get_dynamic_quantize_group_size(prim_params);
            size_t output_batch = get_output_aligned_bf_size(prim_params, false).first;

            // Get index of the added shape-agnostic kernel
            int kernel_offset = 0;
            if (kd.kernels.size() == 3)
                kernel_offset = 1;  // quantize kernel exists

            // Choose one of the two shape agnostic kernels: N == added kernel number
            // - kd.kernels[N-1] for batches <= 240 (default version)
            // - kd.kernels[N] for batches >= 256 (slm version)
            const auto default_alignment = 16;
            // We can use SLM version if `output_batch + default_alignment > min_slm_size(256)` because memory and batch are aligned (whether 16 or 64 elements)
            const auto execute_type = (output_batch + default_alignment > min_slm_size) ? KernelType::SLM : KernelType::DEFAULT;
            const auto execute_kernel_idx = ((execute_type == KernelType::SLM) ? 1 : 0) + kernel_offset;
            const auto skip_kernel_idx = ((execute_type == KernelType::SLM) ? 0 : 1) + kernel_offset;


            // Check default or SLM version FC, and disable remain version
            kd.kernels[skip_kernel_idx].skip_execution = true;

            GPU_DEBUG_TRACE_DETAIL << "FC bf tiled: " << (execute_type == KernelType::SLM ? "SLM" : "Default") << " shape-agnostic kernel version "
                                    << "will be used for batch size = " << output_batch << "\n";

            auto dispatchData = SetDefault(prim_params, -1, static_cast<int>(execute_type));
            kd.kernels[execute_kernel_idx].params.workGroups.global = dispatchData.gws;
            kd.kernels[execute_kernel_idx].params.workGroups.local = dispatchData.lws;
            kd.kernels[execute_kernel_idx].skip_execution = KernelData::SkipKernelExecution(prim_params);

            auto& input = prim_params.inputs[0];
            if (prim_params.outputs[0].GetLayout() == DataLayout::bfyx)
                OPENVINO_ASSERT(input.X().pad.Total() == 0 && input.Y().pad.Total() == 0, "[GPU] Invalid padding in spatial axes observed in FC bf tiled.");
            else
                OPENVINO_ASSERT(input.Feature().pad.Total() == 0, "[GPU] Invalid padding in f axis observed in FC bf tiled.");

            if (!kd.internalBufferSizes.empty()) {
                // Pre-quantizing kernel was generated. Update the kernel and intermediate buffers or disable it.
                if (execute_type == KernelType::DEFAULT) {
                    kd.kernels[0].skip_execution = true;
                } else {
                    kd.kernels[0].skip_execution = false;
                    size_t input_f = get_input_bf_size(prim_params).second;
                    size_t input_size = input_f * dispatchData.tile_m * dispatchData.gws[2];

                    if (kd.internalBufferSizes[0] < input_size) {
                        kd.internalBufferSizes.clear();
                        // quantized input is char type
                        kd.internalBufferSizes.push_back(input_size);
                        // half type of de_quan_scale and activation sum for each quantized group
                        kd.internalBufferSizes.push_back((input_size / quantize_grp_size) * 2 * 2);
                    }

                    kd.kernels[0].params.workGroups.global = {std::max((input_size / quantize_grp_size), (size_t)1), 1, 1};
                    kd.kernels[0].params.workGroups.local = {16, 1, 1};
                }
            }
        };
    }
}

KernelsData FullyConnected_bf_tiled::GetTunedKernelsDataByIndex(const Params &params,
                                                                const int autoTuneIndex) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);

    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(auto_tune_params.size())
        && !TuneParamsSelector::VerifyTuneParams(fc_params, auto_tune_params[autoTuneIndex]))
        return {};

    tune_params tparams = GetAutoTuneParams(fc_params, KernelType::ANY, autoTuneIndex);
    auto output_f = get_output_aligned_bf_size(fc_params, false).second;

    WeightsLayout weights_layout = WeightsLayout::os_iyx_osv16;
    if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16
        && (fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2)
        && (fc_params.weights.GetDType() == WeightsType::INT4 || fc_params.weights.GetDType() == WeightsType::UINT4)
        && is_weight_horizontal(fc_params, output_f)) {
        // Large N + small K case (horizontal weight) to use [osv64_isv2] + TILE_OFM 4 for batch 1
        weights_layout = WeightsLayout::os_is_yx_osv64_isv2;
    } else if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16
        && (fc_params.weights.GetDType() == WeightsType::INT4 || fc_params.weights.GetDType() == WeightsType::UINT4)
        && (fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_iyx_osv16)
        && is_weight_vertical(fc_params, output_f)) {
        // Large K + Small N case (vertical weight)  to use [osv16 + TILE_K 4] + TILE_OFM 1 for batch 1
        weights_layout = WeightsLayout::os_iyx_osv16;
    } else if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16
        // ioyx => os_is_yx_osv32_isv2 is not supported yet
        && (fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2)
        && (fc_params.weights.GetDType() == WeightsType::INT4 || fc_params.weights.GetDType() == WeightsType::UINT4)) {
        weights_layout = WeightsLayout::os_is_yx_osv32_isv2;
    } else if (tparams.tile_ofm * simd == 32) {
        weights_layout = WeightsLayout::os_iyx_osv32;
    } else if (tparams.tile_ofm * simd == 64) {
        weights_layout = WeightsLayout::os_iyx_osv64;
    }

    KernelsData kernels_data;
    if (should_dynamic_quantize(fc_params)) {
        // Use seperate 2 kernels for dynamic quantizing : quantizing_kernel + fc_kernel
        // 1st kernel : Dynamic quantizing by dynamic_quantize_grp_size
        // 2nd kernel : fully connected kernel with KernelType::DEFAULT. Quantized inputs and scale values could be used.
        // 3rd kernel : (optional) fully connected shape_agnostic kernel with KernelType::SLM. Quantized inputs and scale values would be used.
        kernels_data = GetMultiKernelsData(params,
                                                fc_params.inputs[0].GetLayout(),
                                                weights_layout,
                                                tparams.exec_options,
                                                autoTuneIndex);
        OPENVINO_ASSERT(!kernels_data.empty() && !kernels_data[0].kernels.empty(), "[GPU] Error to create multi kernel for dynamic quantizing.");

        if (params.is_shape_agnostic)
            GetUpdateDispatchDataFunc(kernels_data[0]);
    } else {
        kernels_data = GetCommonKernelsData(params,
                                                fc_params.inputs[0].GetLayout(),
                                                weights_layout,
                                                tparams.exec_options,
                                                autoTuneIndex,
                                                0);

        if (params.is_shape_agnostic) {
            auto tparams = GetAutoTuneParams(fc_params, KernelType::SLM, autoTuneIndex);
            auto can_select_slm_kernel = tparams.kernel_type == KernelType::SLM;

            if (!can_select_slm_kernel)
                return kernels_data;

            auto slm_kernel = GetCommonKernelsData(params,
                                                fc_params.inputs[0].GetLayout(),
                                                weights_layout,
                                                tparams.exec_options,
                                                autoTuneIndex,
                                                1);

            if (slm_kernel.empty() || slm_kernel[0].kernels.empty())
                return kernels_data;

            kernels_data[0].kernels.push_back(slm_kernel[0].kernels.back());

            // Update default update_dispatch_data_func function
            GetUpdateDispatchDataFunc(kernels_data[0]);
        }
    }

    return kernels_data;
}

KernelsData FullyConnected_bf_tiled::GetKernelsDataForAutoTune(const Params& params) const {
    KernelsData res = {};
    for (size_t idx = 0; idx < auto_tune_params.size(); ++idx) {
        KernelsData kds = GetTunedKernelsDataByIndex(params, static_cast<int>(idx));

        if (!kds.empty()) {
            res.emplace_back(kds[0]);
        }
    }

    return res;
}

KernelsData FullyConnected_bf_tiled::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto tparams = GetAutoTuneParams(fc_params);

    KernelsData kds = GetTunedKernelsDataByIndex(params, -1);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}


KernelsData FullyConnected_bf_tiled::GetMultiKernelsData(const Params &params,
                                                           DataLayout dl,
                                                           WeightsLayout wl,
                                                           const std::string exeMode,
                                                           int autoTuneIndex) const {
    if (!Validate(params)) {
        return KernelsData();
    }

    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    size_t quantize_grp_size = get_dynamic_quantize_group_size(fc_params);

    bool bProperInput = fc_params.inputs[0].GetLayout() == dl;
    if (!bProperInput && !fc_params.inputs[0].PitchesDifferFromLogicalDims()) {
        bProperInput = (dl == DataLayout::fb && fc_params.inputs[0].GetLayout() == DataLayout::fyxb) ||
                       (dl == DataLayout::bf && fc_params.inputs[0].GetLayout() == DataLayout::bfyx);
    }

    KernelData kd = KernelData::Default<fully_connected_params>(params, 2);
    fully_connected_params& new_params = *static_cast<fully_connected_params*>(kd.params.get());

    if (!bProperInput) {
        new_params.inputs[0] = new_params.inputs[0].TransformIgnorePadding(dl);
        kd.reorderInput = true;
    }

    bool succeed = UpdateWeightsParams(new_params,
                                       wl,
                                       kd.weightsReorderParams,
                                       GetSupportedKey());
    if (!succeed) {
        return {};
    }

    int inputs_count = 1;
    if (new_params.compressed) {
        inputs_count++;
        if (new_params.has_decompression_zp && !new_params.scalar_zp)
            inputs_count++;
    }

    // Generate dispatch data for KernelType::DEFAULT
    int kernel_number = 0;
    const DispatchData dispatchData = SetDefault(new_params, autoTuneIndex, kernel_number);

    // Dynamic-quantize kernel
    {
        auto& quan_kernel = kd.kernels[0];
        DispatchData dyn_quan_dispatch = dispatchData;
        auto input_size = std::max(fc_params.inputs[0].PhysicalSize(), get_input_bf_size(fc_params).second);
        if (!params.is_shape_agnostic)
            input_size = std::max(input_size, Align(get_input_bf_size(fc_params).first, lws_batches) * get_input_bf_size(fc_params).second);
        dyn_quan_dispatch.gws = {input_size / quantize_grp_size, 1, 1};
        dyn_quan_dispatch.lws = {16, 1, 1};
        quan_kernel.params.workGroups.global = dyn_quan_dispatch.gws;
        quan_kernel.params.workGroups.local = dyn_quan_dispatch.lws;
        quan_kernel.skip_execution = false;

        auto quan_entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, kernel_number);
        auto quan_cldnn_jit = GetJitConstants(new_params, dyn_quan_dispatch);
        quan_cldnn_jit.AddConstant(MakeJitConstant("FC_KERNEL_DYNAMIC_QUANTIZE", 1));
        auto quan_jit = CreateJit(kernelName, quan_cldnn_jit, quan_entry_point);

        FillCLKernelData(quan_kernel,
                        dyn_quan_dispatch,
                        params.engineInfo,
                        kernelName,
                        quan_jit,
                        quan_entry_point,
                        exeMode,  // No exec mode
                        false,
                        false,
                        1, // Only INPUT_0 is used for quantizing
                        0, // No fused ops
                        0, // No output
                        fc_params.is_shape_agnostic);

        quan_kernel.params.arguments.clear();  // Clear original output argument
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        // char type quantized input
        kd.internalBufferSizes.push_back(input_size);
        // half type of de_quan_scale and activation sum for each quantized group
        kd.internalBufferSizes.push_back(input_size / quantize_grp_size * 2 * 2);
        kernel_number++;
    }
    kd.internalBufferDataType = Datatype::F16;

    // FC kernel for dynamic quantized input with KernelType::DEFAULT
    {
        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, kernel_number);
        auto cldnn_jit = GetJitConstants(new_params, dispatchData);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& fc_kernel = kd.kernels[1];
        fc_kernel.params.workGroups.global = dispatchData.gws;
        fc_kernel.params.workGroups.local = dispatchData.lws;
        fc_kernel.skip_execution = false;

        FillCLKernelData(fc_kernel,
                        dispatchData,
                        params.engineInfo,
                        kernelName,
                        jit,
                        entry_point,
                        exeMode,
                        true,
                        !fc_params.bias.empty(),
                        inputs_count,
                        GetFusedPrimitiveInputsCount(params),
                        1,
                        fc_params.is_shape_agnostic);

        fc_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        fc_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel_number++;
    }

    const DispatchData slm_Data = SetDefault(new_params, autoTuneIndex, kernel_number);
    auto slm_params = GetAutoTuneParams(fc_params, KernelType::SLM, autoTuneIndex);
    auto can_select_slm_kernel = slm_params.kernel_type == KernelType::SLM;
    // FC kernel for dynamic quantized input with KernelType::SLM
    if (params.is_shape_agnostic && can_select_slm_kernel) {
        kd.kernels.resize(kernel_number + 1);

        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, kernel_number);
        auto cldnn_jit = GetJitConstants(new_params, slm_Data);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& sa_kernel = kd.kernels[2];
        sa_kernel.params.workGroups.global = slm_Data.gws;
        sa_kernel.params.workGroups.local = slm_Data.lws;
        sa_kernel.skip_execution = false;

        FillCLKernelData(sa_kernel,
                        slm_Data,
                        params.engineInfo,
                        kernelName,
                        jit,
                        entry_point,
                        slm_params.exec_options,
                        true,
                        !fc_params.bias.empty(),
                        inputs_count,
                        GetFusedPrimitiveInputsCount(params),
                        1,
                        fc_params.is_shape_agnostic);

        sa_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        sa_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    }

    kd.autoTuneIndex = autoTuneIndex;
    return {kd};
}
}  // namespace kernel_selector
