// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_pil_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

enum AxisIndex {
    eVertical,
    eHorizontal
};

namespace {

Tensor::DataChannelName Convert(InterpolateAxis axis) {
    switch (axis) {
    case InterpolateAxis::BATCH:
        return Tensor::DataChannelName::BATCH;
    case InterpolateAxis::FEATURE:
        return Tensor::DataChannelName::FEATURE;
    case InterpolateAxis::W:
        return Tensor::DataChannelName::W;
    case InterpolateAxis::Z:
        return Tensor::DataChannelName::Z;
    case InterpolateAxis::Y:
        return Tensor::DataChannelName::Y;
    case InterpolateAxis::X:
        return Tensor::DataChannelName::X;
    default:
        throw std::invalid_argument("InterpolateAxis is out of range.");
    }
}

int ConvertToNCHW(InterpolateAxis axis) {
    switch (axis) {
    case InterpolateAxis::BATCH:
        return 0;
    case InterpolateAxis::FEATURE:
        return 1;
    case InterpolateAxis::Y:
        return 2;
    case InterpolateAxis::X:
        return 3;
    default:
        throw std::invalid_argument("InterpolateAxis is out of NCHW range.");
    }
}

Tensor::Dim ExtractDim(const DataTensor& tensor, InterpolateAxis axis) {
    return tensor.Extract(tensor.GetLayout(), Convert(axis), tensor.GetDims());
}

size_t getInputHorizontalSize(const resample_params& params, bool with_pad = false) {
    auto inputHorizontalSize = ExtractDim(params.inputs[0], params.axes[eHorizontal]).v;
    if (with_pad) {
        inputHorizontalSize += params.pads_begin[ConvertToNCHW(params.axes[eHorizontal])];
        inputHorizontalSize += params.pads_end[ConvertToNCHW(params.axes[eHorizontal])];
    }
    return inputHorizontalSize;
}

size_t getInputVerticalSize(const resample_params& params, bool with_pad = false) {
    auto inputVerticalSize = ExtractDim(params.inputs[0], params.axes[eVertical]).v;
    if (with_pad) {
        inputVerticalSize += params.pads_begin[ConvertToNCHW(params.axes[eVertical])];
        inputVerticalSize += params.pads_end[ConvertToNCHW(params.axes[eVertical])];
    }
    return inputVerticalSize;
}

size_t getOutputHorizontalSize(const resample_params& params) {
    return ExtractDim(params.outputs[0], params.axes[eHorizontal]).v;
}

size_t getOutputVerticalSize(const resample_params& params) {
    return ExtractDim(params.outputs[0], params.axes[eVertical]).v;
}

bool NeedHorizontalPass(const resample_params& params) {
    return getInputHorizontalSize(params, true) != getOutputHorizontalSize(params);
}

bool NeedVerticalPass(const resample_params& params) {
    return getInputVerticalSize(params, true) != getOutputVerticalSize(params);
}

std::size_t GetKernelsNum(const resample_params& params) {
    auto horizontal_kernels_num = NeedHorizontalPass(params) ? 2 : 0;
    auto vertical_kernels_num = NeedVerticalPass(params) ? 2 : 0;
    return horizontal_kernels_num + vertical_kernels_num;
}

std::size_t GetFirstRow(const resample_params& params) {
    float scale = static_cast<float>(getInputVerticalSize(params, true)) / getOutputVerticalSize(params);
    float filter_scale = std::max(1.f, scale);
    float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
    float center = 0.5 * scale;
    auto xmin = std::max(0, static_cast<int>(center - support + 0.5));
    return xmin;
}

std::size_t GetLastRow(const resample_params& params) {
    auto inputVerticalSize = getInputVerticalSize(params, true);
    auto outputVerticalSize = getOutputVerticalSize(params);
    float scale = static_cast<float>(inputVerticalSize) / outputVerticalSize;
    float filter_scale = std::max(1.f, scale);
    float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
    float center = (outputVerticalSize - 0.5) * scale;
    auto xmax = std::min(inputVerticalSize, static_cast<std::size_t>(center + support + 0.5));
    return xmax;
}

DataTensor GetIntermediateBufferSize(const resample_params& params) {
    auto& output = params.outputs[0];
    auto layout = output.GetLayout();
    auto ybox_first = GetFirstRow(params);
    auto ybox_last = GetLastRow(params);
    std::vector<size_t> dims = output.LogicalDims();
    auto channelIndex = DataTensor::Channelndex(layout, Convert(params.axes[eVertical]));
    dims[channelIndex] = ybox_last - ybox_first;
    DataTensor result{dims, output.GetDType(), layout};
    return result;
}

} // anonymous namespace

ParamsKey ResampleKernelPilRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::BILINEAR_PILLOW);
    k.EnableReampleType(ResampleType::BICUBIC_PILLOW);
    return k;
}

inline ResampleKernelPilRef::KernelId& operator++(ResampleKernelPilRef::KernelId &id) {
  using IntType = typename std::underlying_type<ResampleKernelPilRef::KernelId>::type;
  id = static_cast<ResampleKernelPilRef::KernelId>((static_cast<IntType>(id) + 1));
  return id;
}

ResampleKernelBase::DispatchData ResampleKernelPilRef::SetDefaultForKernel(KernelId id, const resample_params &arg) const {
    ResampleKernelBase::DispatchData dispatchData;
    switch (id) {
    case ResampleKernelPilRef::eCalcHorizontalCoefficients: {
        auto outputHorizontalSize = ExtractDim(arg.outputs[0], arg.axes[eHorizontal]).v;
        dispatchData.gws = std::vector<std::size_t>{outputHorizontalSize, 1, 1};
        dispatchData.lws = std::vector<std::size_t>{std::min(outputHorizontalSize, arg.engineInfo.maxWorkGroupSize), 1, 1};
        return dispatchData;
    }
    case ResampleKernelPilRef::eResampleHorizontal: {
        auto& output = NeedVerticalPass(arg) ? GetIntermediateBufferSize(arg) : arg.outputs[0];
        auto in_layout = arg.inputs[0].GetLayout();
        auto out_layout = arg.outputs[0].GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
            { Tensor::DataChannelName::Y },
            { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        dispatchData.gws = { output.X().v, output.Y().v, output.Feature().v * output.Batch().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
        return dispatchData;
    }
    case ResampleKernelPilRef::eCalcVerticalCoefficients: {
        auto outputVerticalSize = ExtractDim(arg.outputs[0], arg.axes[eVertical]).v;
        dispatchData.gws = std::vector<std::size_t>{outputVerticalSize, 1, 1};
        dispatchData.lws = std::vector<std::size_t>{std::min(outputVerticalSize, arg.engineInfo.maxWorkGroupSize), 1, 1};
        return dispatchData;
    }
    case ResampleKernelPilRef::eResampleVertical: {
        auto& output = arg.outputs[0];
        auto in_layout = arg.inputs[0].GetLayout();
        auto out_layout = arg.outputs[0].GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
            { Tensor::DataChannelName::Y },
            { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        dispatchData.gws = { output.X().v, output.Y().v, output.Feature().v * output.Batch().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
        return dispatchData;
    }
    default:
        throw std::invalid_argument("Kernel index is out of range. Kernel index for resample_pillow should be in range 0..3.");
    }
    return dispatchData;
}

static void SetKernelArguments(const resample_params& params, ResampleKernelPilRef::KernelId kernelId,
                               cldnn::arguments_desc& arguments,
                               std::vector<InternalBuffer>& internalBuffers) {
    /* maximum number of coeffs */
    switch (kernelId) {
    case ResampleKernelPilRef::eCalcHorizontalCoefficients: {
        auto inputHorizontalSizeWithPadding = getInputHorizontalSize(params, true);
        auto outputHorizontalSize = getOutputHorizontalSize(params);
        float scale = static_cast<float>(inputHorizontalSizeWithPadding) / outputHorizontalSize;
        float filter_scale = std::max(1.f, scale);
        float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
        int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // coefficients
        internalBuffers.push_back(outputHorizontalSize * ksize * sizeof(float));
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // bounds
        internalBuffers.push_back(outputHorizontalSize * 2 * sizeof(int));
        break;
    }
    case ResampleKernelPilRef::eResampleHorizontal: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});           // input image
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // coefficients
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // bounds
        if (NeedVerticalPass(params)) {
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2}); // output
            auto intermediateBufferTensor = GetIntermediateBufferSize(params);
            internalBuffers.push_back(intermediateBufferTensor.PhysicalSize() *
                BytesPerElement(intermediateBufferTensor.GetDType()));
        } else {
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0}); // output
        }
        break;
    }
    case ResampleKernelPilRef::eCalcVerticalCoefficients: {
        auto inputVerticalSizeWithPadding = getInputVerticalSize(params, true);
        auto outputVerticalSize = getOutputVerticalSize(params);
        float scale = static_cast<float>(inputVerticalSizeWithPadding) / outputVerticalSize;
        float filter_scale = std::max(1.f, scale);
        float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
        int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

        internalBuffers.push_back(outputVerticalSize * ksize * sizeof(float)); // coefficients
        internalBuffers.push_back(outputVerticalSize * 2 * sizeof(int));       // bounds
        if (NeedHorizontalPass(params)) {
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3}); // coefficients
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4}); // bounds
        } else {
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // coefficients
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // bounds
        }
        break;
    }
    case ResampleKernelPilRef::eResampleVertical: {
        if (NeedHorizontalPass(params)) {
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2}); // input image
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3}); // coefficients
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4}); // bounds
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0}); // output
        } else {
            arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});           // input image
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // coefficients
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // bounds
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});          // output
        }
        break;
    }
    default:
        throw std::invalid_argument("Kernel index is out of range. Kernel index for resample_pillow should be in range 0..3.");
    }
}

JitConstants ResampleKernelPilRef::GetJitConstantsForKernel(KernelId id, const resample_params& params) const {
    auto jit_constants = MakeBaseParamsJitConstants(params);
    jit_constants.AddConstants({
        MakeJitConstant("ENABLE_BILINEAR_PILLOW_MODE", params.resampleType == ResampleType::BILINEAR_PILLOW),
        MakeJitConstant("ENABLE_BICUBIC_PILLOW_MODE", params.resampleType == ResampleType::BICUBIC_PILLOW),
        MakeJitConstant("RESAMPLE_PILLOW_STAGE", static_cast<int>(id)),
        MakeJitConstant("STAGE_CALC_HORIZONTAL_COEFFICIENTS", static_cast<int>(eCalcHorizontalCoefficients)),
        MakeJitConstant("STAGE_RESAMPLE_HORIZONTAL", static_cast<int>(eResampleHorizontal)),
        MakeJitConstant("STAGE_CALC_VERTICAL_COEFFICIENTS", static_cast<int>(eCalcVerticalCoefficients)),
        MakeJitConstant("STAGE_RESAMPLE_VERTICAL", static_cast<int>(eResampleVertical)),
    });
    switch (id) {
        case eCalcHorizontalCoefficients: {
            auto inputHorizontalSizeWithPadding = getInputHorizontalSize(params, true);
            auto outputHorizontalSize = getOutputHorizontalSize(params);
            float scale = static_cast<float>(inputHorizontalSizeWithPadding) / outputHorizontalSize;
            float filter_scale = std::max(1.f, scale);
            float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
            int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
            jit_constants.AddConstants({MakeJitConstant("IN_DIM_BEGIN", 0.f),
                MakeJitConstant("IN_DIM_END", static_cast<float>(inputHorizontalSizeWithPadding)),
                MakeJitConstant("IN_DIM_SIZE", static_cast<int>(inputHorizontalSizeWithPadding)),
                MakeJitConstant("OUT_DIM_SIZE", static_cast<int>(outputHorizontalSize)),
                MakeJitConstant("SCALE", scale),
                MakeJitConstant("FILTER_SCALE", filter_scale),
                MakeJitConstant("SUPPORT", support),
                MakeJitConstant("KSIZE", ksize),
                MakeJitConstant("CUBE_COEFF", params.cube_coeff),
            });
            break;
        }
        case eResampleHorizontal: {
            auto inputHorizontalSizeWithPadding = getInputHorizontalSize(params, true);
            auto outputHorizontalSize = getOutputHorizontalSize(params);
            float scale = static_cast<float>(inputHorizontalSizeWithPadding) / outputHorizontalSize;
            float filter_scale = std::max(1.f, scale);
            float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
            int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
            auto ybox_first = GetFirstRow(params);
            jit_constants.AddConstants({MakeJitConstant("INTERMEDIATE_BUF", GetIntermediateBufferSize(params)),
                MakeJitConstant("BEGIN_PADDING_BATCH", params.pads_begin[0]),
                MakeJitConstant("BEGIN_PADDING_FEATURE", params.pads_begin[1]),
                MakeJitConstant("BEGIN_PADDING_Y", params.pads_begin[2]),
                MakeJitConstant("BEGIN_PADDING_X", params.pads_begin[3]),
                MakeJitConstant("BATCH_IS_HORIZONTAL_AXIS", params.axes[eHorizontal] == InterpolateAxis::BATCH),
                MakeJitConstant("FEATURE_IS_HORIZONTAL_AXIS", params.axes[eHorizontal] == InterpolateAxis::FEATURE),
                MakeJitConstant("Y_IS_HORIZONTAL_AXIS", params.axes[eHorizontal] == InterpolateAxis::Y),
                MakeJitConstant("X_IS_HORIZONTAL_AXIS", params.axes[eHorizontal] == InterpolateAxis::X),
                MakeJitConstant("BATCH_HORIZONTAL_OFFSET", params.axes[eHorizontal] == InterpolateAxis::BATCH ? ybox_first : 0),
                MakeJitConstant("FEATURE_HORIZONTAL_OFFSET", params.axes[eHorizontal] == InterpolateAxis::FEATURE ? ybox_first : 0),
                MakeJitConstant("Y_HORIZONTAL_OFFSET", params.axes[eHorizontal] == InterpolateAxis::Y ? ybox_first : 0),
                MakeJitConstant("X_HORIZONTAL_OFFSET", params.axes[eHorizontal] == InterpolateAxis::X ? ybox_first : 0),
                MakeJitConstant("ENABLE_VERTICAL_PASS", NeedVerticalPass(params)),
                MakeJitConstant("ENABLE_HORIZONTAL_PASS", NeedHorizontalPass(params)),
                MakeJitConstant("KSIZE", ksize),
            });
            break;
        }
        case eCalcVerticalCoefficients: {
            auto inputVerticalSizeWithPadding = getInputVerticalSize(params, true);
            auto outputVerticalSize = getOutputVerticalSize(params);
            float scale = static_cast<float>(inputVerticalSizeWithPadding) / outputVerticalSize;
            float filter_scale = std::max(1.f, scale);
            float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
            int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
            jit_constants.AddConstants({MakeJitConstant("IN_DIM_BEGIN", 0.f),
                MakeJitConstant("IN_DIM_END", static_cast<float>(inputVerticalSizeWithPadding)),
                MakeJitConstant("IN_DIM_SIZE", static_cast<int>(inputVerticalSizeWithPadding)),
                MakeJitConstant("OUT_DIM_SIZE", static_cast<int>(outputVerticalSize)),
                MakeJitConstant("SCALE", scale),
                MakeJitConstant("FILTER_SCALE", filter_scale),
                MakeJitConstant("SUPPORT", support),
                MakeJitConstant("KSIZE", ksize),
                MakeJitConstant("CUBE_COEFF", params.cube_coeff),
            });
            break;
        }
        case eResampleVertical: {
            auto inputVerticalSizeWithPadding = getInputVerticalSize(params, true);
            auto outputVerticalSize = getOutputVerticalSize(params);
            float scale = static_cast<float>(inputVerticalSizeWithPadding) / outputVerticalSize;
            float filter_scale = std::max(1.f, scale);
            float support = params.resampleType == ResampleType::BILINEAR_PILLOW ? 1.f : 2.f * filter_scale;
            int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
            jit_constants.AddConstants({MakeJitConstant("INTERMEDIATE_BUF", GetIntermediateBufferSize(params)),
                MakeJitConstant("BEGIN_PADDING_BATCH", params.pads_begin[0]),
                MakeJitConstant("BEGIN_PADDING_FEATURE", params.pads_begin[1]),
                MakeJitConstant("BEGIN_PADDING_Y", params.pads_begin[2]),
                MakeJitConstant("BEGIN_PADDING_X", params.pads_begin[3]),
                MakeJitConstant("BATCH_IS_VERTICAL_AXIS", params.axes[eVertical] == InterpolateAxis::BATCH),
                MakeJitConstant("FEATURE_IS_VERTICAL_AXIS", params.axes[eVertical] == InterpolateAxis::FEATURE),
                MakeJitConstant("Y_IS_VERTICAL_AXIS", params.axes[eVertical] == InterpolateAxis::Y),
                MakeJitConstant("X_IS_VERTICAL_AXIS", params.axes[eVertical] == InterpolateAxis::X),
                MakeJitConstant("ENABLE_VERTICAL_PASS", NeedVerticalPass(params)),
                MakeJitConstant("ENABLE_HORIZONTAL_PASS", NeedHorizontalPass(params)),
                MakeJitConstant("KSIZE", ksize),
            });
            break;
        }
        default:
            throw std::invalid_argument("Kernel index is out of range. Kernel index for resample_pillow should be in range 0..3.");
        }
    return jit_constants;
}

KernelsData ResampleKernelPilRef::GetKernelsData(const Params &params) const {
    const resample_params& resample_parameters = static_cast<const resample_params&>(params);
    KernelData kd = KernelData::Default<resample_params>(params, GetKernelsNum(resample_parameters));
    kd.internalBufferDataType = Datatype::F32;
    int i = 0;
    for (ResampleKernelPilRef::KernelId id = eCalcHorizontalCoefficients; id < eEnd; ++id) {
        if (!NeedHorizontalPass(resample_parameters) &&
            (id == eCalcHorizontalCoefficients || id == eResampleHorizontal))
            continue;
        if (!NeedVerticalPass(resample_parameters) &&
            (id == eCalcVerticalCoefficients || id == eResampleVertical))
            continue;
        auto& kernel = kd.kernels[i++];
        const auto entryPoint = GetEntryPoint(kernelName, resample_parameters.layerID, params, i);
        auto jitConstants = GetJitConstantsForKernel(id, resample_parameters);
        const auto jit = CreateJit(kernelName, jitConstants, entryPoint);
        const auto dispatchData = SetDefaultForKernel(id, resample_parameters);
        FillCLKernelData(kernel,
                         dispatchData,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entryPoint,
                         "",
                         false,
                         false,
                         0,
                         0,
                         0);
        SetKernelArguments(resample_parameters, id, kernel.params.arguments, kd.internalBuffers);
    }
    return {kd};
}

} // namespace kernel_selector
