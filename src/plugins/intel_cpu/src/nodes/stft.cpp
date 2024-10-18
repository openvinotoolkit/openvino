// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft.h"

#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/reference/stft.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool STFT::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v15::STFT::get_type_info_static()) {
            errorMessage = "Only STFT operation from the opset15 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

STFT::STFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(2, 3))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto STFT_op = as_type_ptr<op::v15::STFT>(op);
    m_transpose_frames = STFT_op->get_transpose_frames();

    rdft_executor = std::make_shared<RDFTRefExecutor>(false);
}

void STFT::getSupportedDescriptors() {
    if (getParentEdges().size() != 4) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void STFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(DATA_IDX);
    if (!dataPrecision.is_real()) {
        OPENVINO_THROW("STFT has unsupported 'data' input precision: ", dataPrecision.get_type_name());
    }

    std::vector<PortConfigurator> configurators({{LayoutType::ncsp, ov::element::f32},
                                                 {LayoutType::ncsp, ov::element::f32},
                                                 {LayoutType::ncsp, ov::element::i32},
                                                 {LayoutType::ncsp, ov::element::i32}});

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, ov::element::f32}}, impl_desc_type::ref_any);
}

void STFT::prepareParams() {
    const auto& input_shape = getParentEdgeAt(DATA_IDX)->getMemory().getStaticDims();
    if (input_shape.size() < 2) {
        THROW_CPU_NODE_ERR("has incompatible 'data' shape ",
                           PartialShape(input_shape),
                           ". Only tensors of rank at least 2 are allowed.");
    }
}

bool STFT::created() const {
    return getType() == Type::STFT;
}

void transpose_out(const char* in,
                   char* out,
                   const VectorDims& in_shape,
                   const std::vector<size_t>& axes_order,
                   const VectorDims& out_shape,
                   size_t elem_size) {
    parallel_for4d(out_shape[0],
                   out_shape[1],
                   out_shape[2],
                   out_shape[3],
                   [in, out, axes_order, &in_shape, &out_shape, elem_size](size_t i, size_t j, size_t k, size_t l) {
                       size_t in_indexes[4];
                       in_indexes[axes_order[0]] = i;
                       in_indexes[axes_order[1]] = j;
                       in_indexes[axes_order[2]] = k;
                       in_indexes[axes_order[3]] = l;
                       size_t in_off =
                           ((in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2]) * in_shape[3] +
                           in_indexes[3];
                       size_t out_off = ((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + l;
                       cpu_memcpy(out + out_off * elem_size, in + in_off * elem_size, elem_size);
                   });
}

void stft_impl(const float* signal,
               const float* window,
               float* rdft_result,
               const VectorDims& signal_shape,
               const VectorDims& window_shape,
               const int64_t frame_size,
               const int64_t frame_step,
               const bool transpose_frames,
               std::shared_ptr<RDFTExecutor> rdft_executor) {
    constexpr size_t signal_axis = 1;
    const auto batch_size = signal_shape[0];
    const auto signal_length = signal_shape[signal_axis];
    const auto num_frames = static_cast<size_t>((signal_length - frame_size) / frame_step) + 1;
    const auto frame_size_dim = static_cast<size_t>(frame_size);
    const auto fft_out_shape = VectorDims{static_cast<size_t>((frame_size_dim / 2) + 1), 2};
    const auto fft_out_shape_size = shape_size(fft_out_shape);

    const auto window_length = window_shape[0] < frame_size_dim ? window_shape[0] : frame_size_dim;
    std::vector<float> pad_window(frame_size, 0);
    cpu_parallel_memcpy(pad_window.data() + (frame_size_dim - window_length) / 2,
                        window,
                        sizeof(float) * window_shape[0]);

    parallel_for2d(batch_size, num_frames, [&](size_t batch, size_t frame_idx) {
        size_t batch_in_start = batch * signal_length;
        size_t batch_frames_out = batch * num_frames;

        const auto frame_start = batch_in_start + frame_idx * frame_step;
        const auto frame_end = frame_start + frame_size;
        std::vector<float> signal_slice(signal + frame_start, signal + frame_end);
        std::transform(signal_slice.begin(),
                       signal_slice.end(),
                       pad_window.begin(),
                       signal_slice.begin(),
                       std::multiplies<float>());

        const auto result_idx = (batch_frames_out + frame_idx) * fft_out_shape_size;
        auto twiddles = rdft_executor->generateTwiddles({static_cast<int>(signal_slice.size())}, fft_out_shape, {0});
        rdft_executor->execute(signal_slice.data(),
                               rdft_result + result_idx,
                               twiddles,
                               1,
                               {0},
                               {frame_size_dim},
                               {frame_size_dim},
                               fft_out_shape,
                               {1},
                               {2, 1});
    });
    if (transpose_frames) {
        const auto stft_shape = VectorDims{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]};
        const auto stft_transp_out_shape = VectorDims{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]};
        std::vector<float> signal_t(rdft_result, rdft_result + shape_size(stft_transp_out_shape));
        transpose_out(reinterpret_cast<const char*>(signal_t.data()),
                      reinterpret_cast<char*>(rdft_result),
                      stft_shape,
                      {0, 2, 1, 3},
                      stft_transp_out_shape,
                      sizeof(float));
    }
}

void STFT::execute(dnnl::stream strm) {
    stft_impl(getSrcDataAtPortAs<const float>(DATA_IDX),
              getSrcDataAtPortAs<const float>(WINDOW_IDX),
              getDstDataAtPortAs<float>(0),
              getSrcMemoryAtPort(DATA_IDX)->getStaticDims(),
              getSrcMemoryAtPort(WINDOW_IDX)->getStaticDims(),
              (getSrcDataAtPortAs<const int32_t>(FRAME_SIZE_IDX))[0],
              (getSrcDataAtPortAs<const int32_t>(FRAME_STEP_IDX))[0],
              m_transpose_frames,
              rdft_executor);
}

void STFT::executeDynamicImpl(dnnl::stream strm) {
    auto result = shapeInfer();
    if (ShapeInferStatus::success == result.status) {
        redefineOutputMemory(result.dims);
    }
    execute(strm);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
