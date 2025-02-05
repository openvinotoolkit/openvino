// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft.h"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/reference/stft.hpp"

namespace ov::intel_cpu::node {

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
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto stft_op = as_type_ptr<op::v15::STFT>(op);
    m_transpose_frames = stft_op->get_transpose_frames();

    m_is_frame_size_const = is_type<op::v0::Constant>(stft_op->get_input_node_ptr(FRAME_SIZE_IDX));
    m_is_frame_step_const = is_type<op::v0::Constant>(stft_op->get_input_node_ptr(FRAME_STEP_IDX));
}

void STFT::getSupportedDescriptors() {
    if (getParentEdges().size() != 4) {
        THROW_CPU_NODE_ERR("STFT has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("STFT has incorrect number of output edges.");
    }
}

void STFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto dataPrecision = getOriginalInputPrecisionAtPort(DATA_IDX);
    if (!one_of(dataPrecision, ov::element::f32)) {
        dataPrecision = ov::element::f32;
    }

    std::vector<PortConfigurator> configurators({{LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, ov::element::i32},
                                                 {LayoutType::ncsp, ov::element::i32}});

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, dataPrecision}}, impl_desc_type::ref_any);
}

bool STFT::needPrepareParams() const {
    return false;
}

bool STFT::created() const {
    return getType() == Type::STFT;
}

namespace {
static void transpose_out4d(const uint8_t* in,
                            uint8_t* out,
                            const VectorDims& in_shape,
                            const VectorDims& out_shape,
                            size_t elem_size) {
    const std::vector<size_t> axes_order{0, 2, 1, 3};
    parallel_for3d(out_shape[0],
                   out_shape[1],
                   out_shape[2],
                   [in, out, axes_order, &in_shape, &out_shape, elem_size](size_t i, size_t j, size_t k) {
                       size_t in_indexes[3];
                       in_indexes[axes_order[0]] = i;
                       in_indexes[axes_order[1]] = j;
                       in_indexes[axes_order[2]] = k;
                       size_t in_off =
                           ((in_indexes[0] * in_shape[1] + in_indexes[1]) * in_shape[2] + in_indexes[2]) * in_shape[3];
                       size_t out_off = ((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3];
                       cpu_memcpy(out + out_off * elem_size, in + in_off * elem_size, out_shape[3] * elem_size);
                   });
}
}  // namespace

void STFT::execute(const dnnl::stream& strm) {
    const auto* signal = getSrcDataAtPortAs<const float>(DATA_IDX);
    const auto* window = getSrcDataAtPortAs<const float>(WINDOW_IDX);
    auto* rdft_result = getDstDataAtPortAs<float>(0);
    const VectorDims& signal_shape = getSrcMemoryAtPort(DATA_IDX)->getStaticDims();
    const VectorDims& window_shape = getSrcMemoryAtPort(WINDOW_IDX)->getStaticDims();
    const int64_t frame_size = (getSrcDataAtPortAs<const int32_t>(FRAME_SIZE_IDX))[0];
    const int64_t frame_step = (getSrcDataAtPortAs<const int32_t>(FRAME_STEP_IDX))[0];

    const auto is_signal_1D = signal_shape.size() == 1;
    const size_t batch_size = is_signal_1D ? 1 : signal_shape[0];
    const size_t signal_axis = is_signal_1D ? 0 : 1;
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

    float* dst = rdft_result;
    const auto stft_shape = VectorDims{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]};
    if (m_transpose_frames) {  // Store intermediate results
        MemoryPtr dst_mem =
            getScratchPadMem(std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{stft_shape}));
        dst = dst_mem->getDataAs<float>();
    }

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
                       std::multiplies<>());

        const auto result_idx = (batch_frames_out + frame_idx) * fft_out_shape_size;
        auto twiddles = rdft_executor->generateTwiddles({static_cast<int>(signal_slice.size())}, fft_out_shape, {0});
        rdft_executor->execute(signal_slice.data(),
                               dst + result_idx,
                               twiddles,
                               1,
                               {0},
                               {static_cast<int>(frame_size)},
                               {frame_size_dim},
                               fft_out_shape,
                               {1},
                               {2, 1});
    });
    if (m_transpose_frames) {
        const auto stft_transp_out_shape = VectorDims{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]};
        transpose_out4d(reinterpret_cast<const uint8_t*>(dst),
                        reinterpret_cast<uint8_t*>(rdft_result),
                        stft_shape,
                        stft_transp_out_shape,
                        sizeof(float));
    }
}

void STFT::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool STFT::needShapeInfer() const {
    return !(m_is_frame_size_const && m_is_frame_step_const) || Node::needShapeInfer();
}

void STFT::createPrimitive() {
    RDFTKey key{};
    key.isInverse = false;
    auto buildExecutor = [&](const RDFTKey& key) -> std::shared_ptr<RDFTExecutor> {
        return RDFTExecutor::build(key.isInverse, getSelectedPrimitiveDescriptor());
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    rdft_executor = result.first;

    Node::createPrimitive();
}

}  // namespace ov::intel_cpu::node
