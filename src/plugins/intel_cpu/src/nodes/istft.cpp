// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft.h"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/istft.hpp"

namespace ov::intel_cpu::node {

bool ISTFT::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v16::ISTFT::get_type_info_static()) {
            errorMessage = "Only ISTFT operation from the opset16 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ISTFT::ISTFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto istft_op = as_type_ptr<op::v16::ISTFT>(op);

    m_is_frame_size_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(FRAME_SIZE_IDX));
    m_is_frame_step_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(FRAME_STEP_IDX));
    if (istft_op->get_input_size() > SIGNAL_LENGTH_IDX) {
        m_has_signal_length_input = true;
        m_is_signal_length_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(SIGNAL_LENGTH_IDX));
    }
    m_center = istft_op->get_center();
    m_normalized = istft_op->get_normalized();
}

void ISTFT::getSupportedDescriptors() {
    const auto input_size = getParentEdges().size();
    if (input_size < 4 || input_size > 5) {
        THROW_CPU_NODE_ERR("ISTFT has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("ISTFT has incorrect number of output edges.");
    }
}

void ISTFT::initSupportedPrimitiveDescriptors() {
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
    if (m_has_signal_length_input) {
        configurators.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, dataPrecision}}, impl_desc_type::ref_any);
}

bool ISTFT::needPrepareParams() const {
    return false;
}

bool ISTFT::created() const {
    return getType() == Type::ISTFT;
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

void istft_impl(const float* in_data,
                const float* window,
                float* final_result,
                const ov::Shape& data_shape,
                const ov::Shape& window_shape,
                const int64_t frame_size,
                const int64_t frame_step,
                const int64_t length,
                const bool center,
                const bool normalized,
                std::shared_ptr<RDFTExecutor> rdft_executor) {
    const auto is_data_3D = data_shape.size() == 3;
    const size_t frames_axis = 1 + (is_data_3D ? 0 : 1);
    const size_t batch_size = is_data_3D ? 1 : data_shape[0];

    const auto sqrt_frame_size = static_cast<float>(std::sqrt(frame_size));
    const auto num_frames = data_shape[frames_axis];

    const auto signal_length = (num_frames - 1) * frame_step + frame_size;
    const int64_t final_signal_length =
        length > 0 ? length : (center ? (signal_length - (frame_size & ~1)) : signal_length);
    std::fill(final_result, final_result + batch_size * final_signal_length, 0.f);

    std::vector<float> mid_result(batch_size * signal_length, 0.f);

    const auto fft_results_dim = data_shape[data_shape.size() - 3];
    OPENVINO_ASSERT(fft_results_dim == static_cast<size_t>((frame_size / 2) + 1));

    const auto frame_size_dim = static_cast<size_t>(frame_size);
    const auto fft_out_shape = ov::Shape{fft_results_dim, 2};

    const auto window_length = window_shape[0] < frame_size_dim ? window_shape[0] : frame_size_dim;
    std::vector<float> pad_window(frame_size, 0);
    std::copy(window, window + window_shape[0], pad_window.begin() + (frame_size_dim - window_length) / 2);
    std::vector<float> pow_window(frame_size, 0);
    std::transform(pad_window.begin(), pad_window.end(), pow_window.begin(), [](float win_val) {
        return win_val * win_val;
    });

    std::vector<float> data_t(shape_size(data_shape));
    const auto stft_transp_out_shape = ov::Shape{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]};
    transpose_out4d(reinterpret_cast<const uint8_t*>(in_data),
                    reinterpret_cast<uint8_t*>(data_t.data()),
                    ov::Shape{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]},
                    stft_transp_out_shape,
                    sizeof(float));

    // Setting function for the result postprocessing
    const auto norm_window_div = [sqrt_frame_size](float a, float b) {
        if (b != 0.f) {
            return (a * sqrt_frame_size) / b;
        }
        return 0.f;
    };
    const auto window_div = [](float a, float b) {
        if (b != 0.f) {
            return a / b;
        }
        return 0.f;
    };
    std::function<float(float, float)> postprocess_func;
    if (normalized) {
        postprocess_func = norm_window_div;
    } else {
        postprocess_func = window_div;
    }

    const auto fft_out_shape_size = shape_size(fft_out_shape);
    const auto in_batch_single_step = num_frames * fft_out_shape_size;
    const int64_t margin = center ? (frame_size / 2) : 0;
    const int64_t data_end = signal_length - margin;
    const int64_t copy_end = final_signal_length < data_end ? final_signal_length : data_end;

    std::vector<float> window_sum(batch_size * signal_length);
    auto twiddles = rdft_executor->generateTwiddles({static_cast<int>(frame_size)}, {frame_size_dim}, {0});

    parallel_for(batch_size, [&](size_t batch) {
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            size_t batch_in_start = batch * in_batch_single_step;
            size_t batch_out_start = batch * signal_length;

            const auto in_frame_start = batch_in_start + frame_idx * fft_out_shape_size;
            const auto out_frame_start = batch_out_start + frame_idx * frame_step;

            std::vector<float> frame_signal(frame_size);
            rdft_executor->execute(data_t.data() + in_frame_start,
                                   frame_signal.data(),
                                   twiddles,
                                   1,
                                   {0},
                                   {static_cast<int>(frame_size)},
                                   {frame_size_dim},
                                   {frame_size_dim},
                                   {1},
                                   {1});

            // Overlap Add
            float* mid_result_sum = mid_result.data() + out_frame_start;
            float* window_frame_sum = window_sum.data() + out_frame_start;
            for (size_t i = 0; i < frame_signal.size(); ++i) {
                mid_result_sum[i] += frame_signal[i] * pad_window[i];
                window_frame_sum[i] += pow_window[i];
            }
        }
        float* result = mid_result.data() + (batch * signal_length);
        std::transform(result,
                       result + signal_length,
                       window_sum.begin() + batch * signal_length,
                       result,
                       postprocess_func);
        const auto result_start = result + margin;
        std::copy(result_start, result_start + copy_end, final_result + batch * final_signal_length);
    });
}
}  // namespace

void ISTFT::execute(const dnnl::stream& strm) {
    const auto signal_length =
        m_has_signal_length_input ? (getSrcDataAtPortAs<const int32_t>(SIGNAL_LENGTH_IDX))[0] : -1;
    istft_impl(getSrcDataAtPortAs<const float>(DATA_IDX),
               getSrcDataAtPortAs<const float>(WINDOW_IDX),
               getDstDataAtPortAs<float>(0),
               ov::Shape{getSrcMemoryAtPort(DATA_IDX)->getStaticDims()},
               ov::Shape{getSrcMemoryAtPort(WINDOW_IDX)->getStaticDims()},
               (getSrcDataAtPortAs<const int32_t>(FRAME_SIZE_IDX))[0],
               (getSrcDataAtPortAs<const int32_t>(FRAME_STEP_IDX))[0],
               signal_length,
               m_center,
               m_normalized,
               rdft_executor);
}

void ISTFT::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool ISTFT::needShapeInfer() const {
    return (m_has_signal_length_input && !m_is_signal_length_const) ||
           (!m_has_signal_length_input && !(m_is_frame_size_const && m_is_frame_step_const)) || Node::needShapeInfer();
}

void ISTFT::createPrimitive() {
    RDFTKey key{};
    key.isInverse = true;
    auto buildExecutor = [&](const RDFTKey& key) -> std::shared_ptr<RDFTExecutor> {
        return RDFTExecutor::build(key.isInverse, getSelectedPrimitiveDescriptor());
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    rdft_executor = result.first;

    Node::createPrimitive();
}

}  // namespace ov::intel_cpu::node
