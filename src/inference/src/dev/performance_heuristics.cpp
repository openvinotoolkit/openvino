// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/performance_heuristics.hpp"

namespace ov {

MemBandwidthPressure mem_bandwidth_pressure_tolerance(const std::shared_ptr<ov::Model> model,
                                                      const float cache_size,
                                                      const float mem_threshold_assume_limited,
                                                      const ov::element::Type& target_type) {
    int total_convs = 0, mem_limited_convs = 0, compute_convs = 0, total_gemms = 0, mem_limited_gemms = 0,
        total_deconvs = 0, compute_deconvs = 0, mem_limited_deconvs = 0, total_adds = 0, mem_limited_adds = 0,
        total_G_T = 0, total_nodes = 0, total_lstms = 0, total_loops = 0;
    std::vector<int> conv_list(10,0);
    std::vector<int> gemm_list(10,0);
    std::vector<int> add_list(10,0);
    auto memLimitedFactor = [&](size_t size_data_moved, int datatype_size = 4) -> float {
        return (cache_size / (size_data_moved * datatype_size));
    };
    auto isLowPrecision = [&](ov::element::Type type) -> bool {
        return (type == ov::element::i8) || (type == ov::element::u8);
    };
    auto isHalfPrecision = [&](ov::element::Type type) -> bool {
        return (type == ov::element::bf16) || (type == ov::element::f16);
    };

    float worst_case = MemBandwidthPressure::UNKNOWN;
    // Traverse OpenVINO Model in topological order
    for (auto& node : model->get_ordered_ops()) {
        const auto node_name = node->get_type_info().name;

        total_nodes++;

        if (std::strcmp("MatMul", node_name) && std::strcmp("Convolution", node_name) &&
            std::strcmp("LSTMSequence", node_name) && std::strcmp("Loop", node_name) && std::strcmp("Add", node_name) &&
            std::strcmp("ConvolutionBackpropData", node_name)) {
            if (!std::strcmp("GRUSequence", node_name) || !std::strcmp("TensorIterator", node_name)) {
                total_G_T++;
                MemBandwidthPressure res;
                res.max_mem_tolerance = MemBandwidthPressure::UNKNOWN;
                return res;
            }
            continue;
        }
        const auto& in1_type = node->get_input_element_type(1);  // weights
        const auto& type1 =
            ((in1_type == ov::element::f32) && (target_type != ov::element::f32)) ? target_type : in1_type;
        const bool isINT8 = isLowPrecision(type1);
        const bool isBF16orFP16 = isHalfPrecision(type1);
        const int data_type_size = isINT8 ? 1 : isBF16orFP16 ? 2 : 4;

        size_t dataSizeInput = 0, dataSizeOutput = 0;
        if (!std::strcmp("MatMul", node_name)) {
            const auto input0 = node->input(0);
            const auto input1 = node->input(1);
            const auto output = node->output(0);
            // Check that input and output shape a fully defined (not dynamic)
            if (input0.get_partial_shape().is_static() && input1.get_partial_shape().is_static() &&
                output.get_partial_shape().is_static()) {
                const auto& shapeInput0 = input0.get_shape();
                const auto& shapeInput1 = input1.get_shape();
                const auto non_const = !ov::op::util::is_on_constant_path(node->input_value(1));
                const auto& shapeOutput = output.get_shape();
                const auto dataSizeInput0 =
                    std::accumulate(shapeInput0.begin(), shapeInput0.end(), size_t(1), std::multiplies<size_t>());
                const auto dataSizeInput1 =
                    std::accumulate(shapeInput1.begin(), shapeInput1.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto total_data = dataSizeInput0 + non_const * dataSizeInput1 + dataSizeOutput;
                total_gemms++;
                const auto factor = memLimitedFactor(total_data, data_type_size);
                mem_limited_gemms += factor < mem_threshold_assume_limited;
                worst_case = std::min(factor, worst_case);
                const auto gemm_indicator = dataSizeOutput * shapeInput1[2] * data_type_size;
                const int base_threshold = 16 * 6 * 49 * 49 / 8;
                for (int n = 9; n > 0; n--) {
                    if (gemm_indicator > base_threshold * (2^n)) {
                        gemm_list[n - 1]++;
                        break;
                    }
                    if(n == 1) {
                        gemm_list[0]++;
                        break;
                    }
                }
            }
        } else if (!std::strcmp("Convolution", node_name)) {
            // Check that input and output shape a fully defined (not dynamic)
            const auto input = node->input(0);
            const auto output = node->output(0);
            const auto kernels = node->input(1);

            total_convs++;
            if (kernels.get_partial_shape().is_static()) {
                const auto& shape = kernels.get_shape();
                if (shape.size() >= 4 /* conventional 2D/3D conv */ && shape[2] >= 3 && shape[3] >= 3) {
                    compute_convs++;
                    continue;
                }
            }
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static()) {
                const auto& shapeInput = input.get_shape();
                const auto& shapeOutput = output.get_shape();
                const auto& shapeInput1 = kernels.get_shape();
                if (shapeInput.size() > 4 /*5D*/ && isINT8) {
                    compute_convs++;
                    continue;
                }
                dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto factor = memLimitedFactor(static_cast<int>(dataSizeInput + dataSizeOutput), data_type_size);
                mem_limited_convs += factor < mem_threshold_assume_limited;
                worst_case = std::min(factor, worst_case);
                auto conv_indicator = dataSizeOutput * data_type_size;
                for (size_t n = 1; n < shapeInput1.size(); n++) {
                    conv_indicator = conv_indicator * shapeInput1[n];
                }
                const int base_threshold = 44 * 1056 / 8;
                for (int n = 9; n > 0; n--) {
                    if (conv_indicator > base_threshold * (2^n)) {
                        conv_list[n - 1]++;
                        break;
                    }
                    if(n == 1) {
                        conv_list[0]++;
                        break;
                    }
                }
            }
        } else if (!std::strcmp("ConvolutionBackpropData", node_name)) {
            const auto input = node->input(0);
            const auto output = node->output(0);
            total_deconvs++;

            // Check that input and output shape a fully defined (not dynamic)
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static()) {
                const auto& shapeInput = input.get_shape();
                const auto& shapeOutput = output.get_shape();
                if (shapeInput.size() > 4 /*5D*/ && isINT8) {
                    compute_deconvs++;
                    continue;
                }
                dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto factor = memLimitedFactor(static_cast<int>(dataSizeInput + dataSizeOutput), data_type_size);
                mem_limited_deconvs += factor < mem_threshold_assume_limited;
                worst_case = std::min(factor, worst_case);
            }
        } else if (!std::strcmp("Add", node_name)) {
            const auto input = node->input(0);
            const auto output = node->output(0);
            // Check that input and output shape a fully defined (not dynamic)
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static()) {
                const auto& shapeInput = input.get_shape();
                const auto& shapeOutput = output.get_shape();
                total_adds++;
                dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto factor = memLimitedFactor(static_cast<int>(dataSizeInput + dataSizeOutput), data_type_size);
                mem_limited_adds += factor < mem_threshold_assume_limited;
                const auto add_indicator = dataSizeOutput * data_type_size;
                const int base_threshold = 100 * 256 / 8;
                for (int n = 9; n > 0; n--) {
                    if (add_indicator > base_threshold * (2^n)) {
                        add_list[n - 1]++;
                        break;
                    }
                    if(n == 1) {
                        add_list[0]++;
                        break;
                    }
                }
            }
        } else if (!std::strcmp("LSTMSequence", node_name)) {
            total_lstms++;
        } else if (!std::strcmp("Loop", node_name)) {
            total_loops++;
        }
    }
    MemBandwidthPressure res;
    res.max_mem_tolerance = worst_case;
    res.ratio_mem_limited_convs = total_convs ? static_cast<float>(mem_limited_convs) / total_convs : 0;
    res.ratio_mem_limited_deconvs = total_deconvs ? static_cast<float>(mem_limited_deconvs) / total_deconvs : 0;
    res.ratio_mem_limited_gemms = total_gemms ? static_cast<float>(mem_limited_gemms) / total_gemms : 0;
    res.ratio_mem_limited_adds = total_adds ? static_cast<float>(mem_limited_adds) / total_adds : 0;
    res.ratio_compute_convs = total_convs ? static_cast<float>(compute_convs) / total_convs : 0;
    res.ratio_compute_deconvs = total_deconvs ? static_cast<float>(compute_deconvs) / total_deconvs : 0;
    res.total_gemms = total_gemms;
    res.total_convs = total_convs;
    res.total_adds = total_adds;
    res.total_lstms = total_lstms;
    res.total_loops = total_loops;
    res.total_nodes = total_nodes;
    res.total_G_T = total_G_T;
    res.add_list = add_list;
    res.gemm_list = gemm_list;
    res.conv_list = conv_list;
    return res;
}

}  // namespace ov
