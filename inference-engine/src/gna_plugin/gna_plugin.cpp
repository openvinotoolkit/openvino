// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>
#include <list>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <utility>
#include <limits>

#include <legacy/graph_tools.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <legacy/net_pass.h>
#include <debug.h>
#include <gna/gna_config.hpp>
#include "gna_plugin_config.hpp"
#include "gna_plugin.hpp"
#include "optimizer/gna_pass_manager.hpp"
#include "layers/gna_layer_type.hpp"
#include "preprocessing.hpp"
#include "frontend/weights_converter.hpp"
#include "frontend/model_quantizer.hpp"
#include "gna_fused_iterator.hpp"
#include "backend/am_intel_dnn.hpp"
#include "memory/gna_allocator.hpp"
#include "memory/gna_memory_state.hpp"
#include "gna_model_serial.hpp"
#include "runtime/gna_float_runtime.hpp"
#include <layers/gna_fake_quantize_layer.hpp>
#include "gna_graph_patterns.hpp"
#include "gna_tensor_tools.hpp"

#include <ngraph/pass/manager.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>

uint32_t ToByteSize(const Gna2DataType type) {
    switch (type) {
    case Gna2DataTypeInt8:
    case Gna2DataTypeUint8:
        return 1;
    case Gna2DataTypeInt16:
    case Gna2DataTypeUint16:
        return 2;
    case Gna2DataTypeInt32:
    case Gna2DataTypeUint32:
        return 4;
    case Gna2DataTypeInt64:
    case Gna2DataTypeUint64:
        return 8;
    default:
        return 0;
    }
}

constexpr uint32_t GNAPluginNS::GNAPlugin::FAKE_REQUEST_CONFIG_ID;
#endif
using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;
using namespace InferenceEngine::details;

#ifdef __clang__
namespace InferenceEngine {
    template<>
    InferenceEngine::TBlob<intel_compound_bias_t, std::enable_if<true, void> >::~TBlob() { free(); }
}
#endif  // __clang__

template <typename T, typename U>
void GNAPlugin::copyInputData(T *dst,
                const U *src,
                uint32_t num_frames,
                uint32_t num_group,
                uint32_t num_vector_elements,
                uint32_t num_vector_stride,
                intel_dnn_orientation_t orientation,
                float scaleFactor) {
    if (!dst || !src) {
        return;
    }
    if (orientation == kDnnInterleavedOrientation) {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_vector_elements; j++) {
                if (!std::is_same<T, U>::value) {
                    dst[j * num_group + i] = GNAPluginNS::ConvertFloatToInt16(src[i * num_vector_elements + j] * scaleFactor);
                } else {
                    dst[j * num_group + i] = src[i * num_vector_elements + j];
                }
            }
            // pad to meet weight matrix row length requirement
            for (uint32_t j = num_vector_elements; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
        // pad partial group
        for (uint32_t i = num_frames; i < num_group; i++) {
            for (uint32_t j = 0; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
    } else {
        if (!std::is_same<T, U>::value) {
            for (uint32_t i = 0; i < num_frames; i++) {
                T *ptr_dst_vec = reinterpret_cast<T *>(dst) + i * num_vector_stride;
                const U *ptr_src_vec = reinterpret_cast<const U *>(src) + i * num_vector_elements;
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                for (int j=0; j < num_vector_elements; j++) {
                    ptr_dst_vec[j] = GNAPluginNS::ConvertFloatToInt16(ptr_src_vec[j] * scaleFactor);
                }
            }

        } else {
            for (uint32_t i = 0; i < num_frames; i++) {
                void *ptr_dst_vec = reinterpret_cast<uint8_t *>(dst) + i * num_vector_stride * sizeof(T);
                const void *ptr_src_vec = reinterpret_cast<const uint8_t *>(src) + i * num_vector_elements * sizeof(U);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                ie_memcpy(ptr_dst_vec, num_vector_elements * sizeof(T),
                    ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (uint32_t i = num_frames; i < num_group; i++) {
            void *ptr_dst_vec = reinterpret_cast<uint8_t *>(dst) + i * num_vector_stride * sizeof(T);
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

template <typename T, typename U>
void GNAPlugin::copyInputDataWithSplit(T *const dst,
                const U *src,
                const GNASplitLayer& splitInfo,
                size_t precision_size,
                int idx) {
    if (!dst || !src) {
        return;
    }
    T *dst_ptr = dst;
    const U *src_ptr = src;
    precision_size = sizeof(T);
    // we found split/slice layer connected to Input
    for (auto&& outputLayer : splitInfo.splitOutputLayers) {
        uint32_t begin = outputLayer.offset / precision_size;
        uint32_t end = (outputLayer.offset + outputLayer.pure_size)/precision_size;
        if (dst_ptr - dst >= end) {
            // output layer with bind pointer as previous one. Skip
            continue;
        }
        for (uint32_t i = begin; i < end; ++i) {
            if (!std::is_same<T, U>::value) {
                *(dst_ptr++) = GNAPluginNS::ConvertFloatToInt16(*(src_ptr++) * inputsDesc->getScaleFactor(idx));
            } else {
                *(dst_ptr++) = *(src_ptr++);
            }
        }
        begin = end;
        end = (outputLayer.offset + ALIGN64(outputLayer.pure_size))/precision_size;
        std::memset(dst_ptr, 0, (end - begin )* sizeof(uint16_t));
        dst_ptr += end - begin;
    }
}

void GNAPlugin::ExportScores(void *ptr_dst,
                  const void *ptr_src,
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_active_elements,
                  uint32_t num_vector_stride,
                  uint32_t num_bytes_per_element_input,
                  uint32_t num_bytes_per_element) {
    // source scores are possibly padded to multiple of 8 and possibly interleaved
    // rotate if necessary and only copy actual scores (not padding)
    if (orientation == kDnnInterleavedOrientation) {
        if (num_bytes_per_element == 2) {
            int16_t *dst = reinterpret_cast<int16_t *>(ptr_dst);
            const int16_t *src = reinterpret_cast<const int16_t *>(ptr_src);
            for (uint32_t i = 0; i < num_frames; i++) {
                for (uint32_t j = 0; j < num_active_elements; j++) {
                    dst[i * num_vector_elements + j] = src[j * num_group + i];
                }
                for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                    dst[i * num_vector_elements + j] = 0;
                }
            }
        } else if (num_bytes_per_element == 4) {  // should work for both int and float
            int32_t *dst = reinterpret_cast<int32_t *>(ptr_dst);
            const int8_t *src = reinterpret_cast<const int8_t*>(ptr_src);
            for (uint32_t i = 0; i < num_frames; i++) {
                for (uint32_t j = 0; j < num_active_elements; j++) {
                    auto input_ptr = src + (j * num_group + i) * num_bytes_per_element_input;
                    auto dst_ptr = dst + (i * num_vector_elements + j);

                    switch (num_bytes_per_element_input) {
                        case 2 : {
                            *dst_ptr  = static_cast<int32_t>(*reinterpret_cast<const int16_t*>(input_ptr));
                            break;
                        }
                        case 4 : {
                            *dst_ptr = *reinterpret_cast<const int32_t *>(input_ptr);
                            break;
                        }
                        default:
                            THROW_GNA_EXCEPTION << "Unsupported output layer precision: " << num_bytes_per_element_input << "bytes";
                    }
                }
                for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                    dst[i * num_vector_elements + j] = 0;
                }
            }
        } else {
            THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << num_bytes_per_element << "bytes";
        }
    } else {
        if (num_bytes_per_element == 2) {
            for (uint32_t i = 0; i < num_frames; i++) {
                auto ptr_dst_vec = reinterpret_cast<uint8_t *>(ptr_dst) + i * num_vector_elements * sizeof(int16_t);
                auto ptr_src_vec = reinterpret_cast<const uint8_t *>(ptr_src) + i * num_vector_stride * sizeof(int16_t);
                memset(ptr_dst_vec, 0, num_vector_elements * sizeof(int16_t));
                ie_memcpy(ptr_dst_vec, num_active_elements * sizeof(int16_t),
                    ptr_src_vec, num_active_elements * sizeof(int16_t));
            }
        } else if (num_bytes_per_element == 4) {  // should work for both int and float
            if (num_bytes_per_element_input == 2) {
                for (uint32_t i = 0; i < num_frames; i++) {
                    auto ptr_dst_vec = reinterpret_cast<int32_t*>(ptr_dst) + i * num_vector_elements;
                    auto ptr_src_vec = reinterpret_cast<const int16_t*>(ptr_src) + i * num_vector_stride;
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ptr_src_vec[j];
                    }
                }
            } else {
                for (uint32_t i = 0; i < num_frames; i++) {
                    void* ptr_dst_vec = reinterpret_cast<uint8_t*>(ptr_dst) + i * num_vector_elements * sizeof(float);
                    const void* ptr_src_vec = reinterpret_cast<const uint8_t*>(ptr_src) + i * num_vector_stride * sizeof(float);
                    memset(ptr_dst_vec, 0, num_vector_elements * sizeof(float));
                    ie_memcpy(ptr_dst_vec, num_active_elements * sizeof(float),
                        ptr_src_vec, num_active_elements * sizeof(float));
                }
            }
        } else {
            THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << num_bytes_per_element << "bytes";
        }
    }
}

void GNAPlugin::ImportFrames(
                  void *ptr_dst,
                  const void *ptr_src,
                  Precision input_precision,
                  float scaleFactor,
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_vector_stride) {
    if (orientation == kDnnInterleavedOrientation) {
        // TODO : fix that as well
        if (input_precision == Precision::U8) {
            auto src = reinterpret_cast<const uint8_t *>(ptr_src);
            auto dst = reinterpret_cast<int16_t *>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else if (input_precision.size() == 2) {
            auto dst = reinterpret_cast<int16_t *>(ptr_dst);
            auto src = reinterpret_cast<const int16_t *>(ptr_src);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else if (input_precision.size() == 4) {
            if (!gnadevice) {
                auto dst = reinterpret_cast<float *>(ptr_dst);
                auto src = reinterpret_cast<const float *>(ptr_src);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            } else {
                auto dst = reinterpret_cast<int16_t *>(ptr_dst);
                auto src = reinterpret_cast<const float *>(ptr_src);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            }
        }
    } else {
        if (input_precision == Precision::U8) {
            auto src = reinterpret_cast<const uint8_t *>(ptr_src);
            if (!gnadevice) {
                auto dst = reinterpret_cast<float *>(ptr_dst);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            } else {
                auto dst = reinterpret_cast<int16_t *>(ptr_dst);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            }

        } else if (input_precision.size()== 2) {
            auto dst = reinterpret_cast<int16_t *>(ptr_dst);
            auto src = reinterpret_cast<const int16_t *>(ptr_src);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else if (input_precision.size() == 4) {
            if (!gnadevice) {
                auto dst = reinterpret_cast<float *>(ptr_dst);
                auto src = reinterpret_cast<const float *>(ptr_src);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            } else {
                auto dst = reinterpret_cast<uint16_t *>(ptr_dst);
                auto src = reinterpret_cast<const float *>(ptr_src);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            }
        }
    }
}

GNAPlugin::GNAPlugin() {
    Init();
    UpdateFieldsFromConfig();
}

GNAPlugin::GNAPlugin(const std::map<std::string, std::string>& configMap) {
    Init();
    SetConfig(configMap);
}

void GNAPlugin::Init() {
    dnn = std::make_shared<backend::AMIntelDNN>(backend::AMIntelDNN());
    inputsDesc = std::make_shared<GNAPluginNS::InputDesc>(GNAPluginNS::InputDesc());
    gnaFlags = std::make_shared<GNAPluginNS::GNAFlags>(GNAPluginNS::GNAFlags());

    graphCompiler.setDNNPtr(dnn);
    graphCompiler.setInputDescPtr(inputsDesc);
    graphCompiler.setGNAFlagsPtr(gnaFlags);
}

void GNAPlugin::InitGNADevice() {
#if GNA_LIB_VER == 1
    gnadevice = std::make_shared<GNADeviceHelper>(gnaFlags->gna_lib_async_threads_num,
                                                  gnaFlags->gna_openmp_multithreading,
                                                  gnaFlags->performance_counting);
#else
    gnadevice = std::make_shared<GNADeviceHelper>(config.pluginGna2DeviceConsistent,
                gnaFlags->gna_lib_async_threads_num,
                gnaFlags->gna_openmp_multithreading,
                gnaFlags->performance_counting);
#endif
    size_t page_size_bytes = 4096;
    gnamem = std::make_shared<gna_memory_type>(memory::make_polymorph<memory::GNAAllocator>(gnadevice), page_size_bytes);
    graphCompiler.setGNAMemoryPtr(gnamem);
}

void GNAPlugin::UpdateGnaQuantModeFromNetwork(InferenceEngine::CNNNetwork & network) {
    // fp32 emulation mode dont need any modifications to configuration
    if (config.gnaFlags.sw_fp32) return;

    // search for FQ layers
    // only supports cases of int16 or int8
    auto it = details::CNNNetworkIterator(network), end = details::CNNNetworkIterator();
    for (; it != end; it++) {
        if (!LayerInfo(*it).isFakeQuantize()) {
            continue;
        }

        GNAFakeQuantizeLayer fqLayer(*it);
        auto inputLayer = fqLayer.getInputLayer();

        // this fake quantize represents data quantization - not weights
        if (!LayerInfo(inputLayer).isConst()) {
            continue;
        }
        // also in mixed mode i8 should be stated as target precision
        if (fqLayer.getLevels() <= std::numeric_limits<uint8_t>::max()) {
            config.gnaPrecision = InferenceEngine::Precision::I8;
        } else if (fqLayer.getLevels() <= std::numeric_limits<uint16_t>::max()) {
            config.gnaPrecision = InferenceEngine::Precision::I16;
        } else {
            THROW_GNA_LAYER_EXCEPTION(*it)
                << "unsupported quantisation scheme: number of levels is " << fqLayer.getLevels() << " while only up to "
                << std::numeric_limits<uint16_t>::max() << " is supported";
        }

        gnaFlags->fake_quantized = true;
        config.gnaFlags.fake_quantized = true;
    }
}

void GNAPlugin::UpdateInputScaleFromNetwork(InferenceEngine::CNNNetwork & network) {
    // fp32 emulation mode dont need any modifications to configuration
    if (config.gnaFlags.sw_fp32) return;

    // search for FQ layers
    // only supports cases of int16 or int8
    InputsDataMap inputs = network.getInputsInfo();
    size_t inputIdx = 0;
    for (auto&& input : inputs) {
        auto data = input.second->getInputData();
        for (auto && nextToInputLayer : getInputTo(data)) {
            if (!LayerInfo(nextToInputLayer.second).isFakeQuantize()) {
                inputIdx++;
                continue;
            }
            // replacing scale factor from this fq layer
            GNAFakeQuantizeLayer fqLayer(nextToInputLayer.second);
            auto inputRange = fqLayer.getInputRange();
            auto outputRange = fqLayer.getOutputRange();
            if (inputRange.second.size() != 1 || inputRange.second.size() != 1 ||
                outputRange.second.size() != 1 || outputRange.second.size() != 1) {
                THROW_GNA_LAYER_EXCEPTION(nextToInputLayer.second)
                    << "unsupported, per-channel quantization for input layer : " << input.second->name();
            }

            auto fp32eq = [](float p1, float p2) -> bool {
                return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
            };
            float scaleInput = (fqLayer.getLevels() - 1) / (inputRange.second[0] - inputRange.first[0]);
            auto minAbsVal = std::min(std::abs(inputRange.second[0]), std::abs(inputRange.first[0]));
            auto maxAbsVal = std::max(std::abs(inputRange.second[0]), std::abs(inputRange.first[0]));
            if (fp32eq(minAbsVal, 0.0f) && !fp32eq(maxAbsVal, 0.0f)) {
                scaleInput = (fqLayer.getLevels() - 1) / (2 * maxAbsVal);
            }

            if (!config.inputScaleFactors.empty()) {
                gnalog() << "Scale factor calculated during model quantization (" << scaleInput
                    << ") will be used instead of user input (" << inputsDesc->inputScaleFactors[inputIdx] << ").\n";
                if (inputsDesc->inputScaleFactors[inputIdx] < scaleInput) {
                    gnawarn() << "WARNING: Scale factor calculated based on input values (" << inputsDesc->inputScaleFactors[inputIdx]
                        << ") is smaller than scale factor used to quantize model (" << scaleInput << "). "
                        << "Input values will be clamped.\n";
                }
            }

            config.inputScaleFactors[inputIdx] = scaleInput;
            inputsDesc->inputScaleFactors[inputIdx] = scaleInput;

            inputIdx++;
        }
    }
}

bool GNAPlugin::TryToInitOutput(int portId, InferenceEngine::CNNLayerPtr layer) {
    auto initOutput = [this, portId, layer]
            (intel_dnn_orientation_t orientation, size_t numBytesPerElem, size_t numElem, void* outputPtr) {
        auto & desc = outputsDesc[portId];
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

        desc.ptrs.resize(gnaFlags->gna_lib_async_threads_num);
        desc.orientation = orientation;
        desc.num_bytes_per_element = numBytesPerElem;
        desc.scale_factor = quantized != nullptr ? quantized->_dst_quant.GetScale() : 1.0f;
        desc.num_elements = numElem;

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->bind_ptr(&desc.ptrs.front(), outputPtr);
    };

    // probing gna_primitives
    auto irLayerAvatar = std::find_if(
        graphCompiler.dnnComponents.components.begin(),
        graphCompiler.dnnComponents.components.end(),
        [&layer](const backend::DnnComponents::storage_type::value_type & value) {
            return value.name == layer->name;
    });
    if (irLayerAvatar != graphCompiler.dnnComponents.components.end()) {
        initOutput(irLayerAvatar->dnnComponent.orientation_out, irLayerAvatar->dnnComponent.num_bytes_per_output,
                   irLayerAvatar->dnnComponent.num_rows_out, &irLayerAvatar->dnnComponent.ptr_outputs);
        return true;
    }

    // probing concatInfo
    if (LayerInfo(layer).isConcat()) {
        auto concatConnection  = graphCompiler.concat_connection.find(layer->name);
        if (concatConnection != graphCompiler.concat_connection.end()) {
            auto precision = layer->outData.front()->getPrecision().size();
            initOutput(kDnnInterleavedOrientation, precision, concatConnection->second.reserved_size / precision,
                       &concatConnection->second.gna_ptr);
            return true;
        }
    }

    // probing a constant info, for constant trivial networks support
    if (LayerInfo(layer).isConst()) {
        auto const_blob = layer->blobs["custom"];
        auto constConnection  = graphCompiler.const_connections.find(layer->name);
        if (constConnection != graphCompiler.const_connections.end()) {
            initOutput(kDnnInterleavedOrientation, layer->outData.front()->getPrecision().size(),
                       const_blob->size(), &constConnection->second);
            return true;
        }
    }

    return false;
}

void GNAPlugin::FillInputsAndOutputsTranspositionInfo(const InferenceEngine::CNNNetwork& net) {
    auto printTranspositionInfo = [](const std::vector<TranspositionInfo> &transpositionInfo) {
        for (const auto &transpositionInfoPart : transpositionInfo) {
            gnalog() << "transpose=" << transpositionInfoPart.transpose << " rows_num=" << transpositionInfoPart.num_transpose_rows
                     << " columns_num=" << transpositionInfoPart.num_transpose_columns << "\n";
        }
    };

    auto inputLayers = CNNNetGetAllInputLayers(net);
    for (const auto& inputLayer : inputLayers) {
        // Collect information for inputs transposition
        if (!LayerInfo(inputLayer).isInput()) continue;
        auto transpositionInfo = FindTranspositionInfoFromNextLayers(inputLayer);
        if (transpositionInfo.empty()) continue;

        transpose_inputs_info.insert({inputLayer->name, transpositionInfo});
        gnalog() << "Input " << inputLayer->name << " transposition info: \n";
        printTranspositionInfo(transpositionInfo);
    }

    auto outputsMap = net.getOutputsInfo();
    for (const auto& outPort : outputsMap) {
        auto outLayer = getCreatorLayer(outPort.second).lock();
        // Collect information for outputs transposition
        if (!LayerInfo(outLayer).isOutput()) continue;
        auto transpositionInfo = FindTranspositionInfoFromPrevLayers(outLayer);
        if (transpositionInfo.empty()) continue;

        // Swap transposition info rows and columns since we need to transpose output back from NHWC to NCHW
        for (auto && transpositionInfoPart : transpositionInfo) {
            if (transpositionInfoPart.transpose) {
                std::swap(transpositionInfoPart.num_transpose_rows, transpositionInfoPart.num_transpose_columns);
            }
        }
        transpose_outputs_info.insert({outLayer->name, transpositionInfo});
        gnalog() << "Output " << outLayer->name << " transposition info: \n";
        printTranspositionInfo(transpositionInfo);
    }
}

#ifdef PLOT
void GNAPlugin::AddDebugProperties(const InferenceEngine::CNNLayerPtr layer,
    InferenceEngine::ordered_properties& printed_properties,
    InferenceEngine::ordered_properties& node_properties) {
    // printing quantized params
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    if (!quantized) {
        return;
    }
    if (LayerInfo(layer).isWeightable() || LayerInfo(layer).isEltwise()) {
        printed_properties.emplace_back(
            "weights scale factor", std::to_string(quantized->_weights_quant.GetScale()));
        if (quantized->_weights_quant.IsStatsSet()) {
            for (auto& min : quantized->_weights_quant.GetMinValues()) {
                printed_properties.emplace_back(
                    "weights min val", std::to_string(min));
            }
            for (auto& max : quantized->_weights_quant.GetMaxValues()) {
                printed_properties.emplace_back(
                    "weights max val", std::to_string(max));
            }
        }

        if (quantized->_bias_quant.IsStatsSet()) {
            for (auto& min : quantized->_bias_quant.GetMinValues()) {
                printed_properties.emplace_back(
                    "bias min val", std::to_string(min));
            }
            for (auto& max : quantized->_bias_quant.GetMaxValues()) {
                printed_properties.emplace_back(
                    "bias max val", std::to_string(max));
            }
        }
    }
    printed_properties.emplace_back(
        "src scale factor", std::to_string(quantized->_src_quant.GetScale()));
    if (quantized->_src_quant.IsStatsSet()) {
        for (auto& min : quantized->_src_quant.GetMinValues()) {
            printed_properties.emplace_back(
                "src min val", std::to_string(min));
        }
        for (auto& max : quantized->_src_quant.GetMaxValues()) {
            printed_properties.emplace_back(
                "src max val", std::to_string(max));
        }
    }

    printed_properties.emplace_back(
        "dst scale factor", std::to_string(quantized->_dst_quant.GetScale()));
    if (quantized->_dst_quant.IsStatsSet()) {
        for (auto& min : quantized->_dst_quant.GetMinValues()) {
            printed_properties.emplace_back(
                "dst min val", std::to_string(min));
        }
        for (auto& max : quantized->_dst_quant.GetMaxValues()) {
            printed_properties.emplace_back(
                "dst max val", std::to_string(max));
        }
    }
}
#endif

void GNAPlugin::LoadNetwork(CNNNetwork & _network) {
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork;
    if (_network.getFunction()) {
        CNNNetwork clonedNetwork = InferenceEngine::cloneNetwork(_network);
        const auto& graph = clonedNetwork.getFunction();
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
        manager.register_pass<ngraph::pass::ConvertPriorBox>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        // UnrollTI should be the last transformation in the transformation pipeline
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();

        const auto& pass_config = manager.get_pass_config();
        pass_config->set_callback<ngraph::pass::UnrollTensorIterator>(
                [](const std::shared_ptr<const ngraph::Node> &node) -> bool {
                    // UnrollTI transformation is disabled by default, is turned on by LowLatency transformation
                    return node->get_rt_info().count("UNROLL_TI") == 0;
            });
        pass_config->disable<ngraph::pass::FakeQuantizeMulFusion>();
        pass_config->disable<ngraph::pass::FakeQuantizeReshapeFusion>();
        pass_config->disable<ngraph::pass::PullTransposeThroughFQUp>();
        pass_config->disable<ngraph::pass::ReluFakeQuantizeFusion>();
        manager.run_passes(graph);
        convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(graph, clonedNetwork);
    }
    InferenceEngine::CNNNetwork network = convertedNetwork ? InferenceEngine::CNNNetwork{convertedNetwork} : _network;

    NetPass::ConvertPrecision(network, Precision::I64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U32, Precision::I32);

    //  Check the input network
    std::string error;
    if (!AreLayersSupported(network, error)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // FQ networks now replaces certain flags in the plugin - flags will'be owerritten
    UpdateGnaQuantModeFromNetwork(network);
    UpdateInputScaleFromNetwork(network);

    if (MustBeConvertedFromNCHWToNHWC(details::CNNNetSortTopologically(network))) {
        FillInputsAndOutputsTranspositionInfo(network);
    }

    // network optimisation phases
    int passIdx = 0;
    auto run_passes = [&] (const CNNNetwork& network, bool runBeforeCopy) {
        auto passes = make_shared<PassManager>(PassManagerSettings{policy, runBeforeCopy}, network);
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<UnrollTIPass>();
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<InsertIdentityToLSTMCellPass>();
        passes->registerPass<UnrollLSTMCellPass>();
        passes->registerPass<RemoveSingleInputConcatPass>();

        // fake quantisation aware passes
        passes->registerPass<FuseFQIntoWeightsPass>();
        passes->registerPass<MoveFakeQuantizeLayerIntoQuantParamsPass>();

        passes->registerPass<TransposeWeightsFromNCHWToNHWCPass>();

        passes->registerPass<SubstitutePReluPass>();
        passes->registerPass<SubstituteSoftSignPass>();

        passes->registerPass<ReorderMaxPoolPass>();
        passes->registerPass<EltwiseSplitOverChannelsPass>();
        passes->registerPass<InsertSplitAligningFilterPass>();

        passes->registerPass<FlattenTrivialConcatPass>();
        passes->registerPass<InsertConcatAligningFilterPass>();
        passes->registerPass<ReorderConcatInputsPass>();
        if (policy.PermutePolicy != Policy::Permute::DISABLED) {
            passes->registerPass<ReversePermutationsPass>();
        }
        if (policy.NHWCToNCHWPolicy != Policy::NHWCToNCHW::DISABLED) {
            passes->registerPass<RemovePermutationsNHWCToNCHWPass>();
        }
        passes->registerPass<InsertIdentityLayerPass>();
        passes->registerPass<BreakFusingOfOutputLayersPass>();
        passes->registerPass<InsertCopyLayerPass>();
        passes->registerPass<InsertDiagonalLayerPass>();
        passes->registerPass<HandleMultipleActivationsForTheLayerPass>();
#if GNA_LIB_VER == 2
        passes->registerPass<ForbidActivationFusingPass>();
#endif
        passes->registerPass<SubstituteScaleShiftBroadCastPass>();
        passes->registerPass<FuseMultipleIdentitiesPass>();
        passes->registerPass<BroadcastConstPass>();
        passIdx = passes->run(passIdx);
    };

    InferenceEngine::CNNNetwork newNet;
    if (gnaFlags->sw_fp32) {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            transformLayer(lp, WeightsConverter());
            return lp;
        };
        newNet = InferenceEngine::CNNNetCopy(network, visitor);
        // to run all passes need to have two calls to pass manager
        run_passes(newNet, true);
        run_passes(newNet, false);
    } else if (gnaFlags->fake_quantized) {
        switch (config.gnaPrecision) {
            case Precision::I16:
                ModelQuantizer<FakeQuantI16> q16;
                newNet = q16.quantize(network, run_passes, inputsDesc->inputScaleFactors);
                break;
            case Precision::I8:
                ModelQuantizer<FakeQuantI8> q8;
                newNet = q8.quantize(network, run_passes, inputsDesc->inputScaleFactors);
                break;
            default:
                THROW_GNA_EXCEPTION << "unsupported GNA precision for quantisation: " << config.gnaPrecision;
        }
    } else {
        switch (config.gnaPrecision) {
            case Precision::I16:
                ModelQuantizer<QuantI16> q16;
                newNet = q16.quantize(network, run_passes, inputsDesc->inputScaleFactors);
                break;
            case Precision::I8:
                ModelQuantizer<QuantI8> q8;
                newNet = q8.quantize(network, run_passes, inputsDesc->inputScaleFactors);
                break;
            default:
                THROW_GNA_EXCEPTION << "unsupported GNA precision for quantisation: " << config.gnaPrecision;
        }
    }

    auto inputLayers = CNNNetGetAllInputLayers(newNet);

#ifdef PLOT
    std::ofstream file("gna_passes.dot");
    saveGraphToDot(newNet, file, [this](const CNNLayerPtr layer,
        ordered_properties& printed_properties,
        ordered_properties& node_properties) {
            AddDebugProperties(layer, printed_properties, node_properties);
        });
#endif

    auto sortedNet = CNNNetSortTopologicallyEx(newNet, make_fuzed_order);

    // passing policy to compiler
    graphCompiler.setPolicy(policy);

    if (sortedNet.empty()) {
        THROW_GNA_EXCEPTION << "Sorted network is empty";
    }

    std::vector<CNNLayerPtr> sortedNoMem;
    std::unordered_map<std::string, std::vector<InferenceEngine::CNNLayerPtr>> memoryPairs;
    // find all memory layers pairs and mark which one used as outputs
    for (auto &layer : sortedNet) {
        auto generic = dynamic_cast<GenericLayer *>(layer.get());
        if (generic == nullptr) {
            sortedNoMem.push_back(layer);
            continue;
        }
        LayerInfo layerInfo(layer);
        if (layerInfo.isMemory()) {
            // collect all memory pairs
            auto id = generic->GetParamAsString("id");
            memoryPairs[id].resize(generic->GetParamAsInt("size"));
            memoryPairs[id][generic->GetParamAsInt("index")] = layer;
            continue;
        } else if (layerInfo.isConcat()) {
            graphCompiler.fillConcatConnections(layer);
        } else if (layerInfo.isSplit() || layerInfo.isSlice()) {
            graphCompiler.fillSplitConnections(layer);
        }
        sortedNoMem.push_back(layer);
    }

    // fill in extra storage with memory layers
    graphCompiler.fillMemoryConnections(memoryPairs);

    if (!graphCompiler.memory_connection.empty()) {
        gnaFlags->gna_lib_async_threads_num = 1;
    }

    if (gnaFlags->sw_fp32) {
        gnamem.reset(new gna_memory_type(memory::make_polymorph<std::allocator<uint8_t>>()));
        graphCompiler.setGNAMemoryPtr(gnamem);
    } else {
        InitGNADevice();
    }

    // keep inputs information and create input primitives
    inputsDataMap = newNet.getInputsInfo();
    if (inputsDataMap.empty()) {
        gnawarn() << "No inputs for the topology\n";
    }

    // keep output dims
    outputsDataMap = newNet.getOutputsInfo();
    if (outputsDataMap.empty()) {
        THROW_GNA_EXCEPTION << "No outputs for the topology";
    }

    for (auto && input : inputsDataMap) {
        inputsDesc->getPtrInputsGlobal(input.first).resize(gnaFlags->gna_lib_async_threads_num);
    }

    // CreatingLayer primitives
    for (auto & layer : sortedNoMem) {
        graphCompiler.CreateLayerPrimitive(layer);
    }

    for (auto& inputLayer : inputLayers) {
        auto layerInfo = LayerInfo(inputLayer);
        if (layerInfo.isInput() && 0 == inputsDesc->bytes_allocated_for_input[inputLayer->name]) {
            graphCompiler.connectOutput(inputLayer, &inputsDesc->getPtrInputsGlobal(inputLayer->name).front(), 0);
        }
    }

    if (graphCompiler.dnnComponents.components.empty()) {
        gnawarn() << "No GNA primitives created based on topology. This might indicate trivial topology\n";
        trivialTopology = true;
    }

    /// setting-up output layers information
    outputsDesc.resize(outputsDataMap.size());

    int portId = 0;
    for (auto && outPort : outputsDataMap) {
        // gets output layer pointer in original topology not in cloned
        auto outLayer = getCreatorLayer(outPort.second).lock();

        // Memory layers are not dnnComponents hence we need to make switch with identity layer
        if (outLayer->type == "Memory") {
            // traverse memory connection to find corresponding output_memory
            for (auto && memConnection : graphCompiler.memory_connection) {
                if (memConnection.second.getInput()->name == outLayer->name) {
                    // if connection is found, replace memory input layer with memory output layer
                    outLayer = memConnection.second.getOutput();
                    break;
                }
            }
        }

        // searching for outData represented in GNA blob
        // using ufs - upper first search
        gnalog() << "[UFS] searching for : "<< outPort.first << " representation in GNA\n";
        bool stopSearching = false;

        CNNNetDFS(outLayer, [this, &outPort, portId, &stopSearching](CNNLayerPtr layer) {
            gnalog() << "[UFS] from : "<< outPort.first <<" reached: " << layer->name << "\n";
            stopSearching = TryToInitOutput(portId, layer);
        }, true, [&stopSearching](InferenceEngine::CNNLayer* from) {
            return make_upstream_order(!stopSearching ? from : nullptr);
        });
        if (!stopSearching) {
            THROW_GNA_EXCEPTION << "unsupported topology: cannot locate " << outPort.first
                                << " after compiling GNA graph";
        }
        portId++;
    }

    // TODO: how active list will work in multioutput case
    // make room for active list
    gnamem->reserve_ptr(nullptr,
        ALIGN64(outputsDesc.front().num_bytes_per_element * outputsDesc.front().num_elements), 64);

    void *pParallelExecutionData  = nullptr;

    // reserving more bytes for intermediate data in parallel case - TODO: this works incorrectly in compact mode at lest
    rwSegmentSize = gnamem->getRWBytes();
    if (gnaFlags->gna_lib_async_threads_num > 1) {
        gnamem->reserve_ptr(&pParallelExecutionData, gnamem->getRWBytes() * (gnaFlags->gna_lib_async_threads_num - 1), 64);
    }

    gnamem->commit();

    dnn->Init(gnamem->getBasePtr(),
             gnamem->getTotalBytes(),
             gnaFlags->sw_fp32 ? kDnnFloat : kDnnInt,
             1);

    // TODO: this copy is unneeded; in fact, we can directly create gna structs from list
    auto execOrder = graphCompiler.dnnComponents.getExecutionOrder();
    dnn->component.insert(dnn->component.begin(), execOrder.begin(), execOrder.end());

    // in fp32 mode last PWL cannot be computed without that
    if (!graphCompiler.dnnComponents.components.empty()) {
        dnn->InitActiveList(NULL);
    }

#if GNA_LIB_VER == 2
    gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>()));
#else
    nnets.emplace_back(make_shared<CPPWrapper<intel_nnet_type_t>>(), -1, InferenceEngine::BlobMap());
#endif
    if (!gnaFlags->sw_fp32 && !graphCompiler.dnnComponents.components.empty()) {
        // number of layer gets calculated inside that InitGNAStruct function
#if GNA_LIB_VER == 2
        dnn->InitGNAStruct(&std::get<0>(gnaModels.front())->obj);
#else
        dnn->InitGNAStruct(&std::get<0>(nnets.front())->obj);
#endif
    }

    // creating same gna RW segment for parallel infer requests
    for (int i = 1; i != gnaFlags->gna_lib_async_threads_num; i++) {
#if GNA_LIB_VER == 2
        gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>()));
        // this can be improved by just copy all structures, but we are too lazy
        dnn->InitGNAStruct(&std::get<0>(gnaModels.back())->obj);
#else
        nnets.emplace_back(make_shared<CPPWrapper<intel_nnet_type_t>>(), -1, InferenceEngine::BlobMap());
        dnn->InitGNAStruct(&std::get<0>(nnets.back())->obj);
#endif
        // relocate rw pointers to new offset
        auto basePtr = reinterpret_cast<uint8_t*>(pParallelExecutionData) + rwSegmentSize * (i - 1);

        auto relocate = [basePtr, this](void *& ptr_out, void * ptr_in) {
            if (ptr_in == nullptr) {
                ptr_out = nullptr;
            } else {
                auto offset = reinterpret_cast<uint8_t *>(ptr_in) - reinterpret_cast<uint8_t *>(gnamem->getBasePtr());
                ptr_out = basePtr + offset;
            }
        };

        for (auto &&input : inputsDesc->ptr_inputs_global_storage) {
            relocate(input[i], input[0]);
        }

        // relocating all output pointers
        for (int j = 0; j < outputsDesc.size(); ++j) {
            relocate(outputsDesc[j].ptrs[i], outputsDesc[j].ptrs[0]);
        }

#if GNA_LIB_VER == 2
        for (int j = 0; j != std::get<0>(gnaModels.front())->obj.NumberOfOperations; j++) {
            auto & gnaOperation = std::get<0>(gnaModels[i])->obj.Operations[j];
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[0])->Data, gnaOperation.Operands[0]->Data);
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[1])->Data, gnaOperation.Operands[1]->Data);
#else
        for (int j = 0; j != std::get<0>(nnets.front())->obj.nLayers; j++) {
            auto & layer = std::get<0>(nnets[i])->obj.pLayers[j];
            relocate(layer.pInputs, layer.pInputs);
            relocate(layer.pOutputs, layer.pOutputs);
            relocate(layer.pOutputsIntermediate, layer.pOutputsIntermediate);
#endif
        }
    }

    // calculating input orientation without memory layers, since their orientation not changed during infer right now
    std::unordered_map<string, std::vector<string>> skippedLayers;

    bool withConv = false;
    for (auto &layer : sortedNet) {
        auto layerInfo = LayerInfo(layer);
        if (layerInfo.isConvolution()) {
            withConv = true;
            break;
        }
    }
    if (withConv) {
        for (auto &inputLayer : sortedNet) {
            if (!LayerInfo(inputLayer).isInput()) {
                continue;
            }
            auto doesntHaveGnaMapping = [this] (CNNLayerPtr l) {
                auto dnnLayer = graphCompiler.dnnComponents.findComponent(l);
                return dnnLayer == nullptr;
            };

            auto nextLayers = CNNNetGetAllNextLayersSkipCertain(inputLayer, -1, doesntHaveGnaMapping);

            for (auto &nextLayer : nextLayers) {
                auto dnnLayer = graphCompiler.dnnComponents.findComponent(nextLayer);
                // non functional layer - skipped by gna
                if (nullptr == dnnLayer) {
                    THROW_GNA_LAYER_EXCEPTION(inputLayer) << " gna mapped layer search connection failed";
                }
                // input orientation might be already initialized, thus verify that it matches
                if (!inputsDesc->orientation_in.count(inputLayer->name)) {
                    inputsDesc->orientation_in[inputLayer->name] = dnnLayer->orientation_in;
                } else {
                    if (inputsDesc->orientation_in[inputLayer->name] != dnnLayer->orientation_in) {
                        THROW_GNA_EXCEPTION << "orientation for input layer: " << inputLayer->name << "cannot be calculated";
                    }
                }
            }
        }
    } else {
        for (auto &inputLayer : inputLayers) {
            inputsDesc->orientation_in[inputLayer->name] = kDnnInterleavedOrientation;
        }
    }

    if (dnn->do_rotate_input && transpose_inputs_info.empty()) {
        for (auto &inputLayer : inputLayers) {
            transpose_inputs_info.insert({inputLayer->name,
                {TranspositionInfo{dnn->do_rotate_input, dnn->num_rotate_rows, dnn->num_rotate_columns}}});
        }
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob.dot");
#endif
#if GNA_LIB_VER == 2
    createRequestConfigsForGnaModels();
#endif
}

#if GNA_LIB_VER == 2
void GNAPlugin::createRequestConfigsForGnaModels() {
    if (!gnadevice || trivialTopology) {
        gnaRequestConfigToRequestIdMap.push_back(std::make_tuple(FAKE_REQUEST_CONFIG_ID, -1, InferenceEngine::BlobMap()));
        return;
    }
    for (auto& model : gnaModels) {
        auto& gnaNnet = std::get<0>(model).get()->obj;
        const auto modelId = gnadevice->createModel(gnaNnet);
        const auto requestConfigId = gnadevice->createRequestConfig(modelId);
        gnaRequestConfigToRequestIdMap.push_back(std::make_tuple(requestConfigId, -1, InferenceEngine::BlobMap()));
    }
}

#endif

int GNAPlugin::GetDeviceVersionFromString(const std::string deviceString) {
    constexpr uint32_t embeddedSuffix = 0xE;
    if (deviceString.empty())
        return 0x100 + embeddedSuffix;
    if (deviceString.size() == 4 && deviceString.substr(0, 3) == "GNA") {
        int version = deviceString[3] - '0';
            if (version > 0) {
            version <<= 8;
            version += embeddedSuffix;
            return version;
        }
    }
    THROW_GNA_EXCEPTION << "Wrong GNA generation for embedded model dump: " << deviceString;
}

void GNAPlugin::DumpXNNToFile() const {
    // TODO: output  precision as well as pointer might be incorrect, LSTM for sure
    // gna looks automatically set layer 0 as output and adjust it's pointer / precision/ size respectively
    if (config.dumpXNNPath.empty()) {
        return;
    }

    const auto versionInt = GetDeviceVersionFromString(config.dumpXNNGeneration);

    if (!gnadevice) {
        THROW_GNA_EXCEPTION << "Cannot generate XNNDump for float network";
    }
    std::ofstream dumpStream(config.dumpXNNPath, std::ios::out | std::ios::binary);
#if GNA_LIB_VER == 1
    if (versionInt != 0x10E)
        THROW_GNA_EXCEPTION << "Wrong GNA version for embedded model dump: " << config.dumpXNNGeneration;
    auto dump = gnadevice->dumpXnn(&std::get<0>(nnets.front())->obj, ptr_active_indices, num_active_indices);
    dump.header.rw_region_size = gnamem->getRWBytes();
    dump.header.input_scaling_factor = inputsDesc->inputScaleFactors.front();
    dump.header.output_scaling_factor = outputsDesc.front().scale_factor;
    dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(intel_gna_model_header));
    dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.model_size);
#else
    auto const modelId = gnadevice->createModel(std::get<0>(gnaModels.front())->obj);
    if (versionInt == Gna2DeviceVersionEmbedded1_0) {
        auto dump = gnadevice->dumpXnn(modelId);
        dump.header.RwRegionSize = gnamem->getRWBytes();
        dump.header.InputScalingFactor = inputsDesc->inputScaleFactors.front();
        dump.header.OutputScalingFactor = outputsDesc.front().scale_factor;
        dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(Gna2ModelSueCreekHeader));
        dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.ModelSize);
    } else {
        static_assert(sizeof(versionInt) >= sizeof(Gna2DeviceVersion), "");
        gnadevice->dumpXnnForDeviceVersion(modelId, dumpStream,
            *reinterpret_cast<const Gna2DeviceVersion*>(&versionInt));
    }
    gnadevice->releaseModel(modelId);
#endif
}

uint32_t GNAPlugin::QueueInference(const InferenceEngine::BlobMap &inputs, InferenceEngine::BlobMap &result) {
#if GNA_LIB_VER == 2
    auto& nnets = gnaRequestConfigToRequestIdMap;
#endif
    auto freeNnet = std::find_if(std::begin(nnets), std::end(nnets), [](decltype(nnets.front()) & item) {
        return std::get<1>(item) == -1;
    });

    if (freeNnet == nnets.end()) {
        if (!graphCompiler.memory_connection.empty()) {
            Wait(0);
            freeNnet = nnets.begin();
        } else {
            THROW_IE_EXCEPTION << as_status << REQUEST_BUSY
                               << "GNA executable network has max of "
                               << static_cast<uint32_t >(gnaFlags->gna_lib_async_threads_num)
                               << " parallel infer requests, please sync one of already running";
        }
    }

    auto idx = static_cast<uint32_t>(std::distance(std::begin(nnets), freeNnet));

    int inputNum = 0;
    for (auto &input : inputs) {
        auto inputLayout = input.second->getTensorDesc().getLayout();
        if (inputLayout != Layout::C && inputLayout != Layout::NC && inputLayout != Layout::CN &&
            inputLayout != Layout::CHW && inputLayout != Layout::NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or Layout::CHW. But was: "
                                << input.second->getTensorDesc().getLayout();
        }

        if (inputLayout == Layout::NCHW || inputLayout == Layout::CHW) {
            // specific case that can be squeezed to 2d
            inputLayout = Layout::NC;
        }

        auto is1D = input.second->getTensorDesc().getLayout() == Layout::C;
        auto is3D = input.second->getTensorDesc().getLayout() == Layout::CHW;

        if (!inputsDesc->ptr_inputs_global_id.count(input.first)) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (inputsDesc->getPtrInputsGlobal(input.first)[idx] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << idx << " not set";
        }
        const auto inputOrientation = inputsDesc->getOrientation(input.first);
        if (inputOrientation == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input orientation for " << input.first << " not set";
        }

        for (auto& outputDesc : outputsDesc) {
            if (outputDesc.orientation == kDnnUnknownOrientation) {
                // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
                THROW_GNA_EXCEPTION << "network not loaded : output orientation not set";
            }
        }

        auto dims = input.second->getTensorDesc().getDims();
        auto  importedElements = is1D ? dims[0] : details::product(++std::begin(dims), std::end(dims));
        auto  importedFrames = (is3D || is1D) ? 1 : dims[0];
        auto  targetGroups = is1D ? 1 : dims[0]; // TODO: no proper support for groups yet

        auto  importedElementSizeBytes = gnaFlags->sw_fp32 ? 4 : 2;
        auto  importedBytes = importedElements * importedFrames * importedElementSizeBytes;

        if (inputsDesc->bytes_allocated_for_input[input.first] < importedBytes) {
            THROW_GNA_EXCEPTION << "Cannot import input frames for :" << input.first
                                  << ", allocated size: " << inputsDesc->bytes_allocated_for_input[input.first]
                                  << ", but input blob size: " << importedBytes;
        }

        ImportFrames(inputsDesc->getPtrInputsGlobal(input.first)[idx],
                     input.second->cbuffer().as<float *>(),
                     input.second->getTensorDesc().getPrecision(),
                     gnaFlags->sw_fp32 ? 1.0f : inputsDesc->getScaleFactor(inputNum),
                     inputOrientation,
                     importedFrames,
                     targetGroups,
                     importedElements,
                     importedElements);

        auto transpose_info = transpose_inputs_info.find(input.first);
        if (transpose_info != std::end(transpose_inputs_info)) {
            size_t batchSize = (dims.size() > 1) ? dims[0] : 1;
            size_t elementsPerBatch = (dims.size() > 1) ? InferenceEngine::details::product(dims) / dims[0] : dims[0];
            size_t transposed_data_size = 0;
            for (const auto &part_transposition_info : transpose_info->second) {
                transposed_data_size += part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match input buffer length of " << elementsPerBatch;
            }
            auto input_ptr = reinterpret_cast<uint8_t *>(inputsDesc->getPtrInputsGlobal(input.first)[idx]);
            ConvertTensorFromNCHWToNHWC(gnadevice ? 2 : 4, batchSize, elementsPerBatch, input_ptr, true, transpose_info->second);
        }
        ++inputNum;
    }
    // If there is no gnadevice infer using reference FP32 transforamtions
    if (!gnadevice || trivialTopology) {
        auto runtime = runtime::FP(dnn);
        runtime.infer();
        if (freeNnet != nnets.end()) {
            std::get<1>(*freeNnet) = 1;
        }
    } else {
#if GNA_LIB_VER == 1
        auto nnet = std::get<0>(*freeNnet).get();
        std::get<1>(*freeNnet) = gnadevice->propagate(&nnet->obj, ptr_active_indices, num_active_indices, config.gna_proc_type);
#else
        const auto reqConfigId = std::get<0>(*freeNnet);
        if (ptr_active_indices != nullptr && num_active_indices > 0 && activeLayerIndex != 0xffffffff)
            gnadevice->setUpActiveList(reqConfigId, activeLayerIndex, ptr_active_indices, num_active_indices);
        std::get<1>(*freeNnet) = gnadevice->propagate(reqConfigId, config.pluginGna2AccMode);
#endif
    }

#ifdef PLOT
    dnn->BeginNewWrite(dnn_dump_write_index);
    if (dnn->num_components() != 0) {
        dnn->WriteDnnText("Net_.txt", kDnnFloat);
    }
    dnn_dump_write_index++;
#endif
    if (freeNnet != nnets.end()) {
        // TODO: GNA2: Substitute properly when using GNA 2.0 Library setting and CPU
        std::get<2>(*freeNnet) = result;
    }
    return idx;
}

bool GNAPlugin::Wait(uint32_t request_idx) {
    return GNA_REQUEST_COMPLETED == WaitFor(request_idx, MAX_TIMEOUT);
}

GnaWaitStatus GNAPlugin::WaitFor(uint32_t request_idx, int64_t millisTimeout) {
#if GNA_LIB_VER == 2
    auto& nnets = gnaRequestConfigToRequestIdMap;
#endif
    // TODO: GNA2: check whether necessary
    if (nnets.size() <= request_idx) return GNA_REQUEST_COMPLETED;
    // already synced TODO: might be copy required ???
    if (std::get<1>(nnets[request_idx]) == -1) return GNA_REQUEST_COMPLETED;

    if (gnadevice && !trivialTopology) {
        const auto waitStatus = gnadevice->wait(std::get<1>(nnets[request_idx]), millisTimeout);
        if (waitStatus == GNA_REQUEST_ABORTED) {
            std::get<1>(nnets[request_idx]) = -1;
            return GNA_REQUEST_ABORTED;
        }
        if (waitStatus == GNA_REQUEST_PENDING) {
            return GNA_REQUEST_PENDING;
        }
    }

    std::get<1>(nnets[request_idx]) = -1;
    auto &request = std::get<2>(nnets[request_idx]);
#ifdef PLOT
    if (dnn->num_components() != 0) {
        dnn->WriteInputAndOutputText();
    }
#if GNA_LIB_VER == 1
    dnn->WriteInputAndOutputTextGNA(&std::get<0>(nnets[request_idx])->obj);
#else
    dnn->WriteInputAndOutputTextGNA(std::get<0>(gnaModels[request_idx])->obj);
#endif
#endif
    int output_idx = 0;
    for (auto && outputBlobIt : request) {
        auto & outputBlob = outputBlobIt.second;
        auto & outputDesc = outputsDesc[output_idx];
        if (!outputBlob->getTensorDesc().getLayout() == Layout::C && !outputBlob->getTensorDesc().getLayout() == Layout::NC &&
            !outputBlob->getTensorDesc().getLayout() == Layout::CN && !outputBlob->getTensorDesc().getLayout() == Layout::NCHW &&
            !outputBlob->getTensorDesc().getLayout() == Layout::CHW) {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or Layout::CHW. But was "
                << outputBlob->getTensorDesc().getLayout();
        }

        auto dims = outputBlob->getTensorDesc().getDims();
        auto is1D = outputBlob->getTensorDesc().getLayout() == Layout::C;
        auto is3D = outputBlob->getTensorDesc().getLayout() == Layout::CHW;
        auto& exportOutputDims = outputBlob->getTensorDesc().getDims();
        auto batchSize = (is1D || is3D) ? 1 : exportOutputDims[0];
        auto elementsPerBatch = is1D ? exportOutputDims.front() :
            details::product(++std::begin(exportOutputDims), std::end(exportOutputDims));

        auto transpose_output_info = transpose_outputs_info.find(outputBlobIt.first);
        if (transpose_output_info != std::end(transpose_outputs_info)) {
            size_t transposed_data_size = 0;
            for (const auto &part_transposition_info : transpose_output_info->second) {
                transposed_data_size += part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match output buffer length of " << elementsPerBatch;
            }
            ConvertTensorFromNCHWToNHWC(outputDesc.num_bytes_per_element,
                                        batchSize,
                                        elementsPerBatch,
                                        reinterpret_cast<uint8_t*>(outputDesc.ptrs[request_idx]),
                                        true,
                                        transpose_output_info->second);
        }

        ExportScores(outputBlob->buffer(),
                        outputDesc.ptrs[request_idx],
                        outputDesc.orientation,
                        batchSize,
                        batchSize,
                        elementsPerBatch,
                        elementsPerBatch,
                        elementsPerBatch,
                        outputDesc.num_bytes_per_element,
                        sizeof(float));

        if (gnadevice) {
#ifdef PLOT
            FILE* f = nullptr;
            static int num_infers = 0;
            {
                f = fopen("ex_scores.txt", "w");
            }
            num_infers++;
            if (f) {
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < dims[dims.size() - 1]; j++) {
                        fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t*>()[dims[dims.size() - 1] * i + j]);
                    }
                    fprintf(f, "\n");
                }
                fprintf(f, "\n\n");
            }
#endif
            ConvertToFloat(outputBlob->buffer(),
                outputBlob->buffer(),
                elementsPerBatch,
                batchSize,
                outputDesc.scale_factor);
#ifdef PLOT
            if (f) {
                auto dims = outputBlob->getTensorDesc().getDims();
                for (int i = 0; i < batchSize; i++) {
                    for (int j = 0; j < dims[dims.size() - 1]; j++) {
                        fprintf(f, "%.2f ", outputBlob->cbuffer().as<float*>()[dims[dims.size() - 1] * i + j]);
                    }
                    fprintf(f, "\n");
                }
                fclose(f);
            }
#endif
        }

        output_idx++;
    }
    return GNA_REQUEST_COMPLETED;
}

void GNAPlugin::Reset() {
    graphCompiler.Reset();
}

bool GNAPlugin::Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &output) {
    BlobMap bmInput;
    BlobMap bmOutput;
    if (inputsDataMap.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"<< "model accepts " << inputsDataMap.size() << " inputs";
    }

    IE_ASSERT(!inputsDataMap.empty());
    bmInput[inputsDataMap.begin()->first] = std::shared_ptr<Blob>(const_cast<Blob*>(&input), [](Blob*){});
    IE_ASSERT(!outputsDataMap.empty());
    bmOutput[outputsDataMap.begin()->first] = std::shared_ptr<Blob>(&output, [](Blob*){});
    return Infer(bmInput, bmOutput);
}

bool GNAPlugin::Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result) {
    return  Wait(QueueInference(input, result));
}

static InferenceEngine::Layout GetLayoutForDims(const InferenceEngine::SizeVector &dims) {
    switch (dims.size()) {
    case 1: return C;
    case 2: return NC;
    case 3: return CHW;
    case 4: return NCHW;
    default:
        THROW_GNA_EXCEPTION << "Unsupported dimensions size in GNA: " << dims.size();
    }
}

Blob::Ptr GNAPlugin::GetOutputBlob(const std::string& name, InferenceEngine::Precision precision) {
    // need to have intermediate blob for interleave conversion
    InferenceEngine::Blob::Ptr outputBlob;
    auto outputDims = outputsDataMap[name]->getTensorDesc().getDims();
    outputBlob = make_blob_with_precision(TensorDesc(precision, outputDims, GetLayoutForDims(outputDims)));
    outputBlob->allocate();
    return outputBlob;
}

Blob::Ptr GNAPlugin::GetInputBlob(const std::string& name, InferenceEngine::Precision precision) {
    InferenceEngine::Blob::Ptr inputBlob;
    // need to have intermediate blob for interleave conversion
    // TODO: NCHW format support is experimental = c++ MO did insert reshape, while TF mo - not
    auto inputDims = inputsDataMap[name]->getTensorDesc().getDims();
    inputBlob = make_blob_with_precision(TensorDesc(precision, inputDims, GetLayoutForDims(inputDims)));
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::VariableStateInternal::Ptr>  GNAPlugin::QueryState() {
    if (memoryStates.size() != graphCompiler.memory_connection.size()) {
        memoryStates.clear();
        for (auto& connection : graphCompiler.memory_connection) {
            auto state = std::make_shared<memory::GNAVariableState>(connection.first, std::make_shared <GNAMemoryLayer>(connection.second));
            memoryStates.emplace_back(state);
        }
    }
    return memoryStates;
}

std::string GNAPlugin::GetName() const noexcept {
    return _pluginName;
}

void GNAPlugin::SetName(const std::string & pluginName) noexcept {
    _pluginName = pluginName;
}

InferenceEngine::ExecutableNetwork GNAPlugin::ImportNetwork(std::istream& networkModel) {
    auto header = GNAModelSerial::ReadHeader(networkModel);

    InitGNADevice();

    graphCompiler.setGNAMemoryPtr(gnamem);
    void *basePtr = nullptr;
    gnamem->reserve_ptr(&basePtr, header.gnaMemSize);
    gnamem->commit();
#if GNA_LIB_VER == 2
    gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>(header.layersCount)));
#else
    nnets.emplace_back(make_shared<CPPWrapper<intel_nnet_type_t>>(header.layersCount), -1, InferenceEngine::BlobMap());
    std::get<0>(nnets.back())->obj.nGroup = header.nGroup;
#endif
    GNAModelSerial::MemoryType  mt;
#if GNA_LIB_VER == 2
    auto serial = GNAModelSerial(&std::get<0>(gnaModels.back())->obj, mt);
#else
    auto serial = GNAModelSerial(&std::get<0>(nnets.back())->obj, mt);
#endif

    if (!inputsDesc->inputScaleFactors.empty()) {
        gnalog() << "[Import Network] Clearing input scale factors defined in configuration. "
                 << "Scale factors provided in imported model will be used\n";
        inputsDesc->inputScaleFactors.clear();
    }

    serial.setHeader(header);
    serial.Import(basePtr,
            header.gnaMemSize,
            networkModel,
            inputsDesc,
            outputsDesc,
            inputsDataMap,
            outputsDataMap,
            transpose_inputs_info,
            transpose_outputs_info);

#if GNA_LIB_VER == 2
    auto getOrientation = [](Gna2Operation & gnaOperation) {
        return gnaOperation.Type == Gna2OperationTypeConvolution ?
            kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;
    };
#else
    auto getOrientation = [](intel_nnet_layer_t & layer) {
        return layer.nLayerKind == INTEL_CONVOLUTIONAL ?
           kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;
    };
#endif

#if GNA_LIB_VER == 1
    inputsDesc->orientation_in["input"] = getOrientation(std::get<0>(nnets.back())->obj.pLayers[0]);
    outputsDesc[0].orientation = getOrientation(std::get<0>(nnets.back())->obj.pLayers[std::get<0>(nnets.back())->obj.nLayers - 1]);
#endif

    if (header.doRotateInput) {
        for (auto && input : inputsDataMap) {
            transpose_inputs_info.insert({input.first, {{header.doRotateInput, header.nRotateRows, header.nRotateColumns}}});
        }
    }
    if (header.doRotateOutput) {
        for (auto && output : outputsDataMap) {
            transpose_outputs_info.insert({output.first, {{header.doRotateOutput, header.nRotateOutputRows, header.nRotateOutputColumns}}});
        }
    }

    for (auto && memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr, gnaFlags->sw_fp32 ? 4 : 2);
        std::string name;
        std::tie(memoryLayer.gna_ptr, memoryLayer.reserved_size, name, memoryLayer.scale_factor) = memory;
        graphCompiler.memory_connection.emplace_back(make_pair(name, memoryLayer));
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob-imported.dot");
#endif
#if GNA_LIB_VER == 2
    trivialTopology = (std::get<0>(gnaModels.back())->obj.NumberOfOperations == 0);
    createRequestConfigsForGnaModels();
#else
    trivialTopology = (std::get<0>(nnets.back())->obj.nLayers == 0);
#endif
    return {};
}

void GNAPlugin::Export(const std::string &fileName) {
    std::fstream outStream(fileName, ios_base::out | ios_base::binary);
    Export(outStream);
}

void GNAPlugin::Export(std::ostream &outStream) {
    if (inputsDesc->ptr_inputs_global_id.empty() || outputsDesc.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

#if GNA_LIB_VER == 1
    if (inputsDesc->ptr_inputs_global_id.size() != 1) {
        THROW_GNA_EXCEPTION << " exporting network with multiple inputs not supported";
    }
#endif

    // TODO: nnet group parameter looks only used in application - so can we move this line into load network.
    IE_ASSERT(!inputsDataMap.empty());
    auto inputDims = inputsDataMap.begin()->second->getTensorDesc().getDims();
    if (inputDims.size() == 2) {
#if GNA_LIB_VER == 1
        std::get<0>(nnets.front())->obj.nGroup = inputDims[0];
#endif
    }
#if GNA_LIB_VER == 2
    Gna2Model* modelToSerial = &std::get<0>(gnaModels.front())->obj;
#else
    intel_nnet_type_t* modelToSerial = &std::get<0>(nnets.front())->obj;
#endif
    auto serial = GNAModelSerial(modelToSerial,
                                 inputsDesc,
                                 outputsDesc,
                                 inputsDataMap,
                                 outputsDataMap)
                    .SetInputRotation(transpose_inputs_info)
                    .SetOutputRotation(transpose_outputs_info);

    for (auto && memoryConnection : graphCompiler.memory_connection) {
        auto state = std::make_shared<memory::GNAVariableState>(memoryConnection.first, std::make_shared <GNAMemoryLayer>(memoryConnection.second));
        gnalog() << "Scale factor Memory layer " << state->GetScaleFactor() << std::endl;
        serial.AddState(memoryConnection.second.gna_ptr, memoryConnection.second.reserved_size, memoryConnection.first, state->GetScaleFactor());
    }

    serial.Export(gnamem->getBasePtr(), gnamem->getTotalBytes(), outStream);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GNAPlugin::GetPerformanceCounts() {
    if (gnaFlags->performance_counting) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        gnadevice->getGnaPerfCounters(perfMap);
        return perfMap;
    } else {
        return {};
    }
}

void GNAPlugin::AddExtension(InferenceEngine::IExtensionPtr extension) {}

void GNAPlugin::SetConfig(const std::map<std::string, std::string> &config_map) {
    config.UpdateFromMap(config_map);
    UpdateFieldsFromConfig();
}

void GNAPlugin::UpdateFieldsFromConfig() {
    inputsDesc->inputScaleFactors = config.inputScaleFactors;
    *gnaFlags = config.gnaFlags;
}

InferenceEngine::QueryNetworkResult GNAPlugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                            const std::map<std::string, std::string>& config) const {
    InferenceEngine::QueryNetworkResult res;

    if (network.getFunction()) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << " ngraph::Function is not supported natively";
    }

    std::unordered_set<CNNLayer *> allLayers;
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);

    if (inputs.empty()) {
        THROW_GNA_EXCEPTION << "Network is empty (GNA)\n";
    }

    auto const & secondLayers = getInputTo(inputs.begin()->second->getInputData());
    if (secondLayers.empty()) {
        THROW_GNA_EXCEPTION << "Network consists of input layer only (GNA)\n";
    }

    InferenceEngine::details::UnorderedDFS(allLayers,
                                           secondLayers.begin()->second,
                                           [&](CNNLayerPtr const& layer) {
                                                if (LayerTypeFromStr(layer->type) != LayerType::NO_TYPE) {
                                                    res.supportedLayersMap.insert({ layer->name, GetName() });
                                                }
                                            }, false);

    return res;
}
