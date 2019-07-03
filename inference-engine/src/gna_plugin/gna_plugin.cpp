// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "gna_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "debug.h"
#include "blob_factory.hpp"
#include "gna_plugin_log.hpp"
#include "gna_layer_info.hpp"
#include <utility>
#include <limits>
#include "ie_memcpy.h"

#ifdef PLOT
void ExportGnaNetworkAndrzej(const char *ptr_name, intel_nnet_type_t* pNeuralNetwork);
#endif

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <list>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <dnn_memory.hpp>
#include <ie_layers.h>
#include "details/caseless.hpp"
#include <gna-api-types-xnn.h>
#include "gna-api.h"
#include "gna-api-dumper.h"
#include "dnn.h"
#include "pwl.h"
#include "util.h"
#include "quantization/quantization.h"
#include "lstm.hpp"
#include "graph_tools.hpp"
#include "gna_plugin_config.hpp"
#include "gna/gna_config.hpp"
#include "quantization/model_quantizer.hpp"
#include "gna_model_serial.hpp"
#include "gna_memory_state.hpp"
#include "details/ie_cnn_network_tools.h"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;
using namespace InferenceEngine::details;

#ifdef VERBOSE
#define VERBOSE_LEVEL (1)
#else
#define VERBOSE_LEVEL (0)
#endif

#ifdef PLOT
#define PLOT_LEVEL (1)
#else
#define PLOT_LEVEL (0)
#endif


#define PAGE_SIZE_BYTES 4096

#define FROM_IR_DIM(mem, idx)\
((mem->dims.size() > idx - 1) ? mem->dims[idx - 1] : 1)

inline int16_t GNAPluginNS::ConvertFloatToInt16(float src) {
        float rounding_value = (src > 0) ? 0.5f : -0.5f;
        float value = src + rounding_value;
        if (value > 32767.0) {
            return 32767;
        } else if (value < -32768.0) {
            return -32768;
        }
        return (int16_t)value;
}

void GNAPluginNS::ConvertToInt16(int16_t *ptr_dst,
                    const float *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows*num_columns; i++) {
        ptr_dst[i] = GNAPluginNS::ConvertFloatToInt16(ptr_src[i]*scale_factor);
    }
}
void GNAPluginNS::ConvertToFloat(float *ptr_dst,
                    int32_t *ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor) {
    if (!ptr_dst || !ptr_src) {
        return;
    }
    for (uint32_t i = 0; i < num_rows; i++) {
        int32_t *ptr_int_row = ptr_src + i * num_columns;
        float *ptr_float_row = ptr_dst + i * num_columns;
        for (uint32_t j = 0; j < num_columns; j++) {
            ptr_float_row[j] = static_cast<float>(ptr_int_row[j]) / scale_factor;
        }
    }
}

template <typename T, typename U>
void GNAPlugin::copyInputData(T *dst,
                const U *src,
                uint32_t num_frames,
                uint32_t num_group,
                uint32_t num_vector_elements,
                uint32_t num_vector_stride,
                intel_dnn_orientation_t orientation) {
    if (!dst || !src) {
        return;
    }
    if (orientation == kDnnInterleavedOrientation) {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_vector_elements; j++) {
                if (!std::is_same<T, U>::value) {
                    dst[j * num_group + i] = GNAPluginNS::ConvertFloatToInt16(src[i * num_vector_elements + j] * get_input_scale_factor());
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
                T *ptr_dst_vec = const_cast<T *>(reinterpret_cast<const T *>(dst) + i * num_vector_stride);
                U *ptr_src_vec = const_cast<U *>(reinterpret_cast<const U *>(src) + i * num_vector_elements);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                for (int j=0; j < num_vector_elements; j++) {
                    ptr_dst_vec[j] = GNAPluginNS::ConvertFloatToInt16(ptr_src_vec[j] * get_input_scale_factor());
                }
            }

        } else {
            for (uint32_t i = 0; i < num_frames; i++) {
                void *ptr_dst_vec = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(dst) + i * num_vector_stride * sizeof(T));
                void *ptr_src_vec = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(src) + i * num_vector_elements * sizeof(U));
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                std::memcpy(ptr_dst_vec, ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (uint32_t i = num_frames; i < num_group; i++) {
            void *ptr_dst_vec = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(dst) + i * num_vector_stride * sizeof(T));
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

template <typename T, typename U>
void GNAPlugin::copyInputDataWithSplit(T *const dst,
                const U *src,
                const GNASplitLayer& splitInfo,
                size_t precision_size) {
    if (!dst || !src) {
        return;
    }
    T *dst_ptr = dst;
    const U *src_ptr = src;
    precision_size = sizeof(T);
    // we found split/slice layer connected to Input
    for (auto&& outputLayer : splitInfo.splitOutputLayers) {
        uint32_t begin = outputLayer.offset/precision_size;
        uint32_t end = (outputLayer.offset + outputLayer.pure_size)/precision_size;
        if (dst_ptr - dst >= end) {
            // output layer with bind pointer as previous one. Skip
            continue;
        }
        for (uint32_t i = begin; i < end; ++i) {
            if (!std::is_same<T, U>::value) {
                *(dst_ptr++) = GNAPluginNS::ConvertFloatToInt16(*(src_ptr++) * get_input_scale_factor());
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
                  void *ptr_src,
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
            int16_t *src = reinterpret_cast<int16_t *>(ptr_src);
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
            int8_t *src = reinterpret_cast<int8_t*>(ptr_src);
            for (uint32_t i = 0; i < num_frames; i++) {
                for (uint32_t j = 0; j < num_active_elements; j++) {
                    auto input_ptr = src + (j * num_group + i) * num_bytes_per_element_input;
                    auto dst_ptr = dst + (i * num_vector_elements + j);

                    switch (num_bytes_per_element_input) {
                        case 2 : {
                            *dst_ptr  = static_cast<int32_t>(*reinterpret_cast<int16_t*>(input_ptr));
                            break;
                        }
                        case 4 : {
                            *dst_ptr  = *reinterpret_cast<int32_t*>(input_ptr);
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
                void *ptr_dst_vec = reinterpret_cast<void *> (reinterpret_cast<uint8_t *>(ptr_dst) + i * num_vector_elements * sizeof(int16_t));
                void *ptr_src_vec = reinterpret_cast<void *> (reinterpret_cast<uint8_t *>(ptr_src) + i * num_vector_stride * sizeof(int16_t));
                memset(ptr_dst_vec, 0, num_vector_elements * sizeof(int16_t));
                memcpy(ptr_dst_vec, ptr_src_vec, num_active_elements * sizeof(int16_t));
            }
        } else if (num_bytes_per_element == 4) {  // should work for both int and float
            for (uint32_t i = 0; i < num_frames; i++) {
                void *ptr_dst_vec = reinterpret_cast<void *> (reinterpret_cast<uint8_t *>(ptr_dst) + i * num_vector_elements * sizeof(float));
                void *ptr_src_vec = reinterpret_cast<void *> (reinterpret_cast<uint8_t *>(ptr_src) + i * num_vector_stride * sizeof(float));
                memset(ptr_dst_vec, 0, num_vector_elements * sizeof(float));
                memcpy(ptr_dst_vec, ptr_src_vec, num_active_elements * sizeof(float));
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
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_vector_stride) {
    if (orientation == kDnnInterleavedOrientation) {
        // TODO : fix that as well
        if (input_precision == Precision::U8) {
            int16_t *dst = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_dst));
            uint8_t *src = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(ptr_src));
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
        } else if (input_precision.size() == 2) {
            int16_t *dst = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_dst));
            int16_t *src = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_src));
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
        } else if (input_precision.size() == 4) {
            if (!gnadevice) {
                float *dst = const_cast<float *>(reinterpret_cast<const float *>(ptr_dst));
                float *src = const_cast<float *>(reinterpret_cast<const float *>(ptr_src));
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            } else {
                int16_t *dst = reinterpret_cast<int16_t *>(ptr_dst);
                const float *src = reinterpret_cast<const float *>(ptr_src);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            }
        }
    } else {
        if (input_precision == Precision::U8) {
            uint8_t *src = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(ptr_src));
            if (!gnadevice) {
                float *dst = const_cast<float *>(reinterpret_cast<const float *>(ptr_dst));
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            } else {
                int16_t *dst = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_dst));
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            }

        } else if (input_precision.size()== 2) {
            int16_t *dst = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_dst));
            int16_t *src = const_cast<int16_t *>(reinterpret_cast<const int16_t *>(ptr_src));
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
        } else if (input_precision.size() == 4) {
            if (!gnadevice) {
                float *dst = const_cast<float *>(reinterpret_cast<const float *>(ptr_dst));
                float *src = const_cast<float *>(reinterpret_cast<const float *>(ptr_src));
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            } else {
                uint16_t *dst = const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(ptr_dst));
                float *src = const_cast<float *>(reinterpret_cast<const float *>(ptr_src));
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation);
            }
        }
    }
}

void GNAPlugin::fillMemoryConnections(std::unordered_map<std::string,
                                            std::vector<InferenceEngine::CNNLayerPtr>>& memoryPairs) {
    for (auto &memory : memoryPairs) {
        auto inputLayer = memory.second[1];
        auto outputLayer = memory.second[0];

        IE_ASSERT(1 == outputLayer->insData.size());

        // creating connection for layers output as form of extramap
        memory_connection.emplace_back(memory.first, GNAMemoryLayer(inputLayer, outputLayer));
    }
}

void GNAPlugin::fillConcatConnections(InferenceEngine::CNNLayerPtr layer) {
    // creating connection for each layer outputs as form of extramap
    GNAPlugin::GNAConcatLayer layerInfoItem(layer);
    size_t concat_size = 0;
    std::string& id = layer->name;

    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto dataInput = layer->insData[i].lock();
        if (!dataInput) {
            THROW_GNA_EXCEPTION << "Input layer pointer for concat is unexpectedly absent";
        }

        auto ptrConcatLayerInput = dataInput->creatorLayer.lock();
        if (!ptrConcatLayerInput) {
            THROW_GNA_EXCEPTION << "Input layer for concat is unexpectedly absent";
        }
        layerInfoItem.concatInputLayers.emplace_back(
                GNAPlugin::GNAConcatLayer::ConcatConnectedLayerInfo({ptrConcatLayerInput->name, concat_size}));

        size_t layer_size =
                     InferenceEngine::details::product(begin(dataInput->dims),
                                                      end(dataInput->dims)) * dataInput->precision.size();
        concat_size += layer_size;
    }
    layerInfoItem.reserved_size = concat_size;
    concat_connection.emplace(id, layerInfoItem);
}

void GNAPlugin::fillSplitConnections(InferenceEngine::CNNLayerPtr layer) {
    // creating connection for each layer inputs as form of extramap
    GNAPlugin::GNASplitLayer layerInfoItem(layer);
    size_t split_size = 0;
    std::string& id = layer->name;
    auto dataInput = layer->insData.begin()->lock();
    if (!dataInput) {
        THROW_GNA_EXCEPTION << "Input layer pointer for split/slice is unexpectedly absent";
    }
    auto ptrSplitLayerInput = dataInput->creatorLayer.lock();
    if (!ptrSplitLayerInput) {
        THROW_GNA_EXCEPTION << "Input layer for split/slice is unexpectedly absent";
    }

    LayerInfo ptrSplitLayerInputLayerInfo(ptrSplitLayerInput);
    for (size_t i = 0; i < layer->outData.size(); ++i) {
        size_t padding = 0;
        size_t output_layer_size = 0;
        auto& dataOutput = layer->outData[i];

        if (!dataOutput || !dataInput) {
            THROW_GNA_EXCEPTION << "Output layer pointer for split/slice is unexpectedly absent";
        }

        for (auto&& ptrSplitLayerOutputPair : dataOutput->getInputTo()) {
            auto& ptrSplitLayerOutput = ptrSplitLayerOutputPair.second;
            if (!ptrSplitLayerOutput) {
                THROW_GNA_EXCEPTION << "Output layer for split/slice is unexpectedly absent";
            }

            padding = std::max(padding, LayerInfo(ptrSplitLayerOutput).paddingSize())
                                                        * dataOutput->precision.size();
            output_layer_size =
                    InferenceEngine::details::product(begin(dataOutput->dims),
                                                     end(dataOutput->dims)) * dataOutput->precision.size();

            if (ptrSplitLayerOutput->type == "AffineFilter") {
                size_t aligned64_offset = ptrSplitLayerOutput->GetParamAsInt("offset");
                layerInfoItem.splitOutputLayers.emplace_back(ptrSplitLayerOutput->name, aligned64_offset, output_layer_size);
            } else {
                layerInfoItem.splitOutputLayers.emplace_back(ptrSplitLayerOutput->name, split_size, output_layer_size);
            }
        }

        split_size += padding + output_layer_size;
    }
    layerInfoItem.reserved_size = split_size;
    layerInfoItem.splitInputLayer =
                    GNAPlugin::GNASplitLayer::SplitConnectedLayerInfo({ptrSplitLayerInput->type, 0,
                                                                    InferenceEngine::details::product(begin(dataInput->dims),
                                                                    end(dataInput->dims)) * dataInput->precision.size()});
    split_connection.emplace(id, layerInfoItem);
}

void GNAPlugin::DiagonalPrimitive(InferenceEngine::CNNLayerPtr layer) {
    AffinePrimitive(layer, true);
}

void GNAPlugin::ConvolutionPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto &convolution = dynamic_cast<ConvolutionLayer &>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_feature_map_rows = FROM_IR_DIM(inputs, 1) / convolution._stride_x;
    uint32_t num_feature_map_columns = FROM_IR_DIM(inputs, 3) * convolution._stride_x / num_feature_maps;

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 3);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_padding = ALIGN(convolution._kernel_x * num_feature_map_columns * num_feature_maps, 8)
                                            - convolution._kernel_x * num_feature_map_columns * num_feature_maps;
    void *ptr_inputs;
    void *ptr_outputs;
    void *ptr_weights;
    void *ptr_biases;

    // TODO: questionable why for biases that are no in IR we inventing precision
    auto biasPrecision = convolution._biases ? convolution._biases->precision() : outputs->precision;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;

#ifdef PLOT
    cout << "IR layer : " << std::left << std::setw(20) << layer->name << dnnComponentsForLayer.size() - 1 << "\n";
#endif
    auto num_input_padding = ALIGN(num_feature_maps * num_feature_map_columns * num_feature_map_rows, 8)
                                                        -  num_feature_maps * num_feature_map_columns * num_feature_map_rows;
    auto num_filter_rows = convolution._kernel_x / convolution._stride_x;
    dnn.InitConvolutional1DComponent(currentComponent,
                            1,
                            num_feature_maps *  num_feature_map_columns * num_feature_map_rows + num_input_padding,
                            1,
                            num_rows_out * convolution._out_depth,
                            inputs->precision.size(),
                            outputs->precision.size(),
                            convolution._weights->precision().size(),
                            biasPrecision.size(),
                            convolution._out_depth,
                            num_filter_rows,
                            num_feature_maps * num_feature_map_columns * num_filter_rows + num_padding,

                            num_feature_maps,  // interesting - why this is so in gna_example
                            num_feature_map_rows,
                            num_feature_map_columns,

                            quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs,
                            ptr_weights,
                            ptr_biases);

    // update num_feature_maps for next convolutional layer
    num_feature_maps = convolution._out_depth;  // = number of filters

    size_t num_data_bytes_out =
                        InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims))
                                                                                * outputs->precision.size();

    size_t num_data_bytes_in = num_columns_in * (num_rows_in + num_padding) * inputs->precision.size();

    auto connectedInputLayer = connectInput(layer, ptr_inputs, num_data_bytes_in).input;

    // TODO: convolution might be not the first layer in sorted order but connected via split for example - dont know how kaldi will handle that
    if (LayerInfo(connectedInputLayer).isInput()) {
        //  Kaldi features are opposite orientation
        dnn.num_rotate_rows = num_feature_map_columns;
        dnn.num_rotate_columns = num_feature_map_rows;
    }

    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);

    // rotate
    auto TransposeMatrix = [](uint8_t *ptr_matrix, size_t element_size, uint32_t num_rows, uint32_t num_cols) {
        std::vector<uint8_t> temp_buffer(num_rows * num_cols * element_size);
        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_cols; j++) {
                    ie_memcpy(&temp_buffer.front() + (j*num_rows + i)*element_size,
                          temp_buffer.size() - (i * num_cols + j) * element_size,
                          ptr_matrix + (i*num_cols+j)*element_size,
                          element_size);
            }
        }
        return temp_buffer;
    };

    std::vector<uint8_t > transposedWeights;
    for (uint32_t k = 0; k < convolution._out_depth; k++) {
        uint8_t *ptr_filt_current
            = convolution._weights->cbuffer().as<uint8_t *>() + k * num_columns_in * convolution._kernel[X_AXIS] * convolution.precision.size();
        auto transposedPart = TransposeMatrix(ptr_filt_current, convolution.precision.size(), num_columns_in, convolution._kernel[X_AXIS]);
        transposedWeights.insert(transposedWeights.end(), transposedPart.begin(), transposedPart.end());
    }

    if (num_padding == 0) {
        gnamem->readonly().push_local_ptr(ptr_weights, transposedWeights.data(), convolution._weights->byteSize(), 64);
    } else {
        auto elementsIn = convolution._kernel_x * num_feature_map_columns + num_padding;
        auto paddedWeights = elementsIn * convolution._out_depth;
        auto paddedWeightsSize = paddedWeights * convolution.precision.size();
        auto elements_in_row = convolution._kernel_x * num_feature_map_columns;
        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void * data, size_t size) {
            for (int i = 0; i < convolution._out_depth; i++) {
                memcpy(data,
                       transposedWeights.data() + elements_in_row * i * convolution.precision.size(),
                       elements_in_row * convolution.precision.size());

                data = reinterpret_cast<uint8_t *>(data) + elementsIn * convolution.precision.size();
            }
        }, 64);
    }

    if (convolution._biases) {
        gnamem->readonly().push_ptr(ptr_biases,
                                    convolution._biases->cbuffer().as<const void *>(),
                                    convolution._biases->byteSize(),
                                    64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAPlugin::PowerPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto &power = dynamic_cast<PowerLayer &>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    if (power.power != 1.0) {
        THROW_IE_EXCEPTION << "[GNA plugin] unsupported power factor, expected 1 but was " << power.power;
    }

    auto input = layer->insData[0].lock();

    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(input, 1);
    uint32_t num_columns_in = FROM_IR_DIM(input, 2);
    uint32_t num_rows_out = num_rows_in;

    void *ptr_inputs;
    void *ptr_outputs;
    void *ptr_weights;
    void *ptr_biases;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;
    dnn.InitAffineComponent(currentComponent,
                            num_rows_in,
                            num_columns_in,
                            num_rows_out,
                            input->precision.size(),
                            outputs->precision.size(),
                            // TODO: only fp32 and Int16 tested
                            quantized == nullptr ? input->precision.size() : 2,
                            quantized == nullptr ? input->precision.size() : 4,
                            quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs,
                            ptr_weights,
                            ptr_biases,
                            true);

#ifdef PLOT
    cout << "IR layer : " << std::left << std::setw(20) << layer->name << "diagonal_"<< dnnComponentsForLayer.size() - 1 << "\n";
#endif

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims))
        * outputs->precision.size();

    size_t num_data_bytes_in = InferenceEngine::details::product(begin(input->dims), end(input->dims))
        * input->precision.size();

    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);
    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);

    if (power.scale != 1.0f) {
        if (quantized == nullptr) {
            gnamem->readonly().push_value(ptr_weights, power.scale, num_rows_out, 64);
        } else {
            auto scaledIdentity = quantized->_weights_quant.scale * power.scale;

            #define FLOAT_TO_INT16(a) static_cast<int16_t>(((a) < 0)?((a) - 0.5):((a) + 0.5))

            auto quantizedIdentity = FLOAT_TO_INT16(std::min(scaledIdentity, static_cast<float>(INT16_MAX)));
            gnamem->readonly().push_value<int16_t>(ptr_weights, quantizedIdentity, num_rows_out, 64);
        }
    }

    if (power.offset != 0.0f) {
        if (quantized == nullptr) {
            gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
        } else {
            gnamem->readonly().push_value<int32_t>(ptr_biases, 0, num_rows_out, 64);
        }
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAPlugin::PoolingPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto &pooling = dynamic_cast<PoolingLayer &>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 3);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_columns_out = FROM_IR_DIM(outputs, 3);
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    void *ptr_inputs;
    void *ptr_outputs;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;

#ifdef PLOT
    cout << "IR layer : " << std::left << std::setw(20) << layer->name << dnnComponentsForLayer.size() - 1 << "\n";
#endif
    switch (pooling._type) {
        case PoolingLayer::MAX: break;
        // we are loosing precision here
        case PoolingLayer::AVG:
        default:
            // TODO: convert to SUMM pooling
            THROW_GNA_EXCEPTION << "Layer :" << layer->name << " not supported";
    }

    dnn.InitMaxpoolComponent(currentComponent,
                            1,
                            num_columns_in * num_rows_in ,
                            1,
                            num_columns_out * num_rows_out,
                            inputs->precision.size(),
                            outputs->precision.size(),
                            pooling._kernel[X_AXIS],
                            pooling._kernel[X_AXIS],
                            num_columns_in,
                            false,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs);

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims))
        * outputs->precision.size();

    size_t num_data_bytes_in = num_columns_in * (num_rows_in + num_padding) * inputs->precision.size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);
}

void GNAPlugin::CopyPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_columns_out = FROM_IR_DIM(outputs, 2);
    uint32_t num_padding_in = ALIGN(num_rows_in, 8) - num_rows_in;
    uint32_t num_padding_out = ALIGN(num_rows_out, 8) - num_rows_out;
    void *ptr_inputs;
    void *ptr_outputs;
    auto orientation = (num_cnn_rows_out > 0) ? kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;
    dnn.InitCopyComponent(currentComponent,
                          orientation,
                          ALIGN(num_rows_in, 8),
                          num_columns_in,
                          ALIGN(num_rows_out, 8),
                          num_columns_out,
                          inputs->precision.size(),
                          outputs->precision.size(),
                          quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                          num_rows_out + num_padding_out,
                          num_columns_out,
                          ptr_inputs,
                          ptr_outputs);

    size_t num_data_bytes_out = ALIGN(InferenceEngine::details::product(
                                                            begin(outputs->dims), end(outputs->dims)), 8)
                                                                                * outputs->precision.size();
    size_t num_data_bytes_in = num_columns_in * ALIGN(num_rows_in, 8) * inputs->precision.size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);
}

void GNAPlugin::ConcatPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto concatLayer = dynamic_cast<InferenceEngine::ConcatLayer *> (layer.get());

    if (concatLayer == nullptr) {
        return;
    }
    if (concatLayer->insData.size() != 2) {
        THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers.";
    }

    auto prevInput0 = concatLayer->insData[0].lock();
    auto prevInput1 = concatLayer->insData[1].lock();
    if (!prevInput0 || !prevInput1) {
        THROW_GNA_EXCEPTION << "Input layer for concat is unexpectedly absent";
    }
    if (prevInput0->precision.size() != prevInput1->precision.size()) {
        THROW_GNA_EXCEPTION << "Different precision for Concat input layers are not supported";
    }

    auto& concatLayerInfo = concat_connection.find(concatLayer->name)->second;
    for (auto &&outLayer : concatLayer->outData.front()->getInputTo()) {
        if ( LayerInfo(outLayer.second).isConcat() ) {
            connectOutput(layer, &concatLayerInfo.gna_ptr,
                          &concatLayerInfo.gna_ptr, concatLayerInfo.reserved_size);
        }
    }

    size_t idx = 0;
    for (auto && inputLayer : concatLayerInfo.concatInputLayers) {
        if ( InferenceEngine::details::CaselessEq<std::string>()
                                            (inputLayer.name, "input") ) {
            connectInput(layer, &concatLayerInfo.gna_ptr,
                                concatLayerInfo.reserved_size-inputLayer.offset, static_cast<int32_t>(-inputLayer.offset), idx);
        }
        ++idx;
    }
}

void GNAPlugin::CropPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto cropLayer = dynamic_cast<InferenceEngine::CropLayer *> (layer.get());

    if (cropLayer == nullptr) {
        return;
    }
    if (cropLayer->axis.size() > 1) {
        THROW_GNA_EXCEPTION <<
        "Crop layer does not support the number of cropped dimentions = "
        << cropLayer->axis.size() << ".";
    }

    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    size_t cropOffset = cropLayer->offset.back() * cropLayer->precision.size();
    size_t cropOutputSize = cropLayer->dim.back() * cropLayer->precision.size();

    if (ALIGN64(cropOffset) == cropOffset) {
        // leave crop as it is
        GNAPlugin::GNACropLayer cropLayerInfoItem(layer);
        std::string& id = layer->name;
        crop_connection.emplace(id, cropLayerInfoItem);
        auto cropLayerInfo = crop_connection.find(cropLayer->name);

        if (cropLayerInfo == crop_connection.end()) {
            THROW_GNA_EXCEPTION <<
            "Item is not in the storage but it was added recently...\n";
        }

        // calculate index idx for connectInput last parameter
        connectInput(layer, &cropLayerInfo->second.gna_ptr, cropOutputSize + cropOffset, cropOffset, 0);

        // cases for certain output layers
        for (auto &&outLayer : layer->outData.front()->getInputTo()) {
            auto& nextLayer = outLayer.second;
            if ( LayerInfo(nextLayer).isConcat() ) {
                connectOutput(layer, &cropLayerInfo->second.gna_ptr, &cropLayerInfo->second.gna_ptr, cropOutputSize);
            }
        }
    } else {
        gnalog() << "Crop " << layer->name << " is being replaced by Affine layer...\n";
        auto outputs = *layer->outData.begin();
        auto inputs = layer->insData.begin()->lock();

        uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
        uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
        uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
        uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

        void *ptr_inputs;
        void *ptr_outputs;
        void *ptr_weights;
        void *ptr_biases;

        dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
        auto &currentComponent = dnnComponentsForLayer.back().second;
        dnn.InitAffineComponent(currentComponent,
                                num_rows_in + num_padding,
                                num_columns_in,
                                num_rows_out,
                                inputs->precision.size(),
                                4,
                                quantized == nullptr ? inputs->precision.size() : 2,
                                4,
                                quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                                quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                                ptr_inputs,
                                ptr_outputs,
                                ptr_weights,
                                ptr_biases,
                                false);

        size_t num_data_bytes_out =
        InferenceEngine::details::product(
                                          begin(outputs->dims), end(outputs->dims)) * 4;

        size_t num_data_bytes_in = num_columns_in *
                ALIGN(num_rows_in, 8) * inputs->precision.size();

        connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
        connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);

        FillWeightOfAligningFilter(layer, ptr_weights, cropLayer->offset.back(), (quantized == nullptr) ? false : true);

        (quantized == nullptr) ?
            gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64):
            gnamem->readonly().push_value<int32_t>(ptr_biases, 0, num_rows_out, 64);
    }
}

void GNAPlugin::SplitPrimitive(InferenceEngine::CNNLayerPtr layer) {
//  Nothing to do
}

void GNAPlugin::SlicePrimitive(InferenceEngine::CNNLayerPtr layer) {
//  Nothing to do
}

void GNAPlugin::EltwisePrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto &eltwise = dynamic_cast<EltwiseLayer &>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    // for eltwise should be one input of 4 bytes and one of 2 bytes - detecting that
    auto inputs2Bytes = layer->insData[0].lock();
    auto inputs4Bytes = layer->insData[1].lock();

    int biasesLayerIdx = 1;

    if (quantized) {
        if (eltwise._operation == EltwiseLayer::Sum) {
            if (inputs4Bytes->precision.size() != 4) {
                std::swap(inputs4Bytes, inputs2Bytes);
                biasesLayerIdx = 0;
            }
            IE_ASSERT(inputs2Bytes->precision.size() == 2);
            IE_ASSERT(inputs4Bytes->precision.size() == 4);
        } else {
            // for mul both inputs should be 2 bytes precision
            IE_ASSERT(inputs2Bytes->precision.size() == 2);
            IE_ASSERT(inputs4Bytes->precision.size() == 2);
        }
    }

    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs4Bytes, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs4Bytes, 2);
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    void *ptr_inputs;
    void *ptr_outputs;
    void *ptr_weights;
    void *ptr_biases;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;
    dnn.InitAffineComponent(currentComponent,
                            num_rows_in + num_padding,
                            num_columns_in,
                            num_rows_out + num_padding,
                            inputs2Bytes->precision.size(),
                            outputs->precision.size(),
                            // TODO: only fp32 and Int16 tested
                            quantized == nullptr ? inputs2Bytes->precision.size() : 2,
                            quantized == nullptr ? inputs4Bytes->precision.size() : 4,
                            quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs,
                            ptr_weights,
                            ptr_biases,
                            true);

#ifdef PLOT
    cout << "IR layer : " << std::left << std::setw(20) << layer->name << "diagonal_"<< dnnComponentsForLayer.size() - 1 << "\n";
#endif

    size_t num_data_bytes_out =
        InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims)) * outputs->precision.size();

    size_t num_data_bytes_in =
        num_columns_in * (num_rows_in + num_padding) * inputs2Bytes->precision.size();

    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);
    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 1 - biasesLayerIdx);

    switch (eltwise._operation) {
        case EltwiseLayer::Sum:
            if (quantized == nullptr) {
                gnamem->readonly().push_value(ptr_weights, 1.0f, num_rows_out, 64);
            } else {
                auto scaledIdentity = quantized->_weights_quant.scale;

                #define FLOAT_TO_INT16(a) static_cast<int16_t>(((a) < 0)?((a) - 0.5):((a) + 0.5))

                auto quantizedIdentity = FLOAT_TO_INT16(std::min(scaledIdentity, static_cast<float>(INT16_MAX)));

                gnamem->readonly().push_value<int16_t>(ptr_weights, quantizedIdentity, num_rows_out, 64);
            }
            connectInput(layer, ptr_biases, num_data_bytes_in, 0, biasesLayerIdx);
            break;

        case EltwiseLayer::Prod:
            if (quantized == nullptr) {
                gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
            } else {
                gnamem->readonly().push_value<int32_t>(ptr_biases, 0, num_rows_out, 64);
            }
            connectInput(layer, ptr_weights, num_data_bytes_in, 0, biasesLayerIdx);
            break;

        default:
            THROW_GNA_EXCEPTION << "Unsupported eltwise operation: " << eltwise._operation;
    }
}

void GNAPlugin::AffinePrimitive(InferenceEngine::CNNLayerPtr layer, bool isDiag) {
    auto &weightable = dynamic_cast<WeightableLayer &>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = isDiag ? num_rows_in : FROM_IR_DIM(outputs, 1);
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    void *ptr_inputs;
    void *ptr_outputs;
    void *ptr_weights;
    void *ptr_biases;

    // TODO: questionable why for biases that are no in IR we inventing precision
    auto biasPrecision = weightable._biases ? weightable._biases->precision() : outputs->precision;

    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;

#ifdef PLOT
    cout << "IR layer : " << std::left << std::setw(20) << layer->name << (isDiag ? "diagonal_" : "affine_") << dnnComponentsForLayer.size() - 1 << "\n";
#endif

    dnn.InitAffineComponent(currentComponent,
                            num_rows_in + num_padding,
                            num_columns_in,
                            num_rows_out,
                            inputs->precision.size(),
                            outputs->precision.size(),
                            weightable._weights->precision().size(),
                            biasPrecision.size(),
                            quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs,
                            ptr_weights,
                            ptr_biases,
                            isDiag);

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims))
        * outputs->precision.size();

    size_t num_data_bytes_in = num_columns_in * (num_rows_in + num_padding) * inputs->precision.size();

    auto connectionInfo = connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);

    auto transpose = false;
    auto transposedRows = 0;
    auto transposedCols = 0;

    if (0 && connectionInfo.needTransposeWeights) {
        // direct order is 0, 1, 2, 3, supported order is only 0,3,2,1 where dim 2 is usually equals to 1
        auto permuteOrder = connectionInfo.permute->GetParamAsInts("order");
        if (permuteOrder != vector<int>({0, 3, 2, 1})) {
            THROW_IE_EXCEPTION << "[GNA plugin] Unsupported permute order: was " << layer->GetParamAsString("order") <<
                               ", but only support 0, 3, 2, 1";
        }

        /**
         * TODO: weights transpose happened after quantisation might result in poor quality for in 8 - move this to passes
         */
        if (weightable._weights->precision() == Precision::I8) {
            THROW_IE_EXCEPTION << "[GNA plugin] Unsupported permute operation for 8 bit weights for layer: " << layer->name;
        }

        // this affine connected to convolution via pool or activation
        gnalog() << "Transposing weights for layer: " << layer->name << "\n";

        transpose = !isDiag;
        transposedRows = connectionInfo.permute->input()->getDims()[3];
        transposedCols = connectionInfo.permute->input()->getDims()[1];
    }

    if (num_padding == 0) {
        if (!transpose) {
            gnamem->readonly().push_ptr(ptr_weights,
                                        weightable._weights->cbuffer().as<const void *>(),
                                        weightable._weights->byteSize(),
                                        64);
        } else {
            gnamem->readonly().push_initializer(ptr_weights, weightable._weights->byteSize(), [=](void * data, size_t size) {
                for (int k = 0; k < (isDiag ? 1 : num_rows_out); k++) {
                    auto rowOffset = k * transposedRows * transposedCols * weightable.precision.size();
                    auto cbuffer = weightable._weights->cbuffer().as<const uint8_t *>() + rowOffset;
                    auto u8Data = reinterpret_cast<uint8_t *>(data) + rowOffset;
                    for (int j = 0; j < transposedCols; j++) {
                        for (int i = 0; i < transposedRows; i++) {
                            auto offsetWrite = (transposedRows * j + i) * weightable.precision.size();
                            auto offsetRead = (i * transposedCols + j) * weightable.precision.size();
                            std::memcpy(u8Data + offsetWrite, cbuffer + offsetRead, weightable.precision.size());
                        }
                    }
                }
            }, 64);
        }
    } else {
        if (transpose) {
            THROW_GNA_EXCEPTION << "transpozed weights with non zero padding not yet supported";
        }
        auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
        auto paddedWeights = isDiag ? elementsIn : elementsIn * num_rows_out;
        auto paddedWeightsSize = paddedWeights * weightable.precision.size();

        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void * data, size_t size) {
            for (int i = 0; i < (isDiag ? 1 : num_rows_out); i++) {
                memcpy(data,
                       weightable._weights->cbuffer().as<const uint8_t *>() + num_rows_in * i * weightable.precision.size(),
                       num_rows_in * weightable.precision.size());
                data = reinterpret_cast<uint8_t *>(data) + (num_rows_in + num_padding) * weightable.precision.size();
            }
        }, 64);
    }

    if (weightable._biases) {
        gnamem->readonly().push_ptr(ptr_biases,
                         weightable._biases->cbuffer().as<const void *>(),
                         weightable._biases->byteSize(),
                         64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAPlugin::FillWeightOfAligningFilter(InferenceEngine::CNNLayerPtr layer, void* ptrWeights, size_t offset, bool isQuantized) {
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);

    if (!ptrWeights) {
        THROW_GNA_EXCEPTION << "Weights memory is not allocated!!!";
    }

    gnamem->readonly().push_initializer(ptrWeights, num_rows_out * ALIGN(num_rows_in, 8) * layer->precision.size(), [=](void * data, size_t size) {
        int out = 0;
        for (int input = offset; input < num_rows_out + offset; ++input) {
            auto mem_ptr = reinterpret_cast<uint8_t *>(data) + input * layer->precision.size() + out * ALIGN(num_rows_in, 8) * layer->precision.size();
            if (!isQuantized) {
                auto float_ptr = reinterpret_cast<float *>(mem_ptr);
                *float_ptr = 1.0f;
           } else {
               auto int_ptr = reinterpret_cast<uint16_t *>(mem_ptr);
               *int_ptr = 1;
           }
            ++out;
        }
    }, 64);
}

void GNAPlugin::AffineFilterPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto filterLayer = dynamic_cast<InferenceEngine::WeightableLayer *> (layer.get());

    if (filterLayer == nullptr) {
        return;
    }

    std::string& name = filterLayer->name;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    // we look for this concat layer pointer in extra concat map
    auto prevLayer = CNNNetPrevLayer(layer.get(), 0);
    if (!LayerInfo(prevLayer).isSplit() && !LayerInfo(prevLayer).isSlice()) {
        THROW_GNA_EXCEPTION << "Case  with Affine Aligning Filter for not Split/Slice layers is not implemented yet!";
    }

    void *ptr_inputs;
    void *ptr_outputs;
    void *ptr_weights;
    void *ptr_biases;

    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_rows_in = filterLayer->_weights->size() / num_rows_out;

    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    gnalog() << "Filter " << layer->name << " is being inserted...\n";
    auto biasPrecision = filterLayer->_biases ? filterLayer->_biases->precision() : outputs->precision;
    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;
    dnn.InitAffineComponent(currentComponent,
                            num_rows_in + num_padding,
                            num_columns_in,
                            num_rows_out,
                            inputs->precision.size(),
                            outputs->precision.size(),
                            filterLayer->_weights->precision().size(),
                            biasPrecision.size(),
                            quantized == nullptr ? 1 : quantized->_weights_quant.scale,
                            quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                            ptr_inputs,
                            ptr_outputs,
                            ptr_weights,
                            ptr_biases,
                            false);

    size_t num_data_bytes_out =
                InferenceEngine::details::product(
                                        begin(outputs->dims), end(outputs->dims)) * 4;

    size_t num_data_bytes_in = num_columns_in *
                            ALIGN(num_rows_in, 8) * inputs->precision.size();

    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);

    if (num_padding == 0) {
        gnamem->readonly().push_ptr(ptr_weights,
                                filterLayer->_weights->cbuffer().as<const void *>(),
                                filterLayer->_weights->byteSize(),
                                                            64);
    } else {
        auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
        auto paddedWeights = elementsIn * num_rows_out;
        auto paddedWeightsSize = paddedWeights * filterLayer->precision.size();

        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void * data, size_t size) {
            for (int i = 0; i < num_rows_out; i++) {
                std::memcpy(data,
                       filterLayer->_weights->cbuffer().as<const uint8_t *>() + num_rows_in * i * filterLayer->precision.size(),
                       num_rows_in * filterLayer->precision.size());
                data = reinterpret_cast<uint8_t *>(data) + (num_rows_in + num_padding) * filterLayer->precision.size();
            }
        }, 64);
    }

    if (filterLayer->_biases) {
        gnamem->readonly().push_ptr(ptr_biases,
                         filterLayer->_biases->cbuffer().as<const void *>(),
                         filterLayer->_biases->byteSize(),
                         64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAPlugin::PWLPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto *generic = dynamic_cast<GenericLayer *>(layer.get());
    std::string type;
    std::vector<intel_pwl_segment_t> ptr_pwl_segments;
    uint32_t num_rows;
    uint32_t num_columns;
    void *ptr_inputs;
    void *ptr_outputs;

    do {
        if (generic == nullptr) {
            type = layer->type;
            break;
        }

        if (CaselessEq<string>()(layer->type, "activation")) {
            type = generic->GetParamAsString("type");
            break;
        } else {
            type = layer->type;
            break;
        }
    } while (false);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    float output_scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;

    auto orientation = (num_cnn_rows_out > 0) ? kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;

    if (inputs->dims.size() == 4) {
        num_columns = FROM_IR_DIM(inputs, 3) * FROM_IR_DIM(inputs, 1);
        num_rows = 1;
    } else {
        num_columns = FROM_IR_DIM(inputs, 2);
        num_rows = FROM_IR_DIM(inputs, 1);
    }

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->dims), end(outputs->dims))
        * outputs->precision.size();

    size_t num_data_bytes_in = InferenceEngine::details::product(begin(inputs->dims), end(inputs->dims))
        * inputs->precision.size();

    static caseless_unordered_map<std::string, DnnActivationType> supportedActivations = {
        {"sigmoid", kActSigmoid},
        {"tanh", kActTanh},
        {"relu", kActRelu},
        {"leakyrelu", kActLeakyRelu},
        {"clamp", kActKaldiLstmClipping},
        {"identity", kActIdentity}
    };

    auto it = supportedActivations.find(type);
    if (it == supportedActivations.end()) {
        THROW_GNA_EXCEPTION << "Activation function type not yet supported: " << type;
    }
    auto activation_type = DnnActivation::fromType(it->second);
    if (it->second == kActRelu) {
        auto reluLayer = dynamic_cast<ReLULayer *>(layer.get());
        activation_type.negative_slope = reluLayer != nullptr ? reluLayer->negative_slope : 0.0f;
    } else {
        activation_type.negative_slope = 0.0f;
    }

    // TODO: need to take graph dependency instead of linear
    auto &prevComponent = dnnComponentsForLayer.back().second;
    dnnComponentsForLayer.emplace_back(layer->name, intel_dnn_component_t());
    auto &currentComponent = dnnComponentsForLayer.back().second;

    intel_pwl_segment_t *ptr_pwl_segments_target = nullptr;

    if (!inputs->precision.is_float()) {
        // TODO: generalize activation function code
        // now that scale factors are known, create PWL approximations to activation functions
        float input_scale_factor = dnn.OutputScaleFactor(prevComponent);
        if (uniformPwlDesign) {
            switch (activation_type) {
                case kActSigmoid:ptr_pwl_segments.resize(SIGMOID_NUM_SEGMENTS);
                    break;
                case kActTanh:ptr_pwl_segments.resize(TANH_NUM_SEGMENTS);
                    break;
                case kActRelu:ptr_pwl_segments.resize(RELU_NUM_SEGMENTS);
                    break;
                case kActLeakyRelu:ptr_pwl_segments.resize(RELU_NUM_SEGMENTS);
                    break;
                case kActKaldiLstmClipping:
                case kActIdentity:ptr_pwl_segments.resize(IDENTITY_NUM_SEGMENTS);
                    break;
                case kActCustom:
                default:THROW_GNA_EXCEPTION << "Activation function type not yet supported " << activation_type;
            }
            PwlDesign16(activation_type,
                        &*ptr_pwl_segments.begin(),
                        static_cast<uint32_t>(ptr_pwl_segments.size()),
                        input_scale_factor,
                        output_scale_factor);
        } else {
            PwlDesignOpt16(activation_type,
                           ptr_pwl_segments,
                           input_scale_factor,
                           output_scale_factor);
        }
        ptr_pwl_segments_target = reinterpret_cast<intel_pwl_segment_t *>(&ptr_pwl_segments_target);
    }

    dnn.InitPiecewiseLinearComponent(currentComponent,
                                     activation_type,
                                     orientation,
                                     num_rows,
                                     num_columns,
                                     inputs->precision.size(),
                                     outputs->precision.size(),
                                     ptr_pwl_segments.size(),
                                     output_scale_factor,
                                     ptr_inputs,
                                     ptr_outputs,
                                     ptr_pwl_segments_target);
#ifdef PLOT
#define GET_ACTIVATION_NAME(name)\
case name:\
    actName = #name;\
    break;
    string actName = "unknown";
    switch (activation_type) {
        GET_ACTIVATION_NAME(kActSigmoid);
        GET_ACTIVATION_NAME(kActTanh);
        GET_ACTIVATION_NAME(kActRelu);
        GET_ACTIVATION_NAME(kActLeakyRelu);
        GET_ACTIVATION_NAME(kActKaldiLstmClipping);
        GET_ACTIVATION_NAME(kActIdentity);
    }
    cout << "IR layer : " << std::left << std::setw(20) << layer->name <<  actName << "_" << dnnComponentsForLayer.size() - 1 <<"\n";
#endif

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, ptr_inputs, num_data_bytes_out);

    if (ptr_pwl_segments_target != nullptr) {
        gnamem->readonly().push_local_ptr(ptr_pwl_segments_target,
                                          &ptr_pwl_segments.front(),
                                          ptr_pwl_segments.size() * sizeof(intel_pwl_segment_t),
                                          64);
    }
}


void GNAPlugin::PermutePrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto layerOrder = layer->GetParamAsInts("order");

    if (layerOrder != vector<int>({0, 3, 2, 1})) {
        THROW_IE_EXCEPTION << "[GNA plugin] Unsupported permute order: was " << layer->GetParamAsString("order") <<
                           ", but only support 0,3,2,1";
    }
}

class LayersBuilder {
    using CreatorFnc = std::function<void(GNAPlugin*, CNNLayerPtr)>;

 public:
    LayersBuilder(const std::vector<std::string> &types, CreatorFnc callback) {
        for (auto && str : types) {
            getStorage()[str] = callback;
        }
    }
    static caseless_unordered_map<std::string, CreatorFnc> &getStorage() {
        static caseless_unordered_map<std::string, CreatorFnc> LayerBuilder;
        return LayerBuilder;
    }
};

#define CREATE(name) [](GNAPlugin *p, CNNLayerPtr l) {p->name(l);}
void SKIP(GNAPlugin*, CNNLayerPtr) {}

void GNAPlugin::CreateLayerPrimitive(CNNLayerPtr layer) {
    static const LayersBuilder layersBuilder[] = {
        {{"Input"}, [](GNAPlugin*, CNNLayerPtr l) {}},  // skip input layers they are not used in GNA lib, only as a memory blobs
        {{"FullyConnected", "InnerProduct"}, CREATE(AffinePrimitive)},
        {{"ScaleShift"}, CREATE(DiagonalPrimitive)},
        {{"AffineFilter"}, CREATE(AffineFilterPrimitive)},
        {{"Eltwise"},
         CREATE(EltwisePrimitive)},  // same as diagonal while weights are not taken from network, rather than from another output
        {{"Split"}, SKIP},  // skip information about which part of prev layer need to consume handle during layer creation
        {{"Slice"}, SKIP},
        {{"clamp", "sigmoid", "relu", "tanh", "identity"}, CREATE(PWLPrimitive)},
        {{"Convolution"}, CREATE(ConvolutionPrimitive)},
        {{"Permute"}, CREATE(PermutePrimitive)},  // permute of certain form (2D transpose) can be assimilated in followed FC layer
        {{"Pooling"}, CREATE(PoolingPrimitive)},
        {{"Power"} , CREATE(PowerPrimitive)},
        {{"Concat"}, CREATE(ConcatPrimitive)},
        {{"Reshape"}, SKIP},  // TODO: handled not in GNA but rather in GNA plugin
        {{"Crop"}, CREATE(CropPrimitive)},
        {{"Copy"}, CREATE(CopyPrimitive)},
    };
    auto it = LayersBuilder::getStorage().find(layer->type);
    if (it != LayersBuilder::getStorage().end()) {
        it->second(this, layer);
    } else {
        THROW_GNA_EXCEPTION << "Unsupported layer: " << layer->name << ":" << layer->type;
    }
}


GNAPlugin::GNAPlugin(const std::map<std::string, std::string>& configMap) {
    SetConfig(configMap);
}

GNAPluginNS::GNAPlugin::LayerType GNAPlugin::LayerTypeFromStr(const std::string &str) const {
    static const caseless_map<std::string, GNAPlugin::LayerType> LayerNameToType = {
        { "Input" , Input },
        { "Convolution" , Convolution },
        { "ReLU" , ReLU },
        { "Sigmoid" , Sigmoid },
        { "TanH" , TanH },
        { "Pooling" , Pooling },
        { "FullyConnected" , FullyConnected },
        { "InnerProduct" , InnerProduct},
        { "Split" , Split },
        { "Slice" , Slice },
        { "Eltwise" , Eltwise },
        { "Reshape" , Reshape },
        { "ScaleShift" , ScaleShift },
        { "Clamp" , Clamp },
        { "Concat" , Concat },
        { "Copy", Copy },
        { "Permute" , Permute },
        { "Power" , Power},
        { "Memory" , Memory },
        { "Crop" , Crop }
    };
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return NO_TYPE;
}

bool GNAPlugin::AreLayersSupported(ICNNNetwork& network, std::string& errMessage) {
    CNNLayerSet inputLayers;
    InferenceEngine::InputsDataMap inputs;
    std::unordered_set<CNNLayer *> allLayers;
    auto specifiedDevice = network.getTargetDevice();
    auto network_precision = network.getPrecision();
    network.getInputsInfo(inputs);
    auto network_input_precision = inputs.begin()->second->getInputPrecision();
    auto batch_size = network.getBatchSize();
    if (network_precision != Precision::FP32) {
        errMessage = "The plugin does not support networks with " + std::string(network_precision.name()) + " format.\n";
        return false;
    }
    if (network_input_precision != Precision::FP32 &&
        network_input_precision != Precision::I16 &&
        network_input_precision != Precision::U8) {
        errMessage = "The plugin does not support input precision with " + std::string(network_input_precision.name()) + " format.\n";
        return false;
    }
    if (specifiedDevice != InferenceEngine::TargetDevice::eCPU &&
        specifiedDevice != InferenceEngine::TargetDevice::eGNA &&
        specifiedDevice != InferenceEngine::TargetDevice::eDefault) {
        errMessage = "The plugin does not support target device: " + std::string(getDeviceName(specifiedDevice)) + ".\n";
        return false;
    }

    if (inputs.empty()) {
        errMessage = "Network is empty (GNA)\n";
        return false;
    }

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty()) {
        errMessage = "Network consists of input layer only (GNA)\n";
        return false;
    }

    bool check_result = true;
    InferenceEngine::details::UnorderedDFS(allLayers,
                                           secondLayers.begin()->second,
                                           [&](const CNNLayerPtr layer) {
                                                if (LayerTypeFromStr(layer->type) == NO_TYPE) {
                                                    errMessage = "Layer is unsupported by GNA: " + layer->name + ":" + layer->type + "\n";
                                                    check_result =  false;
                                                }
                                                if (batch_size != 1 && LayerInfo::isBatchSizeConstrained(layer->type)) {
                                                    errMessage = "topology with layer: " + layer->name + ", type: " + layer->type +
                                                                 ", and batch size(" + to_string(batch_size) + ") != 1 not supported";
                                                    check_result =  false;
                                                }
                                            }, false);

    return check_result;
}

float GNAPlugin::get_input_scale_factor() const {
    return input_scale_factor.empty() ? 1.0 : input_scale_factor.begin()->second;
}

void GNAPlugin::LoadNetwork(ICNNNetwork &network) {
    //  Check the input network
    std::string error;
    if (!AreLayersSupported(network, error)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // network optimisation phases
    auto run_passes = [&] (CNNNetPtr network) {
        auto layers = CNNNetSortTopologically(*network.get());
        substitutePRelu(layers);
        layers = CNNNetSortTopologically(*network.get());
        reorderMaxPool(layers);
        //  ToDo sort if bool flag "changed"
        //  returned from insertion function
        insertAligningFilterLayer(layers);

#if ENABLE_AUTO_PERMUTE
        layers = CNNNetSortTopologically(*network.get());
        reversePermutations(layers);
#endif
        layers = CNNNetSortTopologically(*network.get());
        insertIdentityLayer(layers);
        layers = CNNNetSortTopologically(*network.get());
        insertCopyLayer(layers);
        layers = CNNNetSortTopologically(*network.get());
        insertDiagonalLayer(layers);
        layers = CNNNetSortTopologically(*network.get());
        substituteScaleShiftBroadCast(layers);
    };

    Config supported = Config({
        {TargetDevice::eGNA, Precision::FP32, [&](InferenceEngine::ICNNNetwork &network) -> CNNNetworkPtr {
            if (gnaPrecision == Precision::I16) {
                ModelQuantizer<QuantI16> q;
                return q.quantize(network, run_passes, get_input_scale_factor());
            }

            if (gnaPrecision == Precision::I8) {
                ModelQuantizer<QuantI8> q;
                return q.quantize(network, run_passes, get_input_scale_factor());
            }
            THROW_GNA_EXCEPTION << "no mans land for GNA precision";
        }},
        // TODO: need to have advanced precision matcher based on layers/biases
        {TargetDevice::eGNA, Precision::MIXED},
        {TargetDevice::eGNA, Precision::I16},
        {TargetDevice::eCPU, Precision::FP32
#define EMULATE_GNA_API_LAYERS
#ifdef  EMULATE_GNA_API_LAYERS
            , [&](InferenceEngine::ICNNNetwork & network) {
            auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
                return lp;
            };
            auto copiedNet = InferenceEngine::CNNNetCopy(network, visitor);
            run_passes(copiedNet);

            return copiedNet;
        }
#endif
    }
    });

    supported.setDefaultDevice(TargetDevice::eGNA);
    auto newNet = supported.find_configuration(network).convert(network);



    // creating intel dnn_t structures from network
    auto sortedNet = CNNNetSortTopologically(*newNet);
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
            fillConcatConnections(layer);
        } else if (layerInfo.isSplit() || layerInfo.isSlice()) {
            fillSplitConnections(layer);
        }
        sortedNoMem.push_back(layer);
    }

    // fill in extra storage with memory layers
    fillMemoryConnections(memoryPairs);

    if (memory_connection.size() != 0) {
        gna_lib_async_threads_num = 1;
    }

    auto networkPrecision = newNet->getPrecision();

    if (!networkPrecision.is_float()) {
        gnadevice.reset(new GNADeviceHelper(gna_proc_type,
                                            gna_lib_async_threads_num,
                                            gna_openmp_multithreading,
                                            performance_counting));
        gnamem.reset(new gna_memory_type(
                make_polymorph<GNAAllocator>(*gnadevice.get()), PAGE_SIZE_BYTES));
    } else {
        gnamem.reset(new gna_memory_type(make_polymorph<std::allocator<uint8_t>>()));
    }

    // keep inputs information and create input primitives
    newNet->getInputsInfo(inputsDataMap);
    if (inputsDataMap.empty()) {
        THROW_GNA_EXCEPTION << " No inputs for the topology";
    }

    // keep output dims
    newNet->getOutputsInfo(outputsDataMap);
    if (outputsDataMap.empty()) {
        THROW_GNA_EXCEPTION << "No outputs for the topology";
    }
    if (outputsDataMap.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer topologies with more than one output";
    }
    outputDims = outputsDataMap.begin()->second->dims;

    for (auto && input : inputsDataMap) {
        get_ptr_inputs_global(input.first).resize(gna_lib_async_threads_num);
    }

    ptr_outputs_global.resize(gna_lib_async_threads_num);
    // CreatingLayer primitives
    // TODO: solely gna_example convolution hack
    num_feature_maps = 1;
    for (auto layer = sortedNoMem.begin(); layer != sortedNoMem.end(); ++layer) {
        CreateLayerPrimitive(*layer);
    }
    if (dnnComponentsForLayer.empty()) {
        THROW_GNA_EXCEPTION << "No outputs found in dnn components structure";
    }

    DnnComponentsForLayer::iterator output_component = std::find_if(dnnComponentsForLayer.begin(),
                                                        dnnComponentsForLayer.end(),
                                                        [&](const std::pair<std::string, intel_dnn_component_t>& v)
                                                        { return outputsDataMap.begin()->first == v.first; });

    if (output_component == dnnComponentsForLayer.end()) {
        // likely layer is fused. Take last one
        auto it = dnnComponentsForLayer.begin();
        std::advance(it, dnnComponentsForLayer.size() - 1);
        output_component = it;
        gnalog() << "Output layer "<< outputsDataMap.begin()->first
            << " has not been found in component list. Took  "
            << output_component->first << " instead \n" << std::flush;
    }
    gnamem->bind_ptr(&ptr_outputs_global.front(), &output_component->second.ptr_outputs);

    // make room for active list
    gnamem->reserve_ptr(nullptr, ALIGN64(output_component->second.num_bytes_per_output * output_component->second.num_rows_out));

    void *pParallelExecutionData  = nullptr;

    // reserving more bytes for intermidiate data in parallel case - TODO: this works incorrectly in compact mode at lest
    rwSegmentSize = gnamem->getRWBytes();
    if (gna_lib_async_threads_num > 1) {
        gnamem->reserve_ptr(&pParallelExecutionData, gnamem->getRWBytes() * (gna_lib_async_threads_num - 1));
    }

    gnamem->commit();

    dnn.Init(gnamem->getBasePtr(),
             gnamem->getTotalBytes(),
             networkPrecision.is_float() ? kDnnFloat : kDnnInt,
             1);

    // TODO: this copy unneed infact we can directly create gna structs from list
    for (auto &element : dnnComponentsForLayer) {
        dnn.component.push_back(element.second);
    }

    // in fp32 mode last PWL cannot be computed without that
    dnn.InitActiveList(NULL);

    nnets.push_back(std::make_tuple(make_shared<CPPWrapper<intel_nnet_type_t>>(), -1, InferenceEngine::BlobMap()));

    if (!networkPrecision.is_float()) {
        // number of layer gets calculated inside that InitGNAStruct function
        dnn.InitGNAStruct(&std::get<0>(nnets.front())->obj);
    }

    // creating same gna RW segment for parallel infer requests
    for (int i = 1; i != gna_lib_async_threads_num; i++) {
        nnets.push_back(std::make_tuple(make_shared<CPPWrapper<intel_nnet_type_t>>(), -1, InferenceEngine::BlobMap()));

        // this can be improved by just copy all structures, but we are too lazy
        dnn.InitGNAStruct(&std::get<0>(nnets.back())->obj);

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

        for (auto &&input : ptr_inputs_global_storage) {
            relocate(input[i], input[0]);
        }

        relocate(ptr_outputs_global[i], ptr_outputs_global[0]);
        for (int j = 0; j != std::get<0>(nnets.front())->obj.nLayers; j++) {
            auto & layer = std::get<0>(nnets[i])->obj.pLayers[j];

            relocate(layer.pInputs, layer.pInputs);
            relocate(layer.pOutputs, layer.pOutputs);
            relocate(layer.pOutputsIntermediate, layer.pOutputsIntermediate);
        }
    }

    // calculating input orientation without memory layers, since their orientation not changed during infer right now
    std::unordered_map<string, string> skippedLayers;
    for (auto &layer : sortedNet) {
        for (int i = 0; CNNNetHasPrevLayer(layer.get(), i); i++) {
            auto prevLayer = CNNNetPrevLayer(layer.get(), i);
            if (!skippedLayers.count(prevLayer->name)) {
                if (CNNNetHasPrevLayer(prevLayer.get())) {
                    continue;
                }

                // we are in the one of input layers
                if (LayerInfo(prevLayer).isMemory()) {
                    continue;
                }
            }

            auto dnnLayer = findDnnLayer(layer);
            string inputName = prevLayer->name;
            if (skippedLayers.count(prevLayer->name)) {
                inputName = skippedLayers[prevLayer->name];
            }

            // non functional layer - skipped by gna
            if (nullptr == dnnLayer) {
                // storing input name for skipped layer
                skippedLayers[layer->name] = inputName;
                continue;
            }

            // input orientation might be already initialized, thus verify that it matches
            if (!orientation_in.count(inputName)) {
                orientation_in[inputName] = dnnLayer->orientation_in;
            } else {
                if (orientation_in[inputName] != dnnLayer->orientation_in) {
                    THROW_GNA_EXCEPTION << "orientation for input layer: " << inputName << "cannot be calculated";
                }
            }
        }
    }

    orientation_out = output_component->second.orientation_out;
    num_bytes_per_output = output_component->second.num_bytes_per_output;

    if (sortedNet.empty()) {
        THROW_GNA_EXCEPTION << "Sorted network is empty";
    }

    // find output layer
    auto output = std::find_if(sortedNet.begin(),
                                sortedNet.end(),
                                [&](const CNNLayerPtr& v)
                                { return outputsDataMap.begin()->first == v.get()->name; });
    if (output == sortedNet.end()) {
        // likely layer is fused. Take last one
        auto it = sortedNet.begin();
        std::advance(it, sortedNet.size() - 1);
        output = it;
    }
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(*output);
    output_scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;

    num_rotate_rows = dnn.num_rotate_rows;
    num_rotate_columns = dnn.num_rotate_columns;

    DumpXNNToFile();

#ifdef PLOT
    dnn.WriteGraphWizModel("graph.dot");
    // ExportGnaNetworkAndrzej("layers/loaded_from_ir", &nnet->obj);
#endif
}
void GNAPlugin::DumpXNNToFile() const {
    // TODO: output  precision as well as pointer might be incorrect, LSTM for sure
    // gna looks automatically set layer 0 as output and adjust it's pointer / precision/ size respectively
    if (!dumpXNNPath.empty()) {
        if (!gnadevice) {
            THROW_GNA_EXCEPTION << "Cannot generate XNNDump for float network";
        }
        auto dump = gnadevice->dumpXnn(&std::get<0>(nnets.front())->obj, ptr_active_indices, num_active_indices);
        dump.header.rw_region_size = gnamem->getRWBytes();
        dump.header.input_scaling_factor = get_input_scale_factor();
        dump.header.output_scaling_factor = output_scale_factor;
        std::ofstream dumpStream(dumpXNNPath, std::ios::out | std::ios::binary);
        dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(intel_gna_model_header));
        dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.model_size);
    }
}

void RotateFeatures(uint8_t *ptr_feat,
                    size_t element_size,
                    uint32_t num_feature_vectors,
                    uint32_t num_feature_vector_elements,
                    uint32_t num_rotate_rows,
                    uint32_t num_rotate_columns) {
    if (num_feature_vector_elements == num_rotate_rows * num_rotate_columns) {
        std::vector<uint8_t> temp(num_feature_vector_elements * element_size);
        for (uint32_t k = 0; k < num_feature_vectors; k++) {
            uint8_t *ptr_in = ptr_feat + k * num_feature_vector_elements * element_size;
            for (uint32_t i = 0; i < num_rotate_rows; i++) {
                for (uint32_t j = 0; j < num_rotate_columns; j++) {
                    ie_memcpy(&temp.front() + (j * num_rotate_rows + i)*element_size,
                              temp.size() - (i * num_rotate_columns + j)*element_size,
                              ptr_in + (i * num_rotate_columns + j)*element_size,
                              element_size);
                }
            }
            memcpy(ptr_in, &temp.front(), num_feature_vector_elements * element_size);
        }
    } else {
        THROW_GNA_EXCEPTION << "Rotate dimensions (" << num_rotate_rows << "," << num_rotate_columns
                           <<") do not match buffer length of "<< num_feature_vector_elements <<" in RotateFeatures()!";
    }
}

uint32_t GNAPlugin::QueueInference(const InferenceEngine::BlobMap &inputs, InferenceEngine::BlobMap &result) {
    auto freeNnet = std::find_if(std::begin(nnets), std::end(nnets), [](decltype(nnets.front()) & item) {
        return std::get<1>(item) == -1;
    });

    if (freeNnet == nnets.end()) {
        if (memory_connection.size() != 0) {
            Wait(0);
            freeNnet = nnets.begin();
        } else {
            THROW_IE_EXCEPTION << as_status << REQUEST_BUSY
                               << "GNA executable network has max of "
                               << static_cast<uint32_t >(gna_lib_async_threads_num)
                               << " parallel infer requests, please sync one of already running";
        }
    }


    auto nnet = std::get<0>(*freeNnet).get();
    auto idx = static_cast<uint32_t>(std::distance(std::begin(nnets), freeNnet));

    for (auto &input : inputs) {
        auto inputLayout = input.second->layout();
        if (inputLayout != Layout::NC && inputLayout != Layout::CN && inputLayout != NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::NC or Layout::CN, but was: "
                                << input.second->layout();
        }
        if (inputLayout == NCHW) {
            inputLayout = NC;
        }
        auto is2D = input.second->layout() == Layout::NC || input.second->layout() == Layout::CN;

        if (!ptr_inputs_global_id.count(input.first)) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (get_ptr_inputs_global(input.first)[idx] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << idx << " not set";
        }

        if (orientation_in[input.first] == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input orientation for " << input.first << " not set";
        }

        if (orientation_out == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : output orientation not set";
        }

        auto dims = input.second->dims();

        ImportFrames(get_ptr_inputs_global(input.first)[idx],
                     input.second->cbuffer().as<float *>(),
                     input.second->precision(),
                     orientation_in[input.first],
                     dims[dims.size() - 1],
                     is2D ? dims[1] : dims[dims.size() - 1],
                     is2D ? dims[0] : dims[0] * dims[1] * dims[2],
                     is2D ? dims[0] : dims[0] * dims[1] * dims[2]);
        bool isOneChannel = input.second->getTensorDesc().getDims()[1] == 1;
        if (((inputLayout == Layout::NC || inputLayout == Layout::NCHW)
            != (orientation_in[input.first] == kDnnInterleavedOrientation))
            && !isOneChannel) {
            RotateFeatures(reinterpret_cast<uint8_t *>(get_ptr_inputs_global(input.first)[idx]),
                           gnadevice ? 2 : 4,
                           // TODO: only works for cnn4a and google command so far
                           dims[dims.size() - 1],
                           is2D ? dims[0] : dims[0] * dims[2],  // num_feature_vectors looks batch should be there
                           num_rotate_rows,
                           num_rotate_columns);
        }
    }

    if (!gnadevice) {
        dnn.Propagate();
        std::get<1>(*freeNnet) = 1;
    } else {
        std::get<1>(*freeNnet) = gnadevice->propagate(&nnet->obj, ptr_active_indices, num_active_indices);
    }
    std::get<2>(*freeNnet) = result;
    return idx;
}

void GNAPlugin::Wait(uint32_t idx) {
    // already synced TODO: might be copy required ???
    if (std::get<1>(nnets[idx]) == -1) return;

    if (gnadevice) {
        gnadevice->wait(std::get<1>(nnets[idx]));
    }

    std::get<1>(nnets[idx]) = -1;
    auto & result = std::get<2>(nnets[idx]);
#ifdef PLOT
    dnn.BeginNewWrite();
    if (dnn.num_components() != 0) {
        dnn.WriteDnnText("Net_.txt", kDnnFloat);
        dnn.WriteInputAndOutputText();
    }
    dnn.WriteInputAndOutputTextGNA(&std::get<0>(nnets.front())->obj);
#endif
    if (result.size() != 1) {
        THROW_GNA_EXCEPTION << "Invalid number of outputs for infer request: " << result.size() << ",  only 1 supported";
    }
    auto & output = *result.begin()->second;

    if (output.layout() == Layout::NC) {
        // TODO: rotate can be incorporated with exporting - used only in unit tests so far
        // TODO: restore:
//        if (orientation_out != kDnnInterleavedOrientation) {
//            if (inputs.size() != 1) {
//                THROW_GNA_EXCEPTION << "Invalid number of inputs for  for deinterleave " << inputs.size()
//                                    << ", only 1 supported";
//            }
//            auto dims = inputs.begin()->second->dims();
//            RotateFeatures(reinterpret_cast<uint8_t*>(ptr_outputs_global),
//                           gnadevice ? 2 : 4,
//                           dims[dims.size() - 1],
//                           dims[0],  // num_feature_vectors looks batch should be there
//                           dims[0],
//                           dims[dims.size() - 1]);
//        }
        // we concider the last layer as output ...
        size_t output_layer_index = std::max(0, static_cast<int>(std::get<0>(nnets[idx])->obj.nLayers - 1));
        if (gnadevice && std::get<0>(nnets[idx])->obj.pLayers[output_layer_index].pOutputs != ptr_outputs_global[idx]) {
            // ...as this is not true, we should look for output layer index
            for (int j = 0; j != std::get<0>(nnets[idx])->obj.nLayers; j++) {
                if (std::get<0>(nnets[idx])->obj.pLayers[j].pOutputs == ptr_outputs_global[idx]) {
                    output_layer_index = j;
                    break;
                }
            }
        }

        ExportScores(output.buffer(),
                     ptr_outputs_global[idx],
                     orientation_out,
                     output.dims()[output.dims().size() - 1],
                     output.dims()[1],
                     output.dims()[0],
                     output.dims()[0],
                     output.dims()[0],
                     // TODO: create better getter consider multiple outputs case
                     gnadevice ? std::get<0>(nnets[idx])->obj.pLayers[output_layer_index].nBytesPerOutput : sizeof(float),
                     sizeof(float));
    } else if (output.layout() != Layout::CN) {
        THROW_GNA_EXCEPTION << "Expected output blob to have Layout::NC or Layout::CN. But was " << output.layout();
    }

    if (gnadevice) {
#ifdef PLOT
        FILE *f = nullptr;
        static int num_infers = 0;
        {
            f = fopen("ex_scores.txt", "w");
        }
        num_infers++;
        if (f) {
            for (int i = 0; i < output.dims()[1]; i++) {
                for (int j = 0; j < output.dims()[0]; j++) {
                    fprintf(f, "%d ", output.cbuffer().as<int32_t *>()[output.dims()[0] * i + j]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n\n");
        }
#endif
        ConvertToFloat(output.buffer(),
                       output.buffer(),
                       output.dims()[0],
                       output.dims()[1],
                       output_scale_factor);
#ifdef PLOT
        if (f) {
            for (int i = 0; i < output.dims()[1]; i++) {
                for (int j = 0; j < output.dims()[0]; j++) {
                    fprintf(f, "%.2f ", output.cbuffer().as<float *>()[output.dims()[0] * i + j]);
                }
                fprintf(f, "\n");
            }
            fclose(f);
        }
#endif
    }
}

void GNAPlugin::Reset() {
    for (auto && memLayer : memory_connection) {
        std::memset(memLayer.second.gna_ptr, 0, memLayer.second.reserved_size);
    }
    for (auto && concatLayer : concat_connection) {
        std::memset(concatLayer.second.gna_ptr, 0, concatLayer.second.reserved_size);
    }
}

void GNAPlugin::Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &output) {
    BlobMap bmInput;
    BlobMap bmOutput;
    if (inputsDataMap.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"<< "model accepts " << inputsDataMap.size() << "inputs";
    }
    if (outputsDataMap.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"<< "model accepts " << outputsDataMap.size() << "outputs";
    }

    bmInput[inputsDataMap.begin()->first] = std::shared_ptr<Blob>(const_cast<Blob*>(&input), [](Blob*){});
    bmOutput[outputsDataMap.begin()->first] = std::shared_ptr<Blob>(&output, [](Blob*){});
    Infer(bmInput, bmOutput);
}

void GNAPlugin::Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result) {
    Wait(QueueInference(input, result));
}

Blob::Ptr GNAPlugin::GetOutputBlob(InferenceEngine::Precision precision) {
    // need to have intermediate blob for interleave conversion
    InferenceEngine::Blob::Ptr outputBlob;
    outputBlob = make_blob_with_precision(precision, NC, outputDims);
    outputBlob->allocate();
    return outputBlob;
}

Blob::Ptr GNAPlugin::GetInputBlob(std::string name, InferenceEngine::Precision precision) {
    InferenceEngine::Blob::Ptr inputBlob;
    // need to have intermediate blob for interleave conversion
    // TODO: NCHW format support is experimental = c++ MO did insert reshape, while TF mo - not
    auto inputDims = inputsDataMap[name]->getDims();
    inputBlob = make_blob_with_precision(precision, inputDims.size() == 2 ? NC : NCHW, inputDims);
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::MemoryStateInternal::Ptr>  GNAPlugin::QueryState() {
    if (memory_connection.empty()) {
        return {};
    }

    return {std::make_shared<GNAMemoryState>(shared_from_this())};
}

InferenceEngine::IExecutableNetwork::Ptr GNAPlugin::ImportNetwork(const std::string &modelFileName) {
    // no need to return anything dueto weird design of internal base classes
    std::fstream inputStream(modelFileName, ios_base::in | ios_base::binary);
    if (inputStream.fail()) {
        THROW_GNA_EXCEPTION << "Cannot open file to import model: " << modelFileName;
    }

    auto header = GNAModelSerial::ReadHeader(inputStream);

    gnadevice.reset(new GNADeviceHelper(gna_proc_type,
                                        gna_lib_async_threads_num,
                                        gna_openmp_multithreading));
    gnamem.reset(new gna_memory_type(make_polymorph<GNAAllocator>(*gnadevice.get()), PAGE_SIZE_BYTES));

    void *basePtr = nullptr;
    gnamem->reserve_ptr(&basePtr, header.gnaMemSize);
    gnamem->commit();

    nnets.push_back(std::make_tuple(make_shared<CPPWrapper<intel_nnet_type_t>>(header.layersCount), -1, InferenceEngine::BlobMap()));
    std::get<0>(nnets.back())->obj.nGroup = header.nGroup;
    GNAModelSerial::MemoryType  mt;
    auto serial = GNAModelSerial(&std::get<0>(nnets.back())->obj, mt);
    serial.Import(basePtr, header.gnaMemSize, inputStream);


    get_ptr_inputs_global("input").push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + header.input.descriptor_offset));
    ptr_outputs_global.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + header.output.descriptor_offset));

    auto getOrientation = [](intel_nnet_layer_t & layer) {
        return layer.nLayerKind == INTEL_CONVOLUTIONAL ?
           kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;
    };

    orientation_in["input"] = getOrientation(std::get<0>(nnets.back())->obj.pLayers[0]);
    orientation_out = getOrientation(std::get<0>(nnets.back())->obj.pLayers[std::get<0>(nnets.back())->obj.nLayers-1]);

    num_bytes_per_output = header.output.element_size;


    outputDims = SizeVector({header.output.elements_count / header.nGroup, header.nGroup});
    auto inputDims = SizeVector({header.input.elements_count / header.nGroup, header.nGroup});

    inputsDataMap["input"] = std::make_shared<InputInfo>();
    inputsDataMap["input"]->setInputData(make_shared<Data>("input",
                                                           inputDims,
                                                           Precision::FP32,
                                                           Layout::NC));
    outputsDataMap["output"] = make_shared<Data>("output",
                                                 outputDims,
                                                 Precision::FP32,
                                                 Layout::NC);

    output_scale_factor = header.output.scaleFactor;
    input_scale_factor["input"] = header.input.scaleFactor;

    num_rotate_rows = header.nRotateRows;
    num_rotate_columns = header.nRotateColumns;

    for (auto && memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr);
        memoryLayer.gna_ptr = memory.first;
        memoryLayer.reserved_size = memory.second;

        memory_connection.emplace_back(make_pair(std::string("noname"), memoryLayer));
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn.WriteGraphWizModel("graph.dot");
    // ExportGnaNetworkAndrzej("layers/loaded_from_aot_file", &nnet->obj);
#endif

    return nullptr;
}

void GNAPlugin::Export(const std::string &fileName) {
    if (ptr_inputs_global_id.empty() || ptr_outputs_global.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

    if (ptr_inputs_global_id.size() != 1) {
        THROW_GNA_EXCEPTION << " exporting network with multiple inputs not supported";
    }

    std::fstream outStream(fileName, ios_base::out | ios_base::binary);

    // TODO: nnet group parameter looks only used in application - so can we move this line into load network.
    auto inputDims = inputsDataMap.begin()->second->getDims();
    if (inputDims.size() == 2) {
        std::get<0>(nnets.front())->obj.nGroup = inputDims[1];
    }

    auto serial = GNAModelSerial(&std::get<0>(nnets.front())->obj,
                   {get_input_scale_factor(),
                    ptr_inputs_global_storage.front()[0],
                    2,
                    static_cast<uint32_t>(InferenceEngine::details::product(inputsDataMap.begin()->second->getDims()))},
                   {output_scale_factor,
                    ptr_outputs_global[0],
                    num_bytes_per_output,
                    static_cast<uint32_t>(InferenceEngine::details::product(outputsDataMap.begin()->second->getDims()))})
        .SetInputRotation(dnn.num_rotate_rows, dnn.num_rotate_columns);

    for (auto && memoryConnection : memory_connection) {
        serial.AddState(memoryConnection.second.gna_ptr, memoryConnection.second.reserved_size);
    }

    serial.Export(gnamem->getBasePtr(), gnamem->getTotalBytes(), outStream);
}

void GNAPlugin::GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) {
    if (performance_counting) {
        gnadevice->getGnaPerfCounters(perfMap);
    }
}

void GNAPlugin::AddExtension(InferenceEngine::IExtensionPtr extension) {}

void GNAPlugin::SetConfig(const std::map<std::string, std::string> &config) {
    std::vector<std::string> supportedConfigOptions = {
        GNA_CONFIG_KEY(SCALE_FACTOR),
        GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE),
        GNA_CONFIG_KEY(DEVICE_MODE),
        GNA_CONFIG_KEY(COMPACT_MODE),
        CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
        GNA_CONFIG_KEY(PRECISION),
        GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN),
        CONFIG_KEY(PERF_COUNT),
        GNA_CONFIG_KEY(LIB_N_THREADS),
        CONFIG_KEY(SINGLE_THREAD)
    };

    for (auto& item : config) {
        auto keys = std::find_if(supportedConfigOptions.begin(), supportedConfigOptions.end(), [&item](std::string supportedConfigOption) {
            return item.first.find(supportedConfigOption) != std::string::npos;
        });
        if (keys == supportedConfigOptions.end()) {
            THROW_GNA_EXCEPTION << as_status << NOT_FOUND << "Incorrect GNA Plugin config. Key " << item.first << " not supported";
        }
    }

    // holds actual value of a found key
    std::string key;
    std::string value;
    auto if_set = [&](std::string keyInput, const std::function<void()> & handler) {
        auto keyInMap = config.find(keyInput);
        if (keyInMap != config.end()) {
            value = keyInMap->second;
            handler();
        }
    };

    auto if_start = [&](std::string keyInput, const std::function<void()> & handler) {
        for (auto && c : config) {
            if (c.first.find(keyInput) == 0) {
                if (c.first.size() > keyInput.size() + 1) {
                    key = c.first.substr(keyInput.size() + 1);
                    value = c.second;
                    handler();
                }
            }
        }
    };

    auto fp32eq = [](float p1, float p2) -> bool {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    };

    auto & log = gnalog();

    if_start(GNA_CONFIG_KEY(SCALE_FACTOR), [&, this] {
        // only identical scale factors supported so far
        auto ref = input_scale_factor.size() ? input_scale_factor.begin()->second : 1.0;
        input_scale_factor[key] = std::stod(value);
        if (ref != 1.0 && !fp32eq(input_scale_factor[key], ref)) {
            std::string message = "only identical input scale factors supported, but provided: "
                    + std::to_string(ref) + " and " + std::to_string(input_scale_factor[key]);
            log << "only identical input scale factors supported, but provided: " << ref <<" and " << input_scale_factor[key];
            THROW_GNA_EXCEPTION << "only identical input scale factors supported, but provided: " << ref <<" and " << input_scale_factor[key];
        }
    });

    if (input_scale_factor.empty()) {
        if_set(GNA_CONFIG_KEY(SCALE_FACTOR), [&] {
            input_scale_factor["placeHolder"] = std::stod(value);
        });
    }

    if_set(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), [&] {
        dumpXNNPath = value;
    });

    if_set(GNA_CONFIG_KEY(DEVICE_MODE), [&] {
        static caseless_unordered_map <std::string, uint32_t> supported_values = {
                {GNAConfigParams::GNA_AUTO, GNA_AUTO},
                {GNAConfigParams::GNA_HW, GNA_HARDWARE},
                {GNAConfigParams::GNA_SW, GNA_SOFTWARE},
                {GNAConfigParams::GNA_SW_EXACT, GNA_SOFTWARE & GNA_HARDWARE}
        };
        auto procType = supported_values.find(value);
        if (procType == supported_values.end()) {
            log << "GNA device mode unsupported: " << value;
            THROW_GNA_EXCEPTION << "GNA device mode unsupported: " << value;
        }
        gna_proc_type = static_cast<intel_gna_proc_t>(procType->second);
    });

    if_set(GNA_CONFIG_KEY(COMPACT_MODE), [&] {
        if (value == PluginConfigParams::YES) {
            compact_mode = true;
        } else if (value == PluginConfigParams::NO) {
            compact_mode = false;
        } else {
            log << "GNA compact mode should be YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "GNA compact mode should be YES/NO, but not" << value;
        }
    });

    if_set(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), [&] {
        if (value == PluginConfigParams::YES) {
            exclusive_async_requests  = true;
        } else if (value == PluginConfigParams::NO) {
            exclusive_async_requests  = false;
        } else {
            log << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
        }
    });

    if_set(GNA_CONFIG_KEY(PRECISION), [&] {
        auto precision = Precision::FromStr(value);
        if (precision != Precision::I8 && precision != Precision::I16) {
            log << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: " << value;
            THROW_GNA_EXCEPTION << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: " << value;
        }
        gnaPrecision = precision;
    });

    if_set(GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN), [&] {
        if (value == PluginConfigParams::YES) {
            uniformPwlDesign = true;
        } else if (value == PluginConfigParams::NO) {
            uniformPwlDesign = false;
        } else {
            log << "GNA pwl uniform algorithm parameter "
                << "should be equal to YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "GNA pwl uniform algorithm parameter "
                                << "should be equal to YES/NO, but not" << value;
        }
    });

    if_set(CONFIG_KEY(PERF_COUNT), [&] {
        if (value == PluginConfigParams::YES) {
            performance_counting = true;
        } else if (value == PluginConfigParams::NO) {
            performance_counting = false;
        } else {
            log << "GNA performance counter enabling parameter "
                << "should be equal to YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "GNA performance counter enabling parameter "
                                << "should be equal to YES/NO, but not" << value;
        }
    });

    if_set(GNA_CONFIG_KEY(LIB_N_THREADS), [&] {
        uint64_t lib_threads = std::stoul(value, NULL, 10);
        if (lib_threads == 0 || lib_threads > std::numeric_limits<uint8_t>::max()/2-1) {
            log << "Unsupported accelerator lib number of threads: " << value << ", should be greateer than 0 and less than 127";
            THROW_GNA_EXCEPTION << "Unsupported accelerator lib number of threads: " << value
                                << ", should be greateer than 0 and less than 127";
        }
        gna_lib_async_threads_num = lib_threads;
    });

    if_set(CONFIG_KEY(SINGLE_THREAD), [&] {
        if (value == PluginConfigParams::YES) {
            gna_openmp_multithreading  = false;
        } else if (value == PluginConfigParams::NO) {
            gna_openmp_multithreading  = true;
        } else {
            log << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
        }
    });
}

/**
 * @depricated Use the version with config parameter
 */
void GNAPlugin::QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                             InferenceEngine::QueryNetworkResult& res) const {
    QueryNetwork(network, {}, res);
}

void GNAPlugin::QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                             const std::map<std::string, std::string>& config,
                             InferenceEngine::QueryNetworkResult& res) const {
    std::unordered_set<CNNLayer *> allLayers;
    InferenceEngine::InputsDataMap inputs;

    network.getInputsInfo(inputs);
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);

    if (inputs.empty()) {
        THROW_GNA_EXCEPTION << "Network is empty (GNA)\n";
    }

    auto const & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty()) {
        THROW_GNA_EXCEPTION << "Network consists of input layer only (GNA)\n";
    }

    InferenceEngine::details::UnorderedDFS(allLayers,
                                           secondLayers.begin()->second,
                                           [&](CNNLayerPtr const layer) {
                                                if (GNAPluginNS::GNAPlugin::LayerTypeFromStr(layer->type) != NO_TYPE) {
                                                    res.supportedLayers.insert(layer->name);
                                                }
                                            }, false);
    }

intel_dnn_component_t * GNAPlugin::find_first_unused_input(InferenceEngine::CNNLayerPtr current) {
    if (current->insData.empty()) return nullptr;

    auto prev_layer = current->insData.front().lock()->creatorLayer.lock();

    return findDnnLayer(prev_layer);
}
void GNAPlugin::connectOutput(InferenceEngine::CNNLayerPtr layer, void *ptr, void *ptr_inputs, size_t num_data_bytes_out) {
    gnalog() << "Connecting output " << layer->name << " ...\n";
    // in case of Memory Layer it's input allocated in meminput layer
    if (layer->outData.size() == 1) {
        for (auto &&outLayer : layer->outData.front()->getInputTo()) {
            auto& nextLayer = outLayer.second;
            auto nextMemoryLayerIt =
                std::find_if(begin(memory_connection), end(memory_connection),
                                                        [&](MemoryConnection::value_type &comp) {
                                                            return comp.second.getOutput()->name
                                                                                == nextLayer->name;
                                                        });
            if (nextMemoryLayerIt != memory_connection.end()) {
                auto &nextMemoryLayer = nextMemoryLayerIt->second;
                // memory layer not yet initialized
                if (nextMemoryLayer.reserved_size == 0) {
                    gnamem->reserve_ptr(&nextMemoryLayer.gna_ptr, ALIGN64(num_data_bytes_out));
                    gnamem->bind_ptr(ptr, &nextMemoryLayer.gna_ptr, 0);

                    nextMemoryLayer.reserved_offset = 0;
                    nextMemoryLayer.reserved_size = ALIGN64(num_data_bytes_out);
                } else {
                    IE_ASSERT(nextMemoryLayer.reserved_size == ALIGN64(num_data_bytes_out));
                    // same offsets
                    gnamem->bind_ptr(ptr, &nextMemoryLayer.gna_ptr, 0);
                }
                return;
            }
        }

        // if one of next layers is concat...
        for (auto &&outLayer : layer->outData.front()->getInputTo()) {
            auto nextLayer = outLayer.second;
            if ( LayerInfo(nextLayer).isConcat() ) {
                auto& name = layer->name;
                // we look for this concat layer pointer in extra concat map
                auto concatLayerInfo = concat_connection.find(
                                nextLayer->name);

                if (concatLayerInfo != concat_connection.end()) {
                    auto &concatLayerInfoItem = concatLayerInfo->second;

                    // find this input in vector sum all outputs in primitive
                    auto it = std::find_if(concatLayerInfoItem.concatInputLayers.begin(),
                                            concatLayerInfoItem.concatInputLayers.end(),
                                            [&name](GNAPlugin::GNAConcatLayer::ConcatConnectedLayerInfo &item) {
                                                return item.name == name;
                                            });
                    if (it != concatLayerInfoItem.concatInputLayers.end()) {
                        // reserve full size for concat
                        if (!concatLayerInfoItem.output_allocation_flag) {
                            // check if this concat is being included by other one
                            // by going thru each concat and checking inputs
                            auto included =
                                    std::find_if(concat_connection.begin(),
                                                 concat_connection.end(),
                                                 [&concatLayerInfo]
                                                         (const std::pair<std::string, GNAPlugin::GNAConcatLayer> &concatItem) -> bool {
                                                     auto it = std::find_if(concatItem.second.concatInputLayers.begin(),
                                                                            concatItem.second.concatInputLayers.end(),
                                                                            [&concatLayerInfo]
                                                                                    (const GNAPlugin::GNAConcatLayer::ConcatConnectedLayerInfo &item) -> bool {
                                                                                return item.name == concatLayerInfo->first;
                                                                            });
                                                     return it != concatItem.second.concatInputLayers.end();
                                                 });
                            if (included == concat_connection.end()) {
                                gnamem->reserve_ptr(&concatLayerInfoItem.gna_ptr, ALIGN64(concatLayerInfoItem.reserved_size));

                                for (auto &&inputLayer : concatLayerInfoItem.concatInputLayers) {
                                    if (InferenceEngine::details::CaselessEq<std::string>()
                                            (inputLayer.name, "input")) {
                                        bytes_alllocated_for_input[inputLayer.name] = ALIGN64(concatLayerInfoItem.reserved_size) - inputLayer.offset;
                                    }
                                }
                            }
                            concatLayerInfo->second.output_allocation_flag = true;
                        }
                        gnamem->bind_ptr(ptr, &concatLayerInfoItem.gna_ptr, it->offset);
                    }
                } else {
                    // error
                }
                return;
            }
        }
    }

    intel_dnn_component_t * unused_input = nullptr;
    if (compact_mode) {
        unused_input = find_first_unused_input(layer);
        if (unused_input != nullptr) {
            gnamem->bind_ptr(ptr, &unused_input->ptr_inputs, 0, ALIGN64(num_data_bytes_out));
        }
    }
    // cannot reuse suitable input
    if (unused_input == nullptr) {
        gnamem->reserve_ptr(ptr, ALIGN64(num_data_bytes_out));
    }
}

intel_dnn_component_t * GNAPlugin::findDnnLayer(CNNLayerPtr __layer) {
    auto component = std::find_if(begin(dnnComponentsForLayer),
                        end(dnnComponentsForLayer),
                        [&](DnnComponentsForLayer::value_type &comp) {
                            return comp.first == __layer->name;
                        });
    // check for generic prev layer
    if (component != dnnComponentsForLayer.end()) {
        return &component->second;
    }

    return nullptr;
}

std::vector<void *>& GNAPlugin::get_ptr_inputs_global(std::string name) {
    if (!ptr_inputs_global_id.count(name)) {
        ptr_inputs_global_storage.push_front({});
        ptr_inputs_global_id[name] = ptr_inputs_global_storage.begin();
    }
    return *ptr_inputs_global_id[name];
}

GNAPlugin::ConnectionDetails GNAPlugin::connectInput(CNNLayerPtr layer, void *ptr, size_t num_data_bytes_in, int32_t offset, int idx) {
    // selecting particular input layers
    auto prevLayer = CNNNetPrevLayer(layer, idx);

    gnalog() << "Connecting input " << layer->name << " to " << prevLayer->name << " ...\n";

    // real input not a memory input
    if (LayerInfo(prevLayer).isInput()) {
        if (0 == bytes_alllocated_for_input[prevLayer->name]) {
            gnamem->push_value(&get_ptr_inputs_global(prevLayer->name).front(), static_cast<uint8_t>(0), num_data_bytes_in, 64);
            bytes_alllocated_for_input[prevLayer->name] = num_data_bytes_in;
        }
        if (ALIGN(num_data_bytes_in, 64) > ALIGN(bytes_alllocated_for_input[prevLayer->name], 64)) {
            THROW_GNA_EXCEPTION
                << "Layer: " << layer->name
                << " Cannot bind pointer to already allocated input(" << prevLayer->name
                << "), due to size_allocated=" << bytes_alllocated_for_input[prevLayer->name]
                << ", and size_requested=" << num_data_bytes_in;
        }

        if (offset >= 0) {
            gnamem->bind_ptr(ptr, &get_ptr_inputs_global(prevLayer->name).front(), offset);
        } else {
            gnamem->bind_ptr(&get_ptr_inputs_global(prevLayer->name).front(), ptr, -offset);
        }

        return prevLayer;
    }

    LayerInfo layerInfoObj(prevLayer);
    LayerInfo thisLayerInfoObj(layer);
    // connecting to split/slice splitiing layers
    if (layerInfoObj.isSplit() || layerInfoObj.isSlice()) {
        auto& splittingLayer = prevLayer;
        auto& splitName = splittingLayer->name;
        auto& name = layer->name;

        // we look for this concat layer pointer in extra concat map
        auto splitLayerInfo = split_connection.find(splitName);

        if (splitLayerInfo != split_connection.end()) {
            auto &splitLayerInfoItem = splitLayerInfo->second;
            // find this input in vector sum all outputs in primitive
            auto it = std::find_if(splitLayerInfoItem.splitOutputLayers.begin(),
                                    splitLayerInfoItem.splitOutputLayers.end(),
                                            [&name](GNAPlugin::GNASplitLayer::SplitConnectedLayerInfo &item) {
                                                return item.name == name;
                                            });

            if (it != splitLayerInfoItem.splitOutputLayers.end()) {
                gnalog()  << "Connecting split/slice input \n";
                auto res = connectInput(splittingLayer, ptr,
                                            splitLayerInfoItem.reserved_size, it->offset, 0);
                gnalog()  << "Connected \n";
                return res;
            }
        }
        THROW_GNA_EXCEPTION << "Split/Slice layer: " << splitName
                                 << " is not included in extra map. Something wrong happened";
    } else if (layerInfoObj.isConcat()) {
        auto concatLayerInfo = concat_connection.find(
                                                    prevLayer->name);
        if (concatLayerInfo != concat_connection.end()) {
            auto & concatLayerInfoItem = concatLayerInfo->second;
            // dnnLayer that is input for concat layer
            gnamem->bind_ptr(ptr, &concatLayerInfoItem.gna_ptr, offset);
            // return layer over concat
            return CNNNetPrevLayer(prevLayer);
        }
    } else if (layerInfoObj.isCrop()) {
        auto cropLayerInfo = crop_connection.find(
                                                    prevLayer->name);
        if (cropLayerInfo != crop_connection.end()) {
            auto & cropLayerInfoItem = cropLayerInfo->second;
            gnamem->bind_ptr(ptr, &cropLayerInfoItem.gna_ptr, offset);
            return CNNNetPrevLayer(prevLayer);
        }
    }
    auto prevDnnLayer = findDnnLayer(prevLayer);

    // check for generic prev layer
    if (prevDnnLayer != nullptr) {
        gnamem->bind_ptr(ptr, &prevDnnLayer->ptr_outputs, offset);
        return prevLayer;
    }

    auto prevMemoryLayer =
        std::find_if(begin(memory_connection), end(memory_connection), [&](MemoryConnection::value_type &comp) {
            return comp.second.getInput()->name == prevLayer->name;
        });
    if (prevMemoryLayer != memory_connection.end()) {
        // dnnLayer that is input for memory output layer
        auto& memoryLayer = prevMemoryLayer->second;
        if (memoryLayer.reserved_size == 0) {
            gnamem->reserve_ptr(&memoryLayer.gna_ptr, ALIGN64(num_data_bytes_in));
            gnamem->bind_ptr(ptr, &memoryLayer.gna_ptr, offset);

            memoryLayer.reserved_offset = offset;
            memoryLayer.reserved_size = ALIGN64(num_data_bytes_in);
        } else {
            IE_ASSERT(memoryLayer.reserved_size == ALIGN64(num_data_bytes_in));
            // same offsets
            gnamem->bind_ptr(ptr, &memoryLayer.gna_ptr, memoryLayer.reserved_offset);
        }

        return prevLayer;
    }

    // several layers are to be skipped right now
    if (LayerInfo(prevLayer).isReshape()) {
        gnalog()  << "Skipping reshape layer: " << prevLayer->name << "\n";
        return connectInput(prevLayer, ptr, num_data_bytes_in, offset, 0);
    }

    if (LayerInfo(prevLayer).isPermute()) {
        gnalog()  << "Skipping permute layer: " << prevLayer->name << "\n";
        return {connectInput(prevLayer, ptr, num_data_bytes_in, offset, 0).input, true, prevLayer};
    }


    THROW_GNA_EXCEPTION << "Cannot connect input for: " << layer->name;
}

