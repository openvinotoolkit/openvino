// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX

#include <vector>
#include <cstring>
#include <list>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
#include <limits>

#include <ie_layers.h>
#include <gna-api-types-xnn.h>
#include <ie_algorithm.hpp>
#include <debug.h>

#include "gna_graph_compiler.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin_log.hpp"
#include "layers/gna_layer_info.hpp"
#include "ie_memcpy.h"
#include "details/caseless.hpp"
#include "gna-api.h"
#include "backend/am_intel_dnn.hpp"
#include "runtime/pwl.h"
#include "gna_graph_tools.hpp"
#include "frontend/model_quantizer.hpp"
#include "layers/layers_builder.hpp"
#include "layers/gna_concat_layer.hpp"
#include "layers/gna_crop_layer.hpp"
#include "round_float_define.hpp"
#include "gna_plugin_policy.hpp"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;

#define CREATE(name) [](GNAGraphCompiler *p, CNNLayerPtr l) {p->name(l);}


void GNAGraphCompiler::setGNAMemoryPtr(std::shared_ptr<GNAPluginNS::gna_memory_type> gnaMemPtr) {
    this->gnamem = std::move(gnaMemPtr);
}

void GNAGraphCompiler::setDNNPtr(std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnnPtr) {
    this->dnn = std::move(dnnPtr);
}

void GNAGraphCompiler::setInputDescPtr(std::shared_ptr<GNAPluginNS::InputDesc> inputDescPtr) {
    this->inputDesc = std::move(inputDescPtr);
}

void GNAGraphCompiler::setGNAFlagsPtr(std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlagsPtr) {
    this->gnaFlags = std::move(gnaFlagsPtr);
}

void GNAGraphCompiler::setPolicy(GNAPluginNS::Policy policyToSet) {
    this->policy = policyToSet;
}

intel_dnn_component_t * GNAGraphCompiler::find_first_unused_input(InferenceEngine::CNNLayerPtr current) {
    if (current->insData.empty())
        return nullptr;
    auto inData = current->insData.front().lock();
    if (inData == nullptr)
        return nullptr;

    auto prev_layer = getCreatorLayer(inData).lock();

    return dnnComponents.findComponent(prev_layer);
}

void GNAGraphCompiler::fillMemoryConnections(std::unordered_map<std::string,
                                      std::vector<InferenceEngine::CNNLayerPtr>>& memoryPairs) {
    for (auto &memory : memoryPairs) {
        auto inputLayer = memory.second[1];
        auto outputLayer = memory.second[0];

        IE_ASSERT(1 == outputLayer->insData.size());

        // creating connection for layers output as form of extramap
        memory_connection.emplace_back(memory.first, GNAMemoryLayer(inputLayer, outputLayer, gnaFlags->sw_fp32 ? 4 : 2));
    }
}

void GNAGraphCompiler::fillConcatConnections(InferenceEngine::CNNLayerPtr layer) {
    // creating connection for each layer outputs as form of extramap
    GNAPluginNS::GNAConcatLayer layerInfoItem(layer);
    size_t concat_size = 0;
    std::string& id = layer->name;

    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto ptrConcatLayerInput = CNNNetPrevLayerSkipCertain(layer, i, [](CNNLayerPtr lp) {
            LayerInfo info(lp);
            return info.isNonFunctional();
        });
        auto dataInput = layer->insData[i].lock();
        if (!dataInput) {
            THROW_GNA_EXCEPTION << "Input layer pointer for concat is unexpectedly absent";
        }

        if (!ptrConcatLayerInput) {
            THROW_GNA_EXCEPTION << "Input layer for concat is unexpectedly absent";
        }

        size_t layer_size =
                InferenceEngine::details::product(begin(dataInput->getDims()),
                                                  end(dataInput->getDims())) * dataInput->getPrecision().size();

        layerInfoItem.concatInputLayers.emplace_back(GNAConcatLayer::ConcatConnectedLayerInfo{ptrConcatLayerInput->name, concat_size, layer_size});

        concat_size += layer_size;
    }
    layerInfoItem.reserved_size = concat_size;
    concat_connection.emplace(id, layerInfoItem);
}

void GNAGraphCompiler::fillSplitConnections(InferenceEngine::CNNLayerPtr layer) {
    // creating connection for each layer inputs as form of extramap
    GNAPluginNS::GNASplitLayer layerInfoItem(layer);
    size_t split_size = 0;
    std::string& id = layer->name;
    IE_ASSERT(!layer->insData.empty());

    auto dataInput = layer->insData.begin()->lock();
    if (!dataInput) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Input layer pointer is unexpectedly absent";
    }
    auto ptrSplitLayerInput = getCreatorLayer(dataInput).lock();
    if (!ptrSplitLayerInput) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Input layer for is unexpectedly absent";
    }

    for (size_t i = 0; i < layer->outData.size(); ++i) {
        size_t padding = 0;
        size_t output_layer_size = 0;


        for (int j = 0; j != getInputTo(layer->outData[i]).size(); j++) {
            auto outFunctionalLayer = CNNNetGetNextLayerSkipCertain(layer, i, j,  [](CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            });

            if (!outFunctionalLayer.first) {
                THROW_GNA_LAYER_EXCEPTION(layer) << " outData["<< i << "]" << " connected by " << j <<" connection doesnt connect to functional layer";
            }

            auto dataOutput = outFunctionalLayer.first->insData[outFunctionalLayer.second].lock();

            padding = std::max(padding, LayerInfo(outFunctionalLayer.first).paddingSize())
                                                        * dataOutput->getPrecision().size();
            output_layer_size =
                    InferenceEngine::details::product(begin(dataOutput->getDims()),
                                                     end(dataOutput->getDims())) * dataOutput->getPrecision().size();

            if (LayerInfo(outFunctionalLayer.first).isAffineFilter()) {
                size_t aligned64_offset = outFunctionalLayer.first->GetParamAsInt("offset");
                layerInfoItem.splitOutputLayers.emplace_back(
                    outFunctionalLayer.first,
                    outFunctionalLayer.second,
                    aligned64_offset * dataOutput->getPrecision().size(),
                    output_layer_size);
            } else {
                layerInfoItem.splitOutputLayers.emplace_back(
                    outFunctionalLayer.first, outFunctionalLayer.second, split_size, output_layer_size);
            }
        }

        split_size += padding + output_layer_size;
    }
    layerInfoItem.reserved_size = split_size;
    split_connection.emplace(id, layerInfoItem);
}

void GNAGraphCompiler::DiagonalPrimitive(InferenceEngine::CNNLayerPtr layer) {
    AffinePrimitive(layer, true);
}

void  GNAGraphCompiler::ConstPrimitive(InferenceEngine::CNNLayerPtr constLayer) {
    if (constLayer->blobs.find("custom") == constLayer->blobs.end()) {
        THROW_GNA_EXCEPTION << "const layer: " << constLayer->name << "doesn't have custom in blobs section";
    }
    auto const_blob = constLayer->blobs["custom"];

    const_connections[constLayer->name] = &const_connections[constLayer->name];
    void* ptr_for_const_blob = &const_connections[constLayer->name];

    connectOutput(constLayer, ptr_for_const_blob, const_blob->byteSize());
    // TODO: segment type for bind, bind initializer not used - need refactor to separate bind and allocation requests
    // dont see practical use case when bind storage type need to be different that allocation type
    gnamem->readonly().bind_initializer(ptr_for_const_blob, [const_blob](void* data, size_t size) {
        ie_memcpy(data, size, const_blob->buffer(), const_blob->byteSize());
        });
}


void GNAGraphCompiler::ConvolutionPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& convolution = dynamic_cast<ConvolutionLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t w_dim_in = FROM_IR_DIM(inputs, 1);
    uint32_t h_dim_in = FROM_IR_DIM(inputs, 2);
    uint32_t c_dim_in = FROM_IR_DIM(inputs, 3);
    uint32_t w_dim_out = FROM_IR_DIM(outputs, 1);
    uint32_t h_dim_out = FROM_IR_DIM(outputs, 2);

    if (w_dim_in == 1) {  // swap dimensions if needed to support swapped 1D case
        swap(h_dim_in, w_dim_in);
        swap(h_dim_out, w_dim_out);
        swap(convolution._kernel_x, convolution._kernel_y);
        swap(convolution._stride_x, convolution._stride_y);
    }

    uint32_t num_feature_map_rows = w_dim_in / convolution._stride_x;
    uint32_t num_feature_map_columns = c_dim_in * convolution._stride_x;

    uint32_t num_columns_in = c_dim_in;
    uint32_t num_rows_out = w_dim_out;

    // padding of convolution kernel to be multiply of 8
    uint32_t num_conv_kernel_padding = ALIGN(convolution._kernel_x * num_feature_map_columns, 8)
        - convolution._kernel_x * num_feature_map_columns;
    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    // TODO: questionable why for biases that are not in IR we inventing precision
    auto biasPrecision = convolution._biases ? convolution._biases->getTensorDesc().getPrecision() : outputs->getPrecision();

    auto& currentComponent = dnnComponents.addComponent(layer->name, "convolution");

    // have to pad input to let last kernel meets it's corresponding input
    auto num_inputs = num_feature_map_columns * num_feature_map_rows + num_conv_kernel_padding;
    auto num_input_padding = ALIGN(num_inputs, 8) - num_inputs;
    auto num_filter_rows = convolution._kernel_x / convolution._stride_x;
    dnn->InitConvolutional1DComponent(currentComponent,
        1,
        num_inputs + num_input_padding,
        1,
        num_rows_out * convolution._out_depth,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        convolution._weights->getTensorDesc().getPrecision().size(),
        biasPrecision.size(),
        convolution._out_depth,
        num_filter_rows,
        num_feature_map_columns * num_filter_rows + num_conv_kernel_padding,
        1,
        num_feature_map_rows,
        num_feature_map_columns,
        quantized == nullptr ? 1 : quantized->_weights_quant.scale,
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases);

    size_t num_data_bytes_out =
        InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
        * outputs->getPrecision().size();

    size_t num_data_bytes_in = (num_inputs + num_input_padding) * inputs->getPrecision().size();

    auto connectedInputLayer = connectInput(layer, ptr_inputs, num_data_bytes_in).input;

    // TODO: convolution might be not the first layer in sorted order but connected via split for example - dont know how kaldi will handle that
    if (LayerInfo(connectedInputLayer).isInput()) {
        //  Kaldi features are opposite orientation
        dnn->num_rotate_rows = num_feature_map_columns;
        dnn->num_rotate_columns = num_feature_map_rows;
    }

    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    // rotate
    auto TransposeMatrix = [](uint8_t* ptr_matrix, size_t element_size, uint32_t num_rows, uint32_t num_cols) {
        std::vector<uint8_t> temp_buffer(num_rows * num_cols * element_size);
        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_cols; j++) {
                ie_memcpy(&temp_buffer.front() + (j * num_rows + i) * element_size,
                    temp_buffer.size() - (i * num_cols + j) * element_size,
                    ptr_matrix + (i * num_cols + j) * element_size,
                    element_size);
            }
        }
        return temp_buffer;
    };

    std::vector<uint8_t > transposedWeights;
    for (uint32_t k = 0; k < convolution._out_depth; k++) {
        uint8_t* ptr_filt_current
            = convolution._weights->cbuffer().as<uint8_t*>() + k * num_columns_in * convolution._kernel[X_AXIS] * convolution.precision.size();
        auto transposedPart = TransposeMatrix(ptr_filt_current, convolution.precision.size(), num_columns_in, convolution._kernel[X_AXIS]);
        transposedWeights.insert(transposedWeights.end(), transposedPart.begin(), transposedPart.end());
    }

    if (num_conv_kernel_padding == 0) {
        gnamem->readonly().push_local_ptr(ptr_weights, transposedWeights.data(), convolution._weights->byteSize(), 64);
    } else {
        auto elementsIn = convolution._kernel_x * num_feature_map_columns + num_conv_kernel_padding;
        auto paddedWeights = elementsIn * convolution._out_depth;
        auto paddedWeightsSize = paddedWeights * convolution.precision.size();
        auto elements_in_row = convolution._kernel_x * num_feature_map_columns;
        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void* data, size_t size) {
            size_t offset = 0;
            for (int i = 0; i < convolution._out_depth && size >= offset; i++) {
                ie_memcpy(reinterpret_cast<uint8_t*>(data) + offset, size - offset,
                    transposedWeights.data() + elements_in_row * i * convolution.precision.size(),
                    elements_in_row* convolution.precision.size());

                offset += elementsIn * convolution.precision.size();
            }
            }, 64);
    }

    if (convolution._biases) {
        gnamem->readonly().push_ptr(ptr_biases,
            convolution._biases->cbuffer().as<const void*>(),
            convolution._biases->byteSize(),
            64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAGraphCompiler::PowerPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& power = dynamic_cast<PowerLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    if (power.power < 0.0f && power.power > 2.8f) {
        THROW_IE_EXCEPTION << "[GNA plugin] unsupported power factor, expected be in <0, 2.8> range but was " << power.power;
    }

    auto inputs = layer->insData[0].lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;
    
    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
        * outputs->getPrecision().size();

    size_t num_data_bytes_in = InferenceEngine::details::product(begin(inputs->getDims()), end(inputs->getDims()))
        * inputs->getPrecision().size();

    //use PWL to calculate power
    std::vector<intel_pwl_segment_t> ptr_pwl_segments;

    float output_pwl_scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
    float input_pwl_scale_factor = quantized != nullptr ? quantized->_src_quant.scale : 1.0f;

    auto orientation = kDnnInterleavedOrientation;

    auto activation_type = DnnActivation::fromType(kActPow);
    activation_type.args.pow.exponent = power.power;
    activation_type.args.pow.scale = power.scale;
    activation_type.args.pow.offset = power.offset;

    auto& pwlComponent = dnnComponents.addComponent(layer->name, "power");

    intel_pwl_segment_t* ptr_pwl_segments_target = nullptr;

    if (!gnaFlags->sw_fp32) {
        if (gnaFlags->uniformPwlDesign) {
            uint32_t num_segments = POW_NUM_SEGMENTS;
            if (activation_type.args.pow.exponent == 0.0f || activation_type.args.pow.exponent == 1.0f) {
                num_segments = 3;
            }
            ptr_pwl_segments.resize(num_segments);

            PwlDesign16(activation_type,
                &*ptr_pwl_segments.begin(),
                static_cast<uint32_t>(ptr_pwl_segments.size()),
                input_pwl_scale_factor,
                output_pwl_scale_factor);
        }
        else {
            PwlDesignOpt16(activation_type,
                ptr_pwl_segments,
                input_pwl_scale_factor,
                output_pwl_scale_factor);
        }
    }

    ptr_pwl_segments_target = reinterpret_cast<intel_pwl_segment_t*>(&ptr_pwl_segments_target);

    void* ptr_pwl_inputs = nullptr;
    void* ptr_pwl_outputs = nullptr;
    dnn->InitPiecewiseLinearComponent(pwlComponent,
        activation_type,
        orientation,
        num_rows_in + num_padding,
        num_columns_in,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        ptr_pwl_segments.size(),
        output_pwl_scale_factor,
        output_pwl_scale_factor,
        ptr_pwl_inputs,
        ptr_pwl_outputs,
        ptr_pwl_segments_target);

    connectOutput(layer, ptr_pwl_outputs, num_data_bytes_out);
    connectInput(layer, ptr_pwl_inputs, num_data_bytes_in, 0, 0);

    if (ptr_pwl_segments_target != nullptr) {
        gnamem->readonly().push_local_ptr(ptr_pwl_segments_target,
            &ptr_pwl_segments.front(),
            ptr_pwl_segments.size() * sizeof(intel_pwl_segment_t),
            64);
    }
}

void GNAGraphCompiler::PoolingPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& pooling = dynamic_cast<PoolingLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t w_dim_in = FROM_IR_DIM(inputs, 1);
    uint32_t h_dim_in = FROM_IR_DIM(inputs, 2);
    uint32_t c_dim_in = FROM_IR_DIM(inputs, 3);
    uint32_t w_dim_out = FROM_IR_DIM(outputs, 1);
    uint32_t h_dim_out = FROM_IR_DIM(outputs, 2);
    uint32_t c_dim_out = FROM_IR_DIM(outputs, 3);

    if (w_dim_in == 1) {  // swap dimensions if needed to support swapped 1D case
        swap(h_dim_in, w_dim_in);
        swap(h_dim_out, w_dim_out);
        swap(pooling._kernel[X_AXIS], pooling._kernel[Y_AXIS]);
    }

    uint32_t num_rows_in = w_dim_in;
    uint32_t num_columns_in = c_dim_in;
    uint32_t num_rows_out = w_dim_out;
    uint32_t num_columns_out = c_dim_out;
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;

    auto& currentComponent = dnnComponents.addComponent(layer->name, "pooling");

    switch (pooling._type) {
    case PoolingLayer::MAX: break;
        // we are loosing precision here
    case PoolingLayer::AVG:
    default:
        // TODO: convert to SUMM pooling
        THROW_GNA_EXCEPTION << "Layer :" << layer->name << " not supported";
    }

    dnn->InitMaxpoolComponent(currentComponent,
        1,
        num_columns_in * num_rows_in,
        1,
        num_columns_out * num_rows_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        pooling._kernel[X_AXIS],
        pooling._kernel[X_AXIS],
        num_columns_in,
        false,
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        ptr_inputs,
        ptr_outputs);

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
        * outputs->getPrecision().size();

    size_t num_data_bytes_in = num_columns_in * (num_rows_in + num_padding) * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);
}

void GNAGraphCompiler::CopyPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_columns_out = FROM_IR_DIM(outputs, 2);
    uint32_t num_padding_out = ALIGN(num_rows_out, 8) - num_rows_out;
    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    auto orientation = kDnnInterleavedOrientation;

    auto& currentComponent = dnnComponents.addComponent(layer->name, "copy");

    dnn->InitCopyComponent(currentComponent,
        orientation,
        ALIGN(num_rows_in, 8),
        num_columns_in,
        ALIGN(num_rows_out, 8),
        num_columns_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        num_rows_out + num_padding_out,
        num_columns_out,
        ptr_inputs,
        ptr_outputs);

    size_t num_data_bytes_out = ALIGN(InferenceEngine::details::product(
        begin(outputs->getDims()), end(outputs->getDims())), 8)
        * outputs->getPrecision().size();
    size_t num_data_bytes_in = num_columns_in * ALIGN(num_rows_in, 8) * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);
}

void GNAGraphCompiler::ConcatPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto concatLayer = dynamic_cast<InferenceEngine::ConcatLayer *> (layer.get());

    if (concatLayer == nullptr) {
        return;
    }
    if (concatLayer->insData.size() < 2) {
        THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers.";
    }

    for (std::size_t layerIndex = 0; layerIndex < concatLayer->insData.size(); layerIndex++) {
        auto input = concatLayer->insData[layerIndex].lock();
        if (!input) {
            THROW_GNA_EXCEPTION << "Input layer " << layerIndex << " for concat is unexpectedly absent";
        }
    }

    std::size_t layerPrecisionSize = concatLayer->insData[0].lock()->getPrecision().size();
    for (std::size_t layerIndex = 0; layerIndex < concatLayer->insData.size(); layerIndex++) {
        auto currentSize = concatLayer->insData[layerIndex].lock()->getPrecision().size();
        if (layerPrecisionSize != currentSize) {
            THROW_GNA_EXCEPTION << "Different precision for Concat Layer '" << concatLayer->name << "' input layers." <<
                                "input 0 precision is '" << concatLayer->insData[0].lock()->getPrecision().name() << "' but input " << layerIndex <<
                                " precision is '" << concatLayer->insData[layerIndex].lock()->getPrecision().name() << "'";
        }
    }

    auto& concatLayerInfo = concat_connection.find(concatLayer->name)->second;
    for (auto &&outLayer : getInputTo(concatLayer->outData.front())) {
        if ( LayerInfo(outLayer.second).isConcat() ) {
            connectOutput(layer, &concatLayerInfo.gna_ptr, concatLayerInfo.reserved_size);
        }
    }

    size_t idx = 0;
    for (auto && inputLayer : concatLayerInfo.concatInputLayers) {
        auto concatLayerInput = concat_connection.find(concatLayer->name)->second.getConcat();
        CNNLayerPtr concatParent;
        int it = 0;

        for (; it != concatLayerInput->insData.size(); it++) {
            concatParent = CNNNetPrevLayerSkipCertain(concatLayerInput, it, [](CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            });
            if (concatParent->name.find(inputLayer.name) != std::string::npos) {
                break;
            }
        }
        IE_ASSERT(it != concatLayerInput->insData.size());
        auto layerInfo = LayerInfo(concatParent);
        // auto layerInfo = LayerInfo(getCreatorLayer(concatLayerInput->insData[it].lock()).lock());
        if (layerInfo.isInput()) {
            if (concatLayerInfo.input_allocated) {
                // for concat input allocated only once, so lets mark this specific input layer also as allocated
                // we will bind it to offset further in connectInput
                // size need to be equal to full layer in order to pass checks
                inputDesc->bytes_allocated_for_input[((InferenceEngine::CNNLayerPtr)layerInfo)->name] = concatLayerInfo.reserved_size;
            }

            connectInput(layer, &concatLayerInfo.gna_ptr,
                         concatLayerInfo.reserved_size, -static_cast<int32_t>(inputLayer.offset), idx);

            // TODO: currently connectInput api accept only total size, for concat we need extension for allocated, and actual sizes
            inputDesc->bytes_allocated_for_input[((InferenceEngine::CNNLayerPtr) layerInfo)->name] = inputLayer.tensorSize;

            concatLayerInfo.input_allocated = true;
        } else  if (layerInfo.isMemory()) {
            connectInput(layer, &concatLayerInfo.gna_ptr, concatLayerInfo.reserved_size, -static_cast<int>(inputLayer.offset), idx);

            concatLayerInfo.input_allocated = true;
        }
        ++idx;
    }
}

void GNAGraphCompiler::CropPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto cropLayer = dynamic_cast<InferenceEngine::CropLayer*> (layer.get());

    if (cropLayer == nullptr) {
        return;
    }

    IE_ASSERT(!layer->insData.empty());
    auto inputs = layer->insData.begin()->lock();

    IE_ASSERT(!cropLayer->axis.empty());
    IE_ASSERT(cropLayer->axis.size() == cropLayer->dim.size());
    IE_ASSERT(cropLayer->axis.size() == cropLayer->offset.size());

    std::vector<int> axis, dim, offset;
    for (int n = 0; n < cropLayer->axis.size(); n++) {
        uint32_t input_dim = FROM_IR_DIM(inputs, inputs->getDims().size() - cropLayer->axis[n]);
        // Exclude crop layer components that do nothing
        if (cropLayer->offset[n] == 0 && cropLayer->dim[n] == input_dim) {
            continue;
        }
        axis.push_back(cropLayer->axis[n]);
        dim.push_back(cropLayer->dim[n]);
        offset.push_back(cropLayer->offset[n]);
    }

    if (axis.size() > 1) {
        THROW_GNA_EXCEPTION <<
            "Crop layer does not support the number of (non-trivial) cropped dimensions more than 1, provided: "
            << axis.size() << ".";
    }

    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    size_t cropOffset = offset.front() * cropLayer->precision.size();
    size_t cropOutputSize = dim.front() * cropLayer->precision.size();

    if (ALIGN64(cropOffset) == cropOffset) {
        // leave crop as it is
        GNAPluginNS::GNACropLayer cropLayerInfoItem(layer);
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
        for (auto&& outLayer : getInputTo(layer->outData.front())) {
            auto& nextLayer = outLayer.second;
            if (LayerInfo(nextLayer).isConcat()) {
                connectOutput(layer, &cropLayerInfo->second.gna_ptr, cropOutputSize);
            }
        }
    } else {
        gnalog() << "Crop " << layer->name << " is being replaced by Affine layer...\n";
        IE_ASSERT(!layer->outData.empty());
        auto outputs = *layer->outData.begin();

        // only 1D crops supported
        if (axis.size() != 1) {
            THROW_GNA_EXCEPTION << "only 1D crop layer supported: " << cropLayer->name;
        }

        // TODO: add unit tests for 4d crops blobs
        uint32_t num_rows_in = FROM_IR_DIM(inputs, inputs->getDims().size() - axis.front());
        uint32_t num_columns_in = 1;

        uint32_t num_rows_out = FROM_IR_DIM(outputs, inputs->getDims().size() - axis.front());
        uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

        void* ptr_inputs = nullptr;
        void* ptr_outputs = nullptr;
        void* ptr_weights = nullptr;
        void* ptr_biases = nullptr;

        auto& currentComponent = dnnComponents.addComponent(layer->name, "crop");

        dnn->InitAffineComponent(currentComponent,
            num_rows_in + num_padding,
            num_columns_in,
            num_rows_out,
            inputs->getPrecision().size(),
            4,
            quantized == nullptr ? inputs->getPrecision().size() : 2,
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
                begin(outputs->getDims()), end(outputs->getDims())) * 4;

        size_t num_data_bytes_in = num_columns_in *
            ALIGN(num_rows_in, 8) * inputs->getPrecision().size();

        connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
        connectOutput(layer, ptr_outputs, num_data_bytes_out);

        FillWeightOfAligningFilter(layer, ptr_weights, offset.front(), (quantized == nullptr) ? false : true);

        (quantized == nullptr) ?
            gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64) :
            gnamem->readonly().push_value<int32_t>(ptr_biases, 0, num_rows_out, 64);
    }
}

void GNAGraphCompiler::SplitPrimitive(InferenceEngine::CNNLayerPtr layer) {
    //  Nothing to do
}

void GNAGraphCompiler::SlicePrimitive(InferenceEngine::CNNLayerPtr layer) {
    //  Nothing to do
}

void GNAGraphCompiler::EltwisePrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& eltwise = dynamic_cast<EltwiseLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    // for eltwise should be one input of 4 bytes and one of 2 bytes - detecting that
    auto inputs2Bytes = layer->insData[0].lock();
    auto inputs4Bytes = layer->insData[1].lock();

    int biasesLayerIdx = 1;

    if (quantized) {
        switch (eltwise._operation) {
        case InferenceEngine::EltwiseLayer::Sum:
        case InferenceEngine::EltwiseLayer::Sub:
        {
            if (inputs4Bytes->getPrecision().size() != 4) {
                std::swap(inputs4Bytes, inputs2Bytes);
                biasesLayerIdx = 0;
            }
            GNA_LAYER_ASSERT(layer, inputs2Bytes->getPrecision().size() == 2);
            GNA_LAYER_ASSERT(layer, inputs4Bytes->getPrecision().size() == 4);
            break;
        }
        case InferenceEngine::EltwiseLayer::Prod:
        {
            // for mul both inputs should be 2 bytes precision
            GNA_LAYER_ASSERT(layer, inputs2Bytes->getPrecision().size() == 2);
            GNA_LAYER_ASSERT(layer, inputs4Bytes->getPrecision().size() == 2);
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Unsupported eltwise operation for quantization: " << eltwise._operation;
        }
    }

    auto outputs = *layer->outData.begin();

    uint32_t num_rows_in = FROM_IR_DIM(inputs4Bytes, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs4Bytes, 2);
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    auto& currentComponent = dnnComponents.addComponent(layer->name, "diagonal");
    dnn->InitAffineComponent(currentComponent,
        num_rows_in + num_padding,
        num_columns_in,
        num_rows_out + num_padding,
        inputs2Bytes->getPrecision().size(),
        outputs->getPrecision().size(),
        // TODO: only fp32 and Int16 tested
        quantized == nullptr ? inputs2Bytes->getPrecision().size() : 2,
        quantized == nullptr ? inputs4Bytes->getPrecision().size() : 4,
        quantized == nullptr ? 1 : quantized->_weights_quant.scale,
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases,
        true);

    size_t num_data_bytes_out =
        InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims())) * outputs->getPrecision().size();

    size_t num_data_bytes_in =
        num_columns_in * (num_rows_in + num_padding) * inputs2Bytes->getPrecision().size();

    connectOutput(layer, ptr_outputs, num_data_bytes_out);
    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 1 - biasesLayerIdx);

    switch (eltwise._operation) {
    case EltwiseLayer::Sub:
        if (quantized == nullptr) {
            gnamem->readonly().push_value(ptr_weights, -1.0f, num_rows_out, 64);
        } else {
            auto scaledIdentity = -quantized->_weights_quant.scale;

            auto quantizedIdentity = FLOAT_TO_INT16(std::min(scaledIdentity, static_cast<float>(INT16_MAX)));

            gnamem->readonly().push_value<int16_t>(ptr_weights, quantizedIdentity, num_rows_out, 64);
        }
        connectInput(layer, ptr_biases, num_data_bytes_in, 0, biasesLayerIdx);
        break;
    case EltwiseLayer::Sum:
        if (quantized == nullptr) {
            gnamem->readonly().push_value(ptr_weights, 1.0f, num_rows_out, 64);
        } else {
            auto scaledIdentity = quantized->_weights_quant.scale;

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

void GNAGraphCompiler::AffinePrimitive(InferenceEngine::CNNLayerPtr layer, bool isDiag) {
    auto& weightable = dynamic_cast<WeightableLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();
    auto inputPrecision = quantized ? Precision(Precision::I16) : inputs->getPrecision();

    uint32_t num_rows_in = FROM_IR_DIM(inputs, 1);
    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = isDiag ? num_rows_in : FROM_IR_DIM(outputs, 1);
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;
    uint32_t num_padding_out = isDiag ? num_padding : 0;

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    // TODO: questionable why for biases that are no in Model we inventing precision
    auto biasPrecision = weightable._biases ? weightable._biases->getTensorDesc().getPrecision() : outputs->getPrecision();

    // layer without biases might be connected to functional layer without activations
    auto prevLayer = CNNNetPrevLayer(layer);
    bool useBiasConnection = false;
    if (LayerInfo(prevLayer).has32BOutput()) {
        if (weightable._biases) {
            THROW_GNA_EXCEPTION << "Layer: "
                << layer->name << ", cannot be connected to its parent: " << prevLayer->name
                << " due to precision mismatch";
        }
        gnalog() << "Connection " << prevLayer->name << " to " << layer->name << " is using BIAS as input" << std::endl;
        useBiasConnection = true;
    }

    auto& currentComponent = dnnComponents.addComponent(layer->name, (isDiag ? "diagonal" : "affine"));

    dnn->InitAffineComponent(currentComponent,
        num_rows_in + num_padding,
        num_columns_in,
        num_rows_out + num_padding_out,
        inputPrecision.size(),
        outputs->getPrecision().size(),
        weightable._weights->getTensorDesc().getPrecision().size(),
        biasPrecision.size(),
        quantized == nullptr ? 1 : quantized->_weights_quant.scale,
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases,
        isDiag);

    size_t num_data_bytes_out =
        num_columns_in * (num_rows_out + num_padding_out) * outputs->getPrecision().size();

    size_t num_data_bytes_in = num_columns_in * (num_rows_in + num_padding) * inputs->getPrecision().size();

    auto connectionInfo = connectInput(layer, useBiasConnection ? ptr_biases : ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    auto transpose = false;
    auto transposedRows = 0;
    auto transposedCols = 0;

    if (0 && connectionInfo.needTransposeWeights) {
        // direct order is 0, 1, 2, 3, supported order is only 0,3,2,1 where dim 2 is usually equals to 1
        auto permuteOrder = connectionInfo.permute->GetParamAsInts("order");
        if (permuteOrder != vector<int>({ 0, 3, 2, 1 })) {
            THROW_IE_EXCEPTION << "[GNA plugin] Unsupported permute order: was " << layer->GetParamAsString("order") <<
                ", but only support 0, 3, 2, 1";
        }

        /**
         * TODO: weights transpose happened after quantisation might result in poor quality for in 8 - move this to passes
         */
        if (weightable._weights->getTensorDesc().getPrecision() == Precision::I8) {
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
                weightable._weights->cbuffer().as<const void*>(),
                weightable._weights->byteSize(),
                64);
        } else {
            gnamem->readonly().push_initializer(ptr_weights, weightable._weights->byteSize(), [=](void* data, size_t size) {
                for (int k = 0; k < (isDiag ? 1 : num_rows_out); k++) {
                    auto rowOffset = k * transposedRows * transposedCols * weightable.precision.size();
                    auto cbuffer = weightable._weights->cbuffer().as<const uint8_t*>() + rowOffset;
                    auto u8Data = reinterpret_cast<uint8_t*>(data) + rowOffset;
                    for (int j = 0; j < transposedCols; j++) {
                        for (int i = 0; i < transposedRows; i++) {
                            auto offsetWrite = (transposedRows * j + i) * weightable.precision.size();
                            auto offsetRead = (i * transposedCols + j) * weightable.precision.size();
                            if (size < rowOffset + offsetWrite) {
                                // zero out dest if error detected
                                memset(data, 0, size);
                                THROW_GNA_EXCEPTION << "Size error";
                            }
                            ie_memcpy(u8Data + offsetWrite, size - rowOffset - offsetWrite,
                                cbuffer + offsetRead, weightable.precision.size());
                        }
                    }
                }
                }, 64);
        }
    } else {
        if (transpose) {
            THROW_GNA_EXCEPTION << "transposed weights with non zero padding not yet supported";
        }
        auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
        auto paddedWeights = isDiag ? elementsIn : elementsIn * num_rows_out;
        auto paddedWeightsSize = paddedWeights * weightable.precision.size();

        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void* data, size_t size) {
            for (int i = 0; i < (isDiag ? 1 : num_rows_out); i++) {
                ie_memcpy(data, size,
                    weightable._weights->cbuffer().as<const uint8_t*>() + num_rows_in * i * weightable.precision.size(),
                    num_rows_in * weightable.precision.size());
                data = reinterpret_cast<uint8_t*>(data) + (num_rows_in + num_padding) * weightable.precision.size();
            }
            }, 64);
    }

    if (weightable._biases) {
        gnamem->readonly().push_ptr(ptr_biases,
            weightable._biases->cbuffer().as<const void*>(),
            weightable._biases->byteSize(),
            64);
    } else {
        // in that case input from previous layer goes into biases, so we have to initialize input pointer by zero
        if (useBiasConnection) {
            gnamem->readonly().push_value(ptr_inputs, 0.0f, num_rows_in + num_padding, 64);
        } else {
            gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out + num_padding_out, 64);
        }
    }
}

void GNAGraphCompiler::FillWeightOfAligningFilter(InferenceEngine::CNNLayerPtr layer, void* ptrWeights, size_t offset, bool isQuantized) {
    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_rows_in = InferenceEngine::details::product(++begin(inputs->getDims()), end(inputs->getDims()));
    uint32_t num_rows_out = InferenceEngine::details::product(++begin(outputs->getDims()), end(outputs->getDims()));

    if (!ptrWeights) {
        THROW_GNA_EXCEPTION << "Weights memory is not allocated!!!";
    }

    gnamem->readonly().push_initializer(ptrWeights, num_rows_out * ALIGN(num_rows_in, 8) * layer->precision.size(), [=](void* data, size_t size) {
        int out = 0;
        for (int input = offset; input < num_rows_out + offset; ++input) {
            auto mem_ptr = reinterpret_cast<uint8_t*>(data) + input * layer->precision.size() + out * ALIGN(num_rows_in, 8) * layer->precision.size();
            if (!isQuantized) {
                auto float_ptr = reinterpret_cast<float*>(mem_ptr);
                *float_ptr = 1.0f;
            } else {
                auto int_ptr = reinterpret_cast<uint16_t*>(mem_ptr);
                *int_ptr = 1;
            }
            ++out;
        }
        }, 64);
}

void GNAGraphCompiler::ConcatAlignFilterPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto filterLayer = dynamic_cast<InferenceEngine::WeightableLayer*> (layer.get());

    if (filterLayer == nullptr) {
        return;
    }

    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_rows_in = filterLayer->_weights->size() / num_rows_out;
    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;

    auto numRowsPadded = filterLayer->GetParamAsInt("num_rows_padded");
    // number of rows we handled by inserting copy layer
    uint32_t num_rows_copied = 0;
    // in case of left alignment succeed, but due to number of elements not multiple of 8 we need to insert align_filter
    // we are improving it by inserting copy layer of size that covers most of elements - remained max of 32x31 affine filter
    if (policy.ConcatAlignmentPolicy == Policy::ConcatAlignment::FAST &&  0 == numRowsPadded && ALIGN(num_rows_in, 32) > 32) {
        // can we use copy at all
        num_rows_copied = ALIGN(num_rows_in, 32) - 32;

        auto orientation = kDnnInterleavedOrientation;

        auto& copyComponent = dnnComponents.addComponent(layer->name + "_synthetic_copy", "copy");

        dnn->InitCopyComponent(copyComponent,
                               orientation,
                               num_rows_copied,
                               num_columns_in,
                               num_rows_copied,
                               num_columns_in,
                               inputs->getPrecision().size(),
                               inputs->getPrecision().size(),
                               quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                               num_rows_copied,
                               num_columns_in,
                               ptr_inputs,
                               ptr_outputs);


        size_t num_data_bytes_in = num_rows_copied * num_rows_copied * num_columns_in
            * inputs->getPrecision().size();
        // need to reserve full tensor so using original size with assumption of identity activation attached to filter lateron
        size_t num_data_bytes_out = num_rows_out * num_columns_in * inputs->getPrecision().size();

        connectInput(layer, ptr_inputs, num_data_bytes_in);
        auto isNonFunctional = [](CNNLayerPtr l) {
            return LayerInfo(l).isNonFunctional();
        };
        auto identity = CNNNetGetNextLayerSkipCertain(layer, 0, 0, isNonFunctional);
        connectOutput(identity.first, ptr_outputs, num_data_bytes_out);

        num_rows_in  -= num_rows_copied;
        num_rows_out -= num_rows_copied;
    }
    filterLayer->params["rows_copied_offset"] = std::to_string(num_rows_copied * inputs->getPrecision().size());


    auto biasPrecision = filterLayer->_biases ? filterLayer->_biases->getTensorDesc().getPrecision() : outputs->getPrecision();
    auto& currentComponent = dnnComponents.addComponent(layer->name, "affine");

    dnn->InitAffineComponent(currentComponent,
        num_rows_in + num_padding,
        num_columns_in,
        num_rows_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        filterLayer->_weights->getTensorDesc().getPrecision().size(),
        biasPrecision.size(),
        quantized == nullptr ? 1 : quantized->_weights_quant.scale,
        quantized == nullptr ? 1 : quantized->_dst_quant.scale,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases,
        false);

    size_t num_data_bytes_out = num_rows_out * num_columns_in * outputs->getPrecision().size();
    size_t num_data_bytes_in = num_columns_in *
        ALIGN(num_rows_in, 8) * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in, num_rows_copied * inputs->getPrecision().size(), 0);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    {
        auto weightsElementSize = filterLayer->_weights->getTensorDesc().getPrecision().size();
        auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
        auto paddedWeights = elementsIn * num_rows_out;
        auto paddedWeightsSize = paddedWeights * weightsElementSize;

        // TODO: this can be improved to not generate unneeded weights at all

        size_t weights_stride =  (num_rows_in + num_rows_copied) * weightsElementSize;
        size_t weights_offset = weights_stride * num_rows_copied +  num_rows_copied * weightsElementSize;

        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void* data, size_t size) {
            size_t roffset = weights_offset;
            size_t woffset = 0;
            for (int i = 0; i < num_rows_out && size >= woffset; i++) {
                ie_memcpy(reinterpret_cast<uint8_t*>(data) + woffset,
                          size - woffset,
                          filterLayer->_weights->cbuffer().as<const uint8_t*>() + roffset,
                          num_rows_in * weightsElementSize);
                roffset += weights_stride;
                woffset += elementsIn * weightsElementSize;
            }
         }, 64);
    }

    if (filterLayer->_biases) {
        gnamem->readonly().push_ptr(ptr_biases,
            filterLayer->_biases->cbuffer().as<const void*>(),
            filterLayer->_biases->byteSize(),
            64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAGraphCompiler::AffineFilterPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto filterLayer = dynamic_cast<InferenceEngine::WeightableLayer*> (layer.get());

    if (filterLayer == nullptr) {
        return;
    }

    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    auto prevLayer = CNNNetPrevLayer(layer.get(), 0);
    if (!LayerInfo(prevLayer).isSplit() && !LayerInfo(prevLayer).isSlice()) {
        THROW_GNA_EXCEPTION << "Case  with Affine Aligning Filter for not Split/Slice layers is not implemented yet!";
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_columns_in = FROM_IR_DIM(inputs, 2);
    uint32_t num_rows_out = FROM_IR_DIM(outputs, 1);
    uint32_t num_rows_in = filterLayer->_weights->size() / num_rows_out;

    uint32_t num_padding = ALIGN(num_rows_in, 8) - num_rows_in;
    auto biasPrecision = filterLayer->_biases ? filterLayer->_biases->getTensorDesc().getPrecision() : outputs->getPrecision();
    auto& currentComponent = dnnComponents.addComponent(layer->name, "affine");

    dnn->InitAffineComponent(currentComponent,
        num_rows_in + num_padding,
        num_columns_in,
        num_rows_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        filterLayer->_weights->getTensorDesc().getPrecision().size(),
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
            begin(outputs->getDims()), end(outputs->getDims())) * 4;

    size_t num_data_bytes_in = num_columns_in *
        ALIGN(num_rows_in, 8) * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    if (num_padding == 0) {
        gnamem->readonly().push_ptr(ptr_weights,
            filterLayer->_weights->cbuffer().as<const void*>(),
            filterLayer->_weights->byteSize(),
            64);
    } else {
        auto elementsIn = (num_rows_in + num_padding) * num_columns_in;
        auto paddedWeights = elementsIn * num_rows_out;
        auto paddedWeightsSize = paddedWeights * filterLayer->precision.size();

        gnamem->readonly().push_initializer(ptr_weights, paddedWeightsSize, [=](void* data, size_t size) {
            size_t offset = 0;
            for (int i = 0; i < num_rows_out && size >= offset; i++) {
                ie_memcpy(reinterpret_cast<uint8_t*>(data) + offset, size - offset,
                    filterLayer->_weights->cbuffer().as<const uint8_t*>() + num_rows_in * i * filterLayer->precision.size(),
                    num_rows_in* filterLayer->precision.size());
                offset += (num_rows_in + num_padding) * filterLayer->precision.size();
            }
            }, 64);
    }

    if (filterLayer->_biases) {
        gnamem->readonly().push_ptr(ptr_biases,
            filterLayer->_biases->cbuffer().as<const void*>(),
            filterLayer->_biases->byteSize(),
            64);
    } else {
        gnamem->readonly().push_value(ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAGraphCompiler::PWLPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto* generic = dynamic_cast<GenericLayer*>(layer.get());
    std::string type;
    std::vector<intel_pwl_segment_t> ptr_pwl_segments;
    uint32_t num_rows;
    uint32_t num_columns;
    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;

    do {
        if (generic == nullptr) {
            type = layer->type;
            break;
        }

        if (InferenceEngine::details::CaselessEq<string>()(layer->type, "activation")) {
            type = generic->GetParamAsString("type");
            break;
        } else {
            type = layer->type;
            break;
        }
    } while (false);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    float output_pwl_scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
    float input_pwl_scale_factor = quantized != nullptr ? quantized->_src_quant.scale : 1.0f;

    auto orientation = kDnnInterleavedOrientation;

    if (inputs->getDims().size() == 4) {
        uint32_t w_dim_in = FROM_IR_DIM(inputs, 1);
        uint32_t h_dim_in = FROM_IR_DIM(inputs, 2);
        uint32_t c_dim_in = FROM_IR_DIM(inputs, 3);

        num_columns = (w_dim_in == 1) ? h_dim_in * c_dim_in : w_dim_in * c_dim_in;
        num_rows = 1;
    } else {
        num_columns = FROM_IR_DIM(inputs, 2);
        num_rows = FROM_IR_DIM(inputs, 1);
    }

    // TODO: solve this by layer level transformations
    auto concatAlignFilter = CNNNetPrevLayer(layer, 0);
    if (LayerInfo(concatAlignFilter).isConcatAlignFilter()) {
        auto rowsCopiedOffset = concatAlignFilter->GetParamAsInt("rows_copied_offset");
        if (rowsCopiedOffset != 0) {
            num_rows -= rowsCopiedOffset / outputs->getPrecision().size();
            layer->params["output_offset"] = std::to_string(rowsCopiedOffset);
        }
    }

    size_t num_data_bytes_out = num_columns * num_rows * outputs->getPrecision().size();
    size_t num_data_bytes_in = num_columns * num_rows * inputs->getPrecision().size();

    static InferenceEngine::details::caseless_unordered_map<std::string, DnnActivationType> supportedActivations = {
        {"sigmoid", kActSigmoid},
        {"tanh", kActTanh},
        {"relu", kActRelu},
        {"leakyrelu", kActLeakyRelu},
        {"clamp", kActKaldiLstmClipping},
        {"exp", kActExp},
        {"log", kActLog},
        {"sign", kActSign},
        {"abs", kActAbs},
        {"neglog", kActNegLog},
        {"neghalflog", kActNegHalfLog},
        {"identity", kActIdentity},
        {"softsign", kActSoftSign}
    };

    auto it = supportedActivations.find(type);
    if (it == supportedActivations.end()) {
        THROW_GNA_EXCEPTION << "Activation function type not yet supported: " << type;
    }
    auto activation_type = DnnActivation::fromType(it->second);
    if (it->second == kActRelu) {
        auto reluLayer = dynamic_cast<ReLULayer*>(layer.get());
        activation_type.args.leru.negative_slope = reluLayer != nullptr ? reluLayer->negative_slope : 0.0f;
    } else {
        activation_type.args.leru.negative_slope = 0.0f;
    }

    string actName = "unknown";

#ifdef PLOT
#define GET_ACTIVATION_NAME(name)\
case name:\
    actName = #name;\
    break
    switch (activation_type) {
        GET_ACTIVATION_NAME(kActSigmoid);
        GET_ACTIVATION_NAME(kActTanh);
        GET_ACTIVATION_NAME(kActRelu);
        GET_ACTIVATION_NAME(kActLeakyRelu);
        GET_ACTIVATION_NAME(kActKaldiLstmClipping);
        GET_ACTIVATION_NAME(kActIdentity);
        GET_ACTIVATION_NAME(kActSoftSign);
        GET_ACTIVATION_NAME(kActCustom);
        GET_ACTIVATION_NAME(kActExp);
        GET_ACTIVATION_NAME(kActLog);
        GET_ACTIVATION_NAME(kActSign);
        GET_ACTIVATION_NAME(kActAbs);
        GET_ACTIVATION_NAME(kActNegLog);
        GET_ACTIVATION_NAME(kActNegHalfLog);
    default: break;
    }
#endif

    auto& currentComponent = dnnComponents.addComponent(layer->name, actName);

    intel_pwl_segment_t* ptr_pwl_segments_target = nullptr;

    if (!gnaFlags->sw_fp32) {
        // TODO: generalize activation function code
        // now that scale factors are known, create PWL approximations to activation functions
        if (gnaFlags->uniformPwlDesign) {
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
            case kActSoftSign:ptr_pwl_segments.resize(SOFTSIGN_NUM_SEGMENTS);
                break;
            case kActCustom:
            default:
                THROW_GNA_EXCEPTION << "Activation function type not yet supported " << activation_type;
            }
            PwlDesign16(activation_type,
                &*ptr_pwl_segments.begin(),
                static_cast<uint32_t>(ptr_pwl_segments.size()),
                input_pwl_scale_factor,
                output_pwl_scale_factor);
        } else {
            PwlDesignOpt16(activation_type,
                ptr_pwl_segments,
                input_pwl_scale_factor,
                output_pwl_scale_factor);
        }
        ptr_pwl_segments_target = reinterpret_cast<intel_pwl_segment_t*>(&ptr_pwl_segments_target);
    }

    dnn->InitPiecewiseLinearComponent(currentComponent,
        activation_type,
        orientation,
        num_rows,
        num_columns,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        ptr_pwl_segments.size(),
        output_pwl_scale_factor,
        input_pwl_scale_factor,
        ptr_inputs,
        ptr_outputs,
        ptr_pwl_segments_target);

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    if (ptr_pwl_segments_target != nullptr) {
        gnamem->readonly().push_local_ptr(ptr_pwl_segments_target,
            &ptr_pwl_segments.front(),
            ptr_pwl_segments.size() * sizeof(intel_pwl_segment_t),
            64);
    }
}

void GNAGraphCompiler::PermutePrimitive(InferenceEngine::CNNLayerPtr layer) {
    if (LayerInfo(layer).isTrivialPermute()) {
        return;
    }
    auto layerOrder = layer->GetParamAsInts("order");
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    auto inputs = layer->insData.begin()->lock();
    auto inputsOrder = inputs->getTensorDesc().getDims();
    auto outputs = layer->outData.front();

    // squeeze order vector
    SizeVector squeezedInputOrder;
    for (auto input_shape : inputsOrder) {
        if (input_shape != 1) squeezedInputOrder.push_back(input_shape);
    }
    SizeVector squeezedOutputOrder;
    for (auto output_shape : layerOrder) {
        if (output_shape != 0) squeezedOutputOrder.push_back(output_shape);
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;

    if (squeezedInputOrder.size() > 2) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "unsupported permute (requested transpose is not 2D)";
    }

    if (std::min(squeezedInputOrder[0], squeezedInputOrder[1]) > 8) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "unsupported permute (minor dimension="
                            << std::min(squeezedInputOrder[0], squeezedInputOrder[1]) << " > 8)";
    }

    // now this can be run on GNA
    if (squeezedInputOrder[0] < squeezedInputOrder[1]) {  // interleave case
        if (ALIGN(squeezedInputOrder[1], 8) != squeezedInputOrder[1]) {
            THROW_GNA_LAYER_EXCEPTION(layer) << "unsupported permute (row size not a multiple of 8)";
        } else {
            auto& currentComponent = dnnComponents.addComponent(layer->name, "interleave");
            dnn->InitInterleaveComponent(currentComponent,
                                         squeezedInputOrder[0],
                                         squeezedInputOrder[1],
                                         inputs->getPrecision().size(),
                                         outputs->getPrecision().size(),
                                         (quantized == nullptr) ? 1.0f : quantized->_dst_quant.scale,
                                         ptr_inputs,
                                         ptr_outputs);
        }

    } else {  // deinterleave case
        if (ALIGN(squeezedInputOrder[0], 8) != squeezedInputOrder[0]) {
            THROW_GNA_LAYER_EXCEPTION(layer) << "[GNA plugin] unsupported permute (column size not a multiple of 8)";
        } else {
            auto& currentComponent = dnnComponents.addComponent(layer->name, "deinterleave");
            dnn->InitDeinterleaveComponent(currentComponent,
                                           squeezedInputOrder[0],
                                           squeezedInputOrder[1],
                                           inputs->getPrecision().size(),
                                           outputs->getPrecision().size(),
                                           quantized == nullptr ? 1 : quantized->_dst_quant.scale,
                                           ptr_inputs,
                                           ptr_outputs);
        }
    }

    size_t num_data_bytes_out = ALIGN(InferenceEngine::details::product(
            begin(outputs->getDims()), end(outputs->getDims())), 8)
                                * outputs->getPrecision().size();
    size_t num_data_bytes_in = squeezedInputOrder[0] * squeezedInputOrder[1] * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);
}

void SKIP(GNAGraphCompiler*, CNNLayerPtr) {}

void GNAGraphCompiler::CreateLayerPrimitive(CNNLayerPtr layer) {
    static const LayersBuilder layersBuilder[] = {
        {{"Input"}, [](GNAGraphCompiler*, CNNLayerPtr l) {}},  // skip input layers they are not used in GNA lib, only as a memory blobs
        {{"FullyConnected", "InnerProduct"}, CREATE(AffinePrimitive)},
        {{"ScaleShift"}, CREATE(DiagonalPrimitive)},
        {{"AffineFilter"}, CREATE(AffineFilterPrimitive)},
        {{"ConcatAlignFilter"}, CREATE(ConcatAlignFilterPrimitive)},
        {{"Const"}, CREATE(ConstPrimitive)},
        {{"Eltwise"}, CREATE(EltwisePrimitive)},  // same as diagonal while weights are not taken from network, rather than from another output
        {{"Split"}, SKIP},  // skip information about which part of prev layer need to consume handle during layer creation
        {{"Slice"}, SKIP},
        {{"link"}, SKIP},
        {{"clamp",
          "sigmoid",
          "relu",
          "tanh",
          "identity",
          "softsign",
          "exp",
          "log",
          "sign",
          "abs",
          "neglog",
          "neghalflog"},
          CREATE(PWLPrimitive)},
        {{"Convolution"}, CREATE(ConvolutionPrimitive)},
        {{"Permute"}, CREATE(PermutePrimitive)},  // permute of certain form (2D transpose) can be assimilated in followed FC layer
        {{"Pooling"}, CREATE(PoolingPrimitive)},
        {{"Power"} , CREATE(PowerPrimitive)},
        {{"Concat"}, CREATE(ConcatPrimitive)},
        {{"Reshape"}, SKIP},  // TODO: handled not in GNA but rather in GNA plugin
        {{"Squeeze"}, SKIP},  // TODO: handled not in GNA but rather in GNA plugin
        {{"Crop"}, CREATE(CropPrimitive)},
        {{"Copy"}, CREATE(CopyPrimitive)},
        {{"TensorIterator"}, SKIP},
        {{"LSTMCell"}, SKIP}
    };
    auto it = LayersBuilder::getStorage().find(layer->type);
    if (it != LayersBuilder::getStorage().end()) {
        it->second(this, layer);
    } else {
        THROW_GNA_EXCEPTION << "Unsupported layer: " << layer->name << ":" << layer->type;
    }
}

void GNAGraphCompiler::connectOutput(InferenceEngine::CNNLayerPtr layer, void *ptr, size_t num_data_bytes_out) {
    gnalog() << "Connecting output " << layer->name << " ...\n";
    // in case of Memory Layer it's input allocated in meminput layer
    if (layer->outData.size() == 1) {
        for (int j = 0; j != getInputTo(layer->outData.front()).size(); j++) {
            auto isNonFunctional = [](CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            };

            if (!CNNNetHasNextLayerSkipCertain(layer, 0, j, isNonFunctional)) {
                continue;
            }
            auto nextLayer = CNNNetGetNextLayerSkipCertain(layer, 0, j, isNonFunctional);

            if (!nextLayer.first) {
                gnalog() << "for layer: " << layer->name << "outData[0] has non functional connection at " << j;
            }

            auto nextMemoryLayerIt =
                    std::find_if(begin(memory_connection), end(memory_connection),
                                 [&](MemoryConnection::value_type &comp) {
                                     return comp.second.getOutput()->name == nextLayer.first->name;
                                 });
            if (nextMemoryLayerIt != memory_connection.end()) {
                auto &nextMemoryLayer = nextMemoryLayerIt->second;
                // memory layer not yet initialized
                if (nextMemoryLayer.reserved_size == 0) {
                    auto memorySize = InferenceEngine::details::product(nextMemoryLayer.getDims()) * nextMemoryLayer.elementSizeBytes();

                    gnamem->reserve_ptr(&nextMemoryLayer.gna_ptr, ALIGN64(memorySize), 64);
                    gnamem->bind_ptr(ptr, &nextMemoryLayer.gna_ptr, 0);

                    nextMemoryLayer.reserved_size = ALIGN64(memorySize);
                } else {
                    IE_ASSERT(nextMemoryLayer.reserved_size >= ALIGN64(num_data_bytes_out));
                    gnamem->bind_ptr(ptr, &nextMemoryLayer.gna_ptr, 0);
                }
                return;
            }
        }

        // if one of next direct or via split layers is concat...
        auto concatChild = [](CNNLayerPtr layer) {
            CNNLayerPtr concat;
            for (auto &&outLayer : getInputTo(layer->outData.front())) {
                auto nextLayer = outLayer.second;
                if (LayerInfo(nextLayer).isConcat()) {
                    concat = nextLayer;
                }
            }
            return concat;
        };
        auto splitChild = [](CNNLayerPtr layer) {
            std::list<CNNLayerPtr> split;
            for (auto &&outLayer : getInputTo(layer->outData.front())) {
                auto nextLayer = outLayer.second;
                if (LayerInfo(nextLayer).isSplit() || LayerInfo(nextLayer).isNonFunctional()) {
                    split.push_back(nextLayer);
                }
            }
            return split;
        };

        std::list<CNNLayerPtr> splits;
        auto concat = concatChild(layer);
        auto concatFather = layer;
        if (!concat) {
            splits = splitChild(layer);
        }

        while (!concat && !splits.empty()) {
            auto firstSplit = splits.front();
            concat = concatChild(firstSplit);
            // now concat prev layer would be this one
            concatFather = firstSplit;
            if (concat) {
                break;
            }
            // inserting into front of queue alow DFS simulation while searching
            splits.pop_front();
            auto nexSplits = splitChild(firstSplit);
            splits.insert(splits.begin(), nexSplits.begin(), nexSplits.end());
        }

        if (concat) {
            // concat father might be non functional - in that case lets skip it
            auto concatFatherActual =
                LayerInfo(concatFather).isNonFunctional() ?
                CNNNetPrevLayerSkipCertain(concatFather, 0, [](CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            }) : concatFather;

            auto& name = concatFatherActual->name;
            // we look for this concat layer pointer in extra concat map
            auto concatLayerInfo = concat_connection.find(concat->name);

            if (concatLayerInfo == concat_connection.end()) {
                THROW_GNA_EXCEPTION << "Cannot find corresponding concat layer: " << concat->name;
            }
            auto &concatLayerInfoItem = concatLayerInfo->second;

            // find this input in vector sum all outputs in primitive
            auto it = std::find_if(concatLayerInfoItem.concatInputLayers.begin(),
                                   concatLayerInfoItem.concatInputLayers.end(),
                                   [&name](GNAPluginNS::GNAConcatLayer::ConcatConnectedLayerInfo &item) {
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
                                                 (const std::pair<std::string, GNAPluginNS::GNAConcatLayer> &concatItem) -> bool {
                                             auto it = std::find_if(concatItem.second.concatInputLayers.begin(),
                                                                    concatItem.second.concatInputLayers.end(),
                                                                    [&concatLayerInfo]
                                                                            (const GNAPluginNS::GNAConcatLayer::ConcatConnectedLayerInfo &item) -> bool {
                                                                        return item.name == concatLayerInfo->first;
                                                                    });
                                             return it != concatItem.second.concatInputLayers.end();
                                         });
                    if (included == concat_connection.end()) {
                        gnamem->reserve_ptr(&concatLayerInfoItem.gna_ptr, ALIGN64(concatLayerInfoItem.reserved_size), 64);

                        for (auto &&inputLayer : concatLayerInfoItem.concatInputLayers) {
                            if (InferenceEngine::details::CaselessEq<std::string>()
                                    (inputLayer.name, "input")) {
                                inputDesc->bytes_allocated_for_input[inputLayer.name] = inputLayer.tensorSize;
                            }
                        }
                        concatLayerInfoItem.input_allocated = true;
                    }
                    concatLayerInfo->second.output_allocation_flag = true;
                }
                // output offset precalculated to serve GNAAlignment requirements
                auto output_offset = it->offset;
                if (layer->params.find("output_offset") != layer->params.end()) {
                    output_offset = layer->GetParamAsInt("output_offset");
                }
                gnamem->bind_ptr(ptr, &concatLayerInfoItem.gna_ptr, output_offset);
            }
            return;
        }
    }

    intel_dnn_component_t * unused_input = nullptr;
    if (gnaFlags->compact_mode) {
        unused_input = find_first_unused_input(layer);
        if (unused_input != nullptr) {
            gnamem->bind_ptr(ptr, &unused_input->ptr_inputs, 0, ALIGN64(num_data_bytes_out));
        }
    }
    // cannot reuse suitable input
    if (unused_input == nullptr) {
        gnamem->reserve_ptr(ptr, ALIGN64(num_data_bytes_out), 64);
    }
}

GNAPluginNS::ConnectionDetails GNAGraphCompiler::connectInput(CNNLayerPtr layer, void *ptr, size_t num_data_bytes_in, int32_t offset, int idx) {
    // selecting particular input layers
    // auto prevLayer = CNNNetPrevLayer(layer, idx);
    auto prevLayer = CNNNetPrevLayerSkipCertain(layer, idx, [](CNNLayerPtr l) {
        return LayerInfo(l).isNonFunctional();
    });


    gnalog() << "Connecting input " << layer->name << " to " << prevLayer->name << " ...\n";

    // real input not a memory input
    if (LayerInfo(prevLayer).isInput()) {
        if (0 == inputDesc->bytes_allocated_for_input[prevLayer->name]) {
            // real allocation pointer will be kept in ptr not in ptf_inputs_global
            if (offset < 0) {
                gnamem->push_value(ptr,
                                   static_cast<uint8_t>(0),
                                   num_data_bytes_in,
                                   64);
            } else {
                gnamem->push_value(&inputDesc->getPtrInputsGlobal(prevLayer->name).front(),
                                   static_cast<uint8_t>(0),
                                   num_data_bytes_in,
                                   64);
            }

            inputDesc->bytes_allocated_for_input[prevLayer->name] = num_data_bytes_in;
        }
        if (ALIGN(num_data_bytes_in, 64) > ALIGN(inputDesc->bytes_allocated_for_input[prevLayer->name], 64)) {
            THROW_GNA_EXCEPTION
                    << "Layer: " << layer->name
                    << " Cannot bind pointer to already allocated input(" << prevLayer->name
                    << "), due to size_allocated=" << inputDesc->bytes_allocated_for_input[prevLayer->name]
                    << ", and size_requested=" << num_data_bytes_in;
        }

        if (offset >= 0) {
            gnamem->bind_ptr(ptr, &inputDesc->getPtrInputsGlobal(prevLayer->name).front(), offset);
        } else {
            gnamem->bind_ptr(&inputDesc->getPtrInputsGlobal(prevLayer->name).front(), ptr, -offset);
        }

        return prevLayer;
    }
    // const input
    if (LayerInfo(prevLayer).isConst()) {
        if (offset >= 0) {
            gnamem->bind_ptr(ptr, const_connections[prevLayer->name], offset);
        } else {
            gnamem->bind_ptr(const_connections[prevLayer->name], ptr, -offset);
        }

        return prevLayer;
    }

    LayerInfo layerInfoObj(prevLayer);

    // connecting to split/slice splitiing layers
    if (layerInfoObj.isSplit() || layerInfoObj.isSlice()) {
        auto& splittingLayer = prevLayer;
        auto& splitName = splittingLayer->name;

        // we look for this split layer pointer in pre calculated map
        auto splitLayerInfo = split_connection.find(splitName);

        if (splitLayerInfo != split_connection.end()) {
            auto &splitLayerInfoItem = splitLayerInfo->second;
            // find this input in vector sum all outputs in primitive
            auto it = std::find_if(splitLayerInfoItem.splitOutputLayers.begin(),
                                   splitLayerInfoItem.splitOutputLayers.end(),
                                   [&idx, &layer](GNAPluginNS::GNASplitLayer::SplitConnectedLayerInfo &item) {
                                       return item.connectedTo == layer && item.insDataIdx == idx;
                                   });

            if (it != splitLayerInfoItem.splitOutputLayers.end()) {
                gnalog()  << "Connecting " << splitName << " input \n";
                auto res = connectInput(splittingLayer, ptr, splitLayerInfoItem.reserved_size, it->offset + offset, 0);
                gnalog()  << "Connected \n";
                return res;
            }
        }
        THROW_GNA_EXCEPTION << prevLayer->type << " layer: " << splitName << " is not included in extra map";
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
    auto prevDnnLayer = dnnComponents.findComponent(prevLayer);

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
        // TODO: this is duplicate with connect output
        auto& memoryLayer = prevMemoryLayer->second;
        if (memoryLayer.reserved_size == 0) {
            auto memorySize = InferenceEngine::details::product(memoryLayer.getDims()) * memoryLayer.elementSizeBytes();

            // negative offset used for  indicate that memory layer should be bound to given buffer
            if (offset >= 0) {
                memorySize = std::max(memorySize, num_data_bytes_in);
                gnamem->reserve_ptr(&memoryLayer.gna_ptr, ALIGN64(memorySize), 64);
                gnamem->bind_ptr(ptr, &memoryLayer.gna_ptr, offset);
            } else {
                if (num_data_bytes_in > memorySize) {
                    THROW_GNA_LAYER_EXCEPTION(layer) <<" invalid allocation request of "
                                                     << num_data_bytes_in << " is more then state tensor size of: " << memorySize;
                }
                gnamem->bind_ptr(&memoryLayer.gna_ptr, ptr, -offset);
            }

            memoryLayer.reserved_size = ALIGN64(memorySize);
        } else {
            IE_ASSERT(memoryLayer.reserved_size >= ALIGN64(num_data_bytes_in));
            gnamem->bind_ptr(ptr, &memoryLayer.gna_ptr, offset);
        }

        return prevLayer;
    }

    // several layers are to be skipped right now
    if (LayerInfo(prevLayer).isNonFunctional()) {
        gnalog()  << "Skipping non functional layer: " << prevLayer->name << "\n";
        return connectInput(prevLayer, ptr, num_data_bytes_in, offset, 0);
    }

    // permute layer resulted in trivial permute
    if (LayerInfo(prevLayer).isPermute()) {
        if (!LayerInfo(prevLayer).isTrivialPermute()) {
            // we should have GNA primitive for it
            THROW_GNA_EXCEPTION << "missed gna primitive for permute: " << prevLayer->name;
        }
        gnalog()  << "Skipping trivial permute layer: " << prevLayer->name << "\n";
        return connectInput(prevLayer, ptr, num_data_bytes_in, offset, 0);
    }


    THROW_GNA_EXCEPTION << "Cannot connect input for: " << layer->name;
}

void GNAGraphCompiler::Reset() {
    for (auto && memLayer : memory_connection) {
        std::memset(memLayer.second.gna_ptr, 0, memLayer.second.reserved_size);
    }
    for (auto && concatLayer : concat_connection) {
        std::memset(concatLayer.second.gna_ptr, 0, concatLayer.second.reserved_size);
    }
}
