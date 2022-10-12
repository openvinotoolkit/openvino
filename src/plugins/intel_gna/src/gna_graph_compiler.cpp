// Copyright (C) 2018-2022 Intel Corporation
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

#include <legacy/ie_layers.h>
#include <ie_algorithm.hpp>
#include <debug.h>

#include "gna_graph_compiler.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin_log.hpp"
#include "layers/gna_layer_info.hpp"
#include "ie_memcpy.h"
#include "caseless.hpp"
#include "backend/am_intel_dnn.hpp"
#include "runtime/pwl.h"
#include "gna_graph_tools.hpp"
#include "frontend/model_quantizer.hpp"
#include "layers/layers_builder.hpp"
#include "layers/gna_concat_layer.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "layers/gna_crop_layer.hpp"
#include "layers/gna_fake_quantize_layer.hpp"
#include "round_float_define.hpp"
#include "gna_groups.hpp"
#include "backend/gna_limitations.hpp"
#include "descriptions/gna_desc.hpp"
#include "ops/pwl.hpp"

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;
using namespace memory;

static bool CheckIFLastComponentIsPrecededByConv2D(const GNAPluginNS::backend::DnnComponents::storage_type& components,
                                            bool verify_with_pooling = true) {
    bool proceded_by_conv2D = false;
    auto last_element = components.rbegin();
    if (components.size() > 1) {
        last_element++;
        if (last_element->dnnComponent.operation == kDnnConvolutional2dOp) {
            proceded_by_conv2D = true;
        } else if (verify_with_pooling && components.size() > 2) {
            auto prev_operation = last_element->dnnComponent.operation;
            last_element++;
            if (last_element->dnnComponent.operation == kDnnConvolutional2dOp) {
                proceded_by_conv2D = (prev_operation == kDnnMaxPoolOp);
            }
        }
    }
    return proceded_by_conv2D;
}

#define CREATE(name) [](GNAGraphCompiler *p, CNNLayerPtr l) {p->name(l);}

void GNAGraphCompiler::setGNAMemoryPtr(std::shared_ptr<GNAPluginNS::gna_memory_type> gnaMemPtr) {
    this->gnamem = std::move(gnaMemPtr);
}

void GNAGraphCompiler::setDNNPtr(std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnnPtr) {
    this->dnn = std::move(dnnPtr);
}

void GNAGraphCompiler::setInputsPtr(std::shared_ptr<GNAPluginNS::GnaInputs> inputsPtr) {
    this->inputs_ptr_ = std::move(inputsPtr);
}

void GNAGraphCompiler::setGNAFlagsPtr(std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlagsPtr) {
    this->gnaFlags = std::move(gnaFlagsPtr);
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

        // concat align layer can have additional padding, so the size of layer needs to be calculated
        // based on original number of rows
        if (ptrConcatLayerInput->CheckParamPresence("original_num_rows")) {
            layer_size = ptrConcatLayerInput->GetParamAsInt("original_num_rows") * dataInput->getPrecision().size();
        }

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
            auto outFunctionalLayer = CNNNetCheckNextLayerSkipCertain(layer, i, j, true, [](CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            });

            if (!outFunctionalLayer.first) {
                output_layer_size =
                    InferenceEngine::details::product(begin(layer->outData[i]->getDims()),
                                                      end(layer->outData[i]->getDims())) * layer->outData[i]->getPrecision().size();
                continue;
            }

            for (int idx : outFunctionalLayer.second) {
                auto dataOutput = outFunctionalLayer.first->insData[idx].lock();

                padding = std::max(padding, LayerInfo(outFunctionalLayer.first).paddingSize())
                                                            * dataOutput->getPrecision().size();
                output_layer_size =
                        InferenceEngine::details::product(begin(dataOutput->getDims()),
                                                        end(dataOutput->getDims())) * dataOutput->getPrecision().size();

                if (LayerInfo(outFunctionalLayer.first).isConvolutionFilter()) {
                    size_t aligned64_offset = outFunctionalLayer.first->GetParamAsInt("offset");
                    layerInfoItem.splitOutputLayers.emplace_back(
                        outFunctionalLayer.first,
                        idx,
                        aligned64_offset * dataOutput->getPrecision().size(),
                        output_layer_size);
                } else {
                    layerInfoItem.splitOutputLayers.emplace_back(
                        outFunctionalLayer.first, idx, split_size, output_layer_size);
                }
             }
        }

        // in case of unconnected split - we need properly increment size
        if (getInputTo(layer->outData[i]).empty()) {
            output_layer_size =
                    InferenceEngine::details::product(begin(layer->outData[i]->getDims()),
                                                      end(layer->outData[i]->getDims())) * layer->outData[i]->getPrecision().size();
        }

        split_size += padding + output_layer_size;
    }
    layerInfoItem.reserved_size = split_size;
    split_connection.emplace(id, layerInfoItem);
}

void GNAPluginNS::GNAGraphCompiler::SetValidatorTarget(const std::string& target) {
    auto temp = GNALimitations::Cnn2D::AbstractValidator::Create(target);
    cnn2dValidator.reset(temp.release());
}

void GNAPluginNS::GNAGraphCompiler::ValidateCnn2D(const std::string& name,
                                                  const uint32_t inHeight,
                                                  const uint32_t inWidth,
                                                  const uint32_t inChannels,
                                                  const uint32_t kH,
                                                  const uint32_t kW,
                                                  const uint32_t kN,
                                                  const uint32_t strideH,
                                                  const uint32_t strideW,
                                                  const uint32_t dilH,
                                                  const uint32_t dilW,
                                                  OvGnaType inPrecision) const {
    if (cnn2dValidator) {
        cnn2dValidator
            ->ValidateCnn2D(name, inHeight, inWidth, inChannels, kH, kW, kN, strideH, strideW, dilH, dilW, inPrecision);
    } else {
        THROW_GNA_EXCEPTION << "No Cnn2D validator found for layer " << name;
    }
}

void GNAPluginNS::GNAGraphCompiler::ValidatePooling2D(const std::string& name,
                                                      const uint32_t windowH,
                                                      const uint32_t windowW,
                                                      const uint32_t strideH,
                                                      const uint32_t strideW) const {
    if (cnn2dValidator) {
        cnn2dValidator->ValidatePooling2D(name, windowH, windowW, strideH, strideW);
    } else {
        THROW_GNA_EXCEPTION << "No Pooling2D validator found for layer " << name;
    }
}

bool GNAPluginNS::GNAGraphCompiler::IsCnn2DInputPaddingSupported(const std::string& name) const {
    if (cnn2dValidator) {
        return cnn2dValidator->IsPaddingSupported();
    } else {
        THROW_GNA_EXCEPTION << "No Cnn2D input padding validator found for layer " << name;
    }
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
    gnamem->getQueue(REGION_AUTO)->bind_initializer(nullptr, ptr_for_const_blob, [const_blob](void* data, size_t size) {
        ie_memcpy(data, size, const_blob->buffer(), const_blob->byteSize());
        });
}

void GNAGraphCompiler::assertConvolutionLayoutProper(const InferenceEngine::DataPtr& data) {
    if (data->getLayout() != Layout::NHWC &&
        data->getLayout() != Layout::NCHW &&
        data->getLayout() != Layout::NC) {
        THROW_GNA_EXCEPTION << "layer: \"Convolution\" with layout " << data->getLayout() << " isn't currently supported on GNA";
    }
}

/**
 * Create AMIntelDNN Convolutional1DComponent from ConvolutionLayer
 *
 * GNA Convolution input is NHCW and output is transposed to NHWC
 *
 * OpenVINO default layout is NCHW
 * TensorFlow default layout is NHWC
 *
 * There is option in ModelOptimizer
 * --disable_nhwc_to_nchw
 *                      Disables default translation from NHWC to NCHW
 * By default MO converts TensorFlow default NHWC to OpenVino default NCHW
 * So when MR was created with this option layout will be NHWC
 *
 * @param layer Pointer to ConvolutionLayer
 */
void GNAGraphCompiler::ConvolutionPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& convolution = dynamic_cast<ConvolutionLayer&>(*layer.get());
    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());

    const auto inputs = layer->insData.front().lock();
    const auto outputs = layer->outData.front();
    assertConvolutionLayoutProper(inputs);

    const auto in_batch = GetDataDimSize(inputs, InferenceEngine::DataDimName::N);
    const auto in_channels = GetDataDimSize(inputs, InferenceEngine::DataDimName::C);
    auto in_height = GetDataDimSize(inputs, InferenceEngine::DataDimName::H);
    auto in_width = GetDataDimSize(inputs, InferenceEngine::DataDimName::W);

    const auto out_batch = GetDataDimSize(outputs, InferenceEngine::DataDimName::N);
    const auto out_channels = GetDataDimSize(outputs, InferenceEngine::DataDimName::C);
    auto out_height = GetDataDimSize(outputs, InferenceEngine::DataDimName::H);
    auto out_width = GetDataDimSize(outputs, InferenceEngine::DataDimName::W);

    if (in_height > 1 && in_width == 1) {
        std::swap(in_height, in_width);
        std::swap(out_height, out_width);
        std::swap(convolution._kernel_x, convolution._kernel_y);
        std::swap(convolution._padding_x, convolution._padding_y);
        std::swap(convolution._stride_x, convolution._stride_y);
        std::swap(convolution._dilation_x, convolution._dilation_y);
    }

    auto in_kernel_w = convolution._kernel_x;
    auto in_kernel_h = convolution._kernel_y;
    bool transpose_h_w = false;

    // Map 2d convolution to 1d if it's possible
    if (GNAConvolutionLayer::isMappableFrom2DTo1D(in_height, in_width, in_channels,
                                                  convolution._kernel_y, convolution._kernel_x,
                                                  convolution._stride_y, convolution._stride_x)) {
        transpose_h_w = (in_height == convolution._kernel_y);
        in_width *= in_height;
        in_height = 1;
        out_width *= out_height;
        out_height = 1;
        convolution._stride_x *= transpose_h_w ? (convolution._stride_y * convolution._kernel_y) :
                                                 (convolution._stride_y * convolution._kernel_x);
        convolution._kernel_x *= convolution._kernel_y;
        convolution._kernel_y = 1;
        // since _kernel_y = 1 && in_height = 1
        // it will be finalized with finalizeConvolution1DPrimitive()
        // unless some other exception thrown
    }

    if (in_batch != 1 || out_batch != 1) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "with batch size not equals 1 is not supported";
    }

    if (convolution._dilation_x != 1 || convolution._dilation_y != 1) {
        // TODO: Issue 24839
        THROW_GNA_LAYER_EXCEPTION(layer) << "with dilation is not supported on GNA";
    }

    if (convolution._kernel_x > in_width * in_height) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Kernel dimensions X (" << convolution._kernel_x << ")"
            << " is bigger than total input dimensions WxH (" << in_width << "x" << in_height << ")";
    }

    if (out_channels != convolution._out_depth) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Output channels do not equal output depth. "
            << out_channels << " vs " << convolution._out_depth;
    }

    if (dnn->new_num_conv_columns) {
        dnn->new_num_conv_columns = 0;
    }

    if (GNAConvolutionLayer::isConv2D(in_height, in_width, in_channels, convolution._kernel_y, convolution._kernel_x) ||
        in_height != 1) {
        // TensorFlow default layout is NHWC
        // OpenVino Default layout is   NCHW
        // GNA Convolution input is     NHCW (old) or NHWC (new)
        // When layer layout is in NHWC it means that is was created by PassManager
        return finalizeConvolution2DPrimitive(layer, in_batch, in_channels, in_height, in_width,
                                              out_batch, out_channels, out_height, out_width);
    }
    finalizeConvolution1DPrimitive(layer, in_batch, in_channels, in_width,
                                   out_batch, out_channels, out_width, in_kernel_w, in_kernel_h, transpose_h_w);
}

void GNAGraphCompiler::finalizeConvolution1DPrimitive(InferenceEngine::CNNLayerPtr layer,
    uint32_t in_batch, uint32_t in_channels, uint32_t in_width,
    uint32_t out_batch, uint32_t out_channels, uint32_t out_width,
    uint32_t in_kernel_w, uint32_t in_kernel_h, bool transpose_h_w) {
    auto& convolution = dynamic_cast<ConvolutionLayer&>(*layer.get());
    printConvolutionLayer(convolution);

    const auto inputs = convolution.insData.front().lock();
    const auto outputs = convolution.outData.front();

    if (layer->GetParamAsString("auto_pad", "explicit") != "valid" &&
        (convolution._padding[0] != 0 || convolution._padding[0] != 0 ||
         convolution._pads_end[0] != 0 || convolution._pads_end[1] != 0)) {
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "Padding isn't supported by GNA";
    }

    std::size_t calculated_out_width = (in_width - convolution._kernel_x + 2 * convolution._padding_x) / convolution._stride_x + 1;
    if (out_width != calculated_out_width) {
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "Invalid output configuration. "
            << calculated_out_width << " != " << out_width;
    }

    IE_ASSERT(convolution._kernel_y == 1);
    uint32_t total_conv_kernel_size = convolution._kernel_x * convolution._out_depth * in_channels;
    const uint32_t single_conv_kernel_size = convolution._kernel_x * in_channels;
    auto actual_kernel_size = details::product(convolution._weights->getTensorDesc().getDims());
    if (total_conv_kernel_size != actual_kernel_size) {
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "Weights size does not equal kernel size "
            << actual_kernel_size << " vs " << total_conv_kernel_size;
    }

    // GNA HW Convolution 1D layer natively supports single channel input and filter
    // to use it for multiple channels input and filter
    // the convolution stride must be multiplied accordingly
    auto effectiveStride = in_channels * convolution._stride_x;

    if (convolution._stride_y == 1 && in_width == 1 && convolution._stride_x != 1) {
        // TODO: investigate whether this condition is needed at all
        // seams that if in_width == 1, then convolution._stride_x == 1 as well
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "Convolution 1D Layer has horizontal stride != 1 despite input width == 1\n";
    }

    // padding of convolution kernel to be multiply of 8
    // additionally have to pad filter for stride > filter since
    // GNA HW supports CNN1D with convolution stride not greater than filter length
    const auto num_filter_coefficients = ALIGN(std::max(single_conv_kernel_size, effectiveStride), 8);
    const auto num_conv_kernel_padding = num_filter_coefficients - single_conv_kernel_size;
    if (num_conv_kernel_padding == 0) {
        gnalog() << LAYER_NAME(&convolution) << "Kernel is aligned \n";
    } else {
        gnalog() << LAYER_NAME(&convolution) << "Kernel padding is " << num_conv_kernel_padding << "\n";
    }

    // have to pad input to let last kernel meets it's corresponding input
    const auto num_inputs = in_width * in_channels;

    uint32_t num_input_padding = ALIGN(num_inputs, 8) - num_inputs;
    uint32_t num_columns_in = num_inputs + num_input_padding;

    const uint32_t num_filters = convolution._out_depth;
    uint32_t num_columns_out = (((num_inputs - num_filter_coefficients) / effectiveStride) + 1) * num_filters;
    uint32_t num_columns_out_unpadded = (((num_inputs - single_conv_kernel_size) / effectiveStride) + 1) * num_filters;

    uint32_t original_input_padding = num_input_padding;
    uint32_t additional_padding = 0;

    // if kernel padding to multiple of 8 will cause missed outputs, need to pad further
    while (num_columns_out < out_batch * out_channels * out_width) {
        num_input_padding = original_input_padding + additional_padding;
        num_columns_in = num_inputs + num_input_padding;
        num_columns_out = (((num_inputs + num_input_padding - num_filter_coefficients) / effectiveStride) + 1) * num_filters;
        dnn->new_num_conv_columns = num_columns_out;
        additional_padding += 8;
    }

    if (num_input_padding == 0) {
        gnalog() << LAYER_NAME(&convolution) << "Inputs are aligned \n";
    } else {
        gnalog() << LAYER_NAME(&convolution) << "Inputs padding is " << num_input_padding << "\n";
    }

    if (num_columns_out_unpadded != out_batch * out_channels * out_width) {
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "Number of output columns does not equal output tensor size "
            << num_columns_out_unpadded << " vs " << out_batch * out_channels * out_width;
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    // TODO: questionable why for biases that are not in IR we inventing precision
    auto biasPrecision = convolution._biases ? convolution._biases->getTensorDesc().getPrecision() : outputs->getPrecision();

    uint32_t num_bytes_per_input = inputs->getPrecision().size();
    uint32_t num_bytes_per_output = outputs->getPrecision().size();
    uint32_t num_bytes_per_weight = convolution._weights->getTensorDesc().getPrecision().size();
    uint32_t num_bytes_per_bias = biasPrecision.size();

    float weight_scale_factor = getScaleFactor(layer, QuantizedDataType::weights);
    float output_scale_factor = getScaleFactor(layer, QuantizedDataType::output);

    auto& currentComponent = dnnComponents.addComponent(convolution.name, "convolution");
    dnn->InitConvolutional1DComponent(currentComponent,
        num_columns_in,
        num_columns_out,
        num_bytes_per_input,
        num_bytes_per_output,
        num_bytes_per_weight,
        num_bytes_per_bias,
        num_filters,
        num_filter_coefficients,
        effectiveStride,
        weight_scale_factor,
        output_scale_factor,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases);

    // Keep both variants of kaldi models working:
    // Old one has layout which is different from NHWC
    // New one has layout NHWC, but it is mapped from 2d by H
    if (inputs->getLayout() == Layout::NHWC && !transpose_h_w) {
        currentComponent.orientation_in  = kDnnInterleavedOrientation;
        currentComponent.orientation_out = kDnnInterleavedOrientation;
    }

    size_t num_data_bytes_out = num_columns_out * outputs->getPrecision().size();
    size_t num_data_bytes_in = (num_inputs + num_input_padding) * inputs->getPrecision().size();

    auto connectedInputLayer = connectInput(layer, ptr_inputs, num_data_bytes_in).input;
    // Skip FakeQuantize and ScaleShift between Convolution and Input
    if (LayerInfo(connectedInputLayer).isFakeQuantize()) {
            connectedInputLayer = CNNNetPrevLayerSkipCertain(connectedInputLayer, 0, [](CNNLayerPtr l) {
            return LayerInfo(l).isScaleShift();
        });
    }

    // TODO: convolution might be not the first layer in sorted order but connected via split for example - dont know how kaldi will handle that
    if (!dnn->do_rotate_input) {
         if ((inputs->getLayout() != Layout::NHWC || transpose_h_w) && LayerInfo(connectedInputLayer).isInput()) {
            //  Kaldi features are opposite orientation
            dnn->do_rotate_input = true;
            dnn->num_rotate_rows = effectiveStride;
            dnn->num_rotate_columns = num_inputs / effectiveStride;
        } else {
            dnn->do_rotate_input = false;
        }
    }

    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    // Transpose H with W or C with HW
    auto A = transpose_h_w ? in_kernel_h : in_channels;
    auto B = transpose_h_w ? in_kernel_w : convolution._kernel[X_AXIS];

    std::vector<uint8_t> transposedWeights;
    for (uint32_t k = 0; k < num_filters; k++) {
        uint8_t * ptr_filt_current
            = convolution._weights->cbuffer().as<uint8_t*>() +
            k * A * B * convolution.precision.size();
        auto transposedPart = transposeMatrix(ptr_filt_current, convolution.precision.size(), A, B);
        transposedWeights.insert(transposedWeights.end(), transposedPart.begin(), transposedPart.end());
    }
    if (transposedWeights.size() != convolution._weights->byteSize()) {
        THROW_GNA_LAYER_EXCEPTION(&convolution) << "weights was transposed incorrectly. "
            << transposedWeights.size() << ' '
            << convolution._weights->byteSize();
    }

    if (num_conv_kernel_padding == 0) {
        gnamem->getQueue(REGION_RO)->push_local_ptr(layer, ptr_weights,
            transposedWeights.data(),
            convolution._weights->byteSize(),
            64);
    } else {
        auto paddedWeights = num_filter_coefficients * num_filters;
        auto paddedWeightsSize = paddedWeights * convolution.precision.size();
        std::string layerName = (layer)->type + " layer : \"" + (layer)->name + "\" ";
        const auto cpSize = convolution.precision.size();

        auto initializer = [paddedWeightsSize,
                            layerName,
                            num_conv_kernel_padding,
                            cpSize,
                            transposedWeights,
                            num_filters,
                            single_conv_kernel_size](void* data, std::size_t size) {
            if (paddedWeightsSize > size) {
                THROW_GNA_EXCEPTION << layerName << "size is less than paddedWeightsSize";
            }
            std::size_t offset = 0;
            std::vector<uint8_t> padding_zeros(num_conv_kernel_padding * cpSize, 0);
            uint8_t* dstPtr = reinterpret_cast<uint8_t*>(data);
            for (int i = 0; i < num_filters; i++) {
                ie_memcpy(dstPtr + offset,
                    size - offset,
                    transposedWeights.data() + single_conv_kernel_size * i * cpSize,
                    single_conv_kernel_size * cpSize);
                offset += single_conv_kernel_size * cpSize;
                ie_memcpy(dstPtr + offset,
                    size - offset,
                    &padding_zeros[0],
                    padding_zeros.size());
                offset += padding_zeros.size();
            }
        };

        gnamem->getQueue(REGION_RO)->push_initializer(layer, ptr_weights,
            paddedWeightsSize,
            initializer,
            64);
    }

    if (convolution._biases) {
        gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_biases,
            convolution._biases->cbuffer().as<const void*>(),
            convolution._biases->byteSize(),
            64);
    } else {
        gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, out_channels, 64);
    }
}

void GNAGraphCompiler::finalizeConvolution2DPrimitive(InferenceEngine::CNNLayerPtr layer,
    uint32_t in_batch, uint32_t in_channels, uint32_t in_height, uint32_t in_width,
    uint32_t out_batch, uint32_t out_channels, uint32_t out_height, uint32_t out_width) {

    auto& convolution = dynamic_cast<ConvolutionLayer&>(*layer.get());

    // TODO add function
    // printConvolution2DLayer(convolution);

    auto effectiveInputWidth = in_width;
    auto effectiveInputHeight = in_height;

    if (!IsCnn2DInputPaddingSupported(convolution.name) &&
        (convolution._padding_x != 0 || convolution._padding_y != 0 ||
        convolution._pads_end.at(X_AXIS) != 0 || convolution._pads_end.at(Y_AXIS) != 0)) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Convolution's input padding is not supported";
    }

    if (convolution._padding_x != convolution._pads_end.at(X_AXIS)) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Convolution's input padding is not symetric along X axis";
    }
    if (convolution._padding_y != convolution._pads_end.at(Y_AXIS)) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Convolution's input padding is not symetric along Y axis";
    }
    convolution._padding_x = convolution._pads_end.at(X_AXIS);
    convolution._padding_y = convolution._pads_end.at(Y_AXIS);

    if (convolution._kernel_x > effectiveInputWidth ||
        convolution._kernel_y > effectiveInputHeight) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Kernel dimensions XY (" << convolution._kernel_x << ", " << convolution._kernel_y << ")"
            << " are bigger than input dimensions WH (" << in_width << "," << in_height << ")";
    }
    const auto inputs = convolution.insData.front().lock();
    const auto outputs = convolution.outData.front();

    // have to pad input to let last kernel meets it's corresponding input
    const auto num_inputs = in_width * in_height * in_channels;
    uint32_t num_input_padding = ALIGN(num_inputs, 8) - num_inputs;

    const uint32_t filter_n = convolution._out_depth;

    // if kernel padding to multiple of 8 will cause missed outputs, need to pad further
    if (num_input_padding == 0) {
        gnalog() << LAYER_NAME(&convolution) << "Inputs are aligned \n";
    } else {
        gnalog() << LAYER_NAME(&convolution) << "Inputs padding is " << num_input_padding << "\n";
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    // TODO: questionable why for biases that are not in IR we inventing precision
    auto biasPrecision = convolution._biases ? convolution._biases->getTensorDesc().getPrecision() : outputs->getPrecision();

    const auto inputPrec = OvGnaTypeIntFromBytes(inputs->getPrecision().size());
    const auto outputPrec = OvGnaTypeIntFromBytes(outputs->getPrecision().size());
    const auto weightPrec = OvGnaTypeIntFromBytes(convolution._weights->getTensorDesc().getPrecision().size());
    const auto biasPrec = OvGnaTypeIntFromBytes(biasPrecision.size());

    ValidateCnn2D(layer->name,
        in_height, in_width, in_channels,
        convolution._kernel_y, convolution._kernel_x, filter_n, convolution._stride_y, convolution._stride_x,
        convolution._dilation_y, convolution._dilation_x, inputPrec);

    float weight_scale_factor = getScaleFactor(layer, QuantizedDataType::weights);
    float output_scale_factor = getScaleFactor(layer, QuantizedDataType::output);

    auto& currentComponent = dnnComponents.addComponent(convolution.name, "convolution");
    dnn->InitConvolutional2DComponent(currentComponent,
        { {in_batch, in_height, in_width, in_channels}, inputPrec, {} },   //NHWC for GNA
        { {out_batch, out_height, out_width, out_channels}, outputPrec, {} },
        { {filter_n, convolution._kernel_y, convolution._kernel_x, in_channels}, weightPrec, {} },
        { {filter_n}, biasPrec, {} },
        { convolution._stride_y, convolution._stride_x},
        { convolution._padding_y, convolution._padding_x },
        weight_scale_factor,
        output_scale_factor,
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases);
    currentComponent.num_bytes_per_input = inputs->getPrecision().size();
    currentComponent.num_bytes_per_output = outputs->getPrecision().size();

    if (inputs->getLayout() == Layout::NHWC) {
        currentComponent.orientation_in = kDnnInterleavedOrientation;
        currentComponent.orientation_out = kDnnInterleavedOrientation;
    }

    size_t num_data_bytes_out =
        InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
        * outputs->getPrecision().size();

    size_t num_data_bytes_in = (num_inputs + num_input_padding) * inputs->getPrecision().size();

    auto connectedInputLayer = connectInput(layer, ptr_inputs, num_data_bytes_in).input;

    // TODO: convolution might be not the first layer in sorted order but connected via split for example - dont know how kaldi will handle that
    if (!dnn->do_rotate_input && inputs->getLayout() != Layout::NHWC && LayerInfo(connectedInputLayer).isInput()) {
        //  Kaldi features are opposite orientation
        dnn->do_rotate_input = true;
        dnn->num_rotate_rows = in_channels;
        if (in_height != 1) {
            dnn->num_rotate_rows *= convolution._stride_y;
        }
        if (in_width != 1) {
            dnn->num_rotate_rows *= convolution._stride_x;
        }
        dnn->num_rotate_columns = num_inputs / dnn->num_rotate_rows;
    }

    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    const auto kernelHW = convolution._kernel_y * convolution._kernel_x;

    std::vector<uint8_t> transposedWeights;
    const auto singleKernelSize = in_channels* kernelHW * convolution.precision.size();
    const auto kernelPad = Gna2RoundUp(singleKernelSize, 16) - singleKernelSize;
    for (uint32_t k = 0; k < convolution._out_depth; k++) {
        uint8_t* ptr_filt_current
            = convolution._weights->cbuffer().as<uint8_t*>() +
            k * singleKernelSize;
        auto transposedPart = transposeMatrix(ptr_filt_current, convolution.precision.size(), in_channels, kernelHW);
        transposedWeights.insert(transposedWeights.end(), transposedPart.begin(), transposedPart.end());
        transposedWeights.resize(transposedWeights.size() + kernelPad);
    }

    gnamem->getQueue(REGION_RO)->push_local_ptr(layer, ptr_weights,
        transposedWeights.data(),
        transposedWeights.size(),
        64);

    if (convolution._biases) {
        gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_biases,
            convolution._biases->cbuffer().as<const void*>(),
            convolution._biases->byteSize(),
            64);
    } else {
        gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, out_channels, 64);
    }
}

void GNAGraphCompiler::PowerPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& power = dynamic_cast<PowerLayer&>(*layer.get());
    if (power.power < 0.0f || power.power > 2.8f) {
        IE_THROW() << "[GNA plugin] unsupported power factor, expected be in <0, 2.8> range but was " << power.power;
    }

    auto input = layer->insData[0].lock();

    auto outputs = *layer->outData.begin();
    auto reshaped_dims = Get2DReshapedData(input, GNALimitations::GetMinBatchToFitInBuffer(input), 8)->getDims();
    const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
        GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;
    uint32_t num_rows_in = reshaped_dims[1];
    uint32_t num_columns_in = reshaped_dims[0];
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
        * outputs->getPrecision().size();

    size_t num_data_bytes_in = InferenceEngine::details::product(begin(input->getDims()), end(input->getDims()))
        * input->getPrecision().size();

    if (power.power == 1.0f) {
        void* ptr_inputs = nullptr;
        void* ptr_outputs = nullptr;
        void* ptr_weights = nullptr;
        void* ptr_biases = nullptr;

        auto& currentComponent = dnnComponents.addComponent(layer->name, "power");

        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
        IE_ASSERT(gnaFlags->sw_fp32 ? (quantized == nullptr) : (quantized != nullptr));
        dnn->InitAffineComponent(currentComponent,
            num_rows_in + num_padding,
            num_columns_in,
            num_rows_out + num_padding,
            input->getPrecision().size(),
            outputs->getPrecision().size(),
            // TODO: only fp32 and Int16 tested
            quantized == nullptr ? input->getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 2),
            quantized == nullptr ? input->getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 4),
            quantized == nullptr ? 1 : quantized->_weights_quant.GetScale(),
            quantized == nullptr ? 1 : quantized->_dst_quant.GetScale(),
            ptr_inputs,
            ptr_outputs,
            ptr_weights,
            ptr_biases,
            true);
        connectOutput(layer, ptr_outputs, num_data_bytes_out);
        connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);

        if (gnaFlags->sw_fp32) {
            IE_ASSERT(quantized == nullptr);
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_weights, power.scale, num_rows_out, 64);
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, power.offset, num_rows_out, 64);
        } else {
            IE_ASSERT(quantized != nullptr);
            if (!gnaFlags->input_low_precision) {
                auto quantizedScale = FLOAT_TO_INT16(std::min(quantized->_weights_quant.GetScale() * power.scale,
                    static_cast<float>(INT16_MAX)));
                auto quantizedOffset = FLOAT_TO_INT32(std::min(quantized->_dst_quant.GetScale() * power.offset,
                    static_cast<float>(INT32_MAX)));
                gnamem->getQueue(REGION_RO)->push_value<int16_t>(layer, ptr_weights, quantizedScale, num_rows_out, 64);
                gnamem->getQueue(REGION_RO)->push_value<int32_t>(layer, ptr_biases, quantizedOffset, num_rows_out, 64);
            } else {
                auto quantizedScale = FLOAT_TO_INT8(std::min(quantized->_weights_quant.GetScale() * power.scale,
                    static_cast<float>(INT8_MAX)));
                auto quantizedOffset = FLOAT_TO_INT8(std::min(quantized->_dst_quant.GetScale() * power.offset,
                    static_cast<float>(INT8_MAX)));
                gnamem->getQueue(REGION_RO)->push_value<int8_t>(layer, ptr_weights, quantizedScale, num_rows_out, 64);
                gnamem->getQueue(REGION_RO)->push_value<int8_t>(layer, ptr_biases, quantizedOffset, num_rows_out, 64);
            }
        }
    } else {
        //use PWL to calculate power
        std::vector<gna_pwl_segment_t> ptr_pwl_segments;

        auto orientation = kDnnInterleavedOrientation;

        auto activation_type = DnnActivation::fromType(kActPow);
        activation_type.fqParams.set = false;
        activation_type.srcFQParams.set = false;
        activation_type.args.pow.exponent = power.power;
        activation_type.args.pow.scale = power.scale;
        activation_type.args.pow.offset = power.offset;

        auto& pwlComponent = dnnComponents.addComponent(layer->name, "power");

        gna_pwl_segment_t* ptr_pwl_segments_target = nullptr;

        float output_pwl_scale_factor = getScaleFactor(layer, QuantizedDataType::output);
        float input_pwl_scale_factor = getScaleFactor(layer, QuantizedDataType::input);

        if (!gnaFlags->sw_fp32 && gnaFlags->uniformPwlDesign) {
            uint32_t num_segments = POW_NUM_SEGMENTS;
            if (activation_type.args.pow.exponent == 0.0f) {
                num_segments = 3;
            }
            ptr_pwl_segments.resize(num_segments);

            PwlDesign(activation_type,
                &*ptr_pwl_segments.begin(),
                static_cast<uint32_t>(ptr_pwl_segments.size()),
                input_pwl_scale_factor,
                output_pwl_scale_factor,
                gnaFlags->input_low_precision);
        }

        ptr_pwl_segments_target = reinterpret_cast<gna_pwl_segment_t*>(&ptr_pwl_segments_target);

        void* ptr_pwl_input = nullptr;
        void* ptr_pwl_outputs = nullptr;
        dnn->InitPiecewiseLinearComponent(pwlComponent,
            activation_type,
            orientation,
            num_rows_in + num_padding,
            num_columns_in,
            input->getPrecision().size(),
            outputs->getPrecision().size(),
            ptr_pwl_segments.size(),
            output_pwl_scale_factor,
            output_pwl_scale_factor,
            ptr_pwl_input,
            ptr_pwl_outputs,
            ptr_pwl_segments_target);
        connectOutput(layer, ptr_pwl_outputs, num_data_bytes_out);
        connectInput(layer, ptr_pwl_input, num_data_bytes_in, 0, 0);

        if (ptr_pwl_segments_target != nullptr) {
            gnamem->getQueue(REGION_RO)->push_local_ptr(layer, ptr_pwl_segments_target,
                &ptr_pwl_segments.front(),
                ptr_pwl_segments.size() * sizeof(gna_pwl_segment_t),
                64);
        }
    }
}

void GNAGraphCompiler::PoolingPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto& pooling = dynamic_cast<PoolingLayer&>(*layer.get());

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    printPoolingLayer(pooling);

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    uint32_t w_dim_in = GetDataDimSize(inputs, InferenceEngine::DataDimName::W);
    uint32_t h_dim_in = GetDataDimSize(inputs, InferenceEngine::DataDimName::H);
    const uint32_t c_dim_in = GetDataDimSize(inputs, InferenceEngine::DataDimName::C);

    uint32_t w_dim_out = GetDataDimSize(outputs, InferenceEngine::DataDimName::W);
    uint32_t h_dim_out = GetDataDimSize(outputs, InferenceEngine::DataDimName::H);
    const uint32_t c_dim_out = GetDataDimSize(outputs, InferenceEngine::DataDimName::C);

    if (w_dim_in == 1) {  // swap dimensions if needed to support swapped 1D case
        swap(h_dim_in, w_dim_in);
        swap(h_dim_out, w_dim_out);
        swap(pooling._kernel[X_AXIS], pooling._kernel[Y_AXIS]);
        swap(pooling._stride[X_AXIS], pooling._stride[Y_AXIS]);
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;

    bool is2DPooling = false;
    if (dnnComponents.components.size() > 0) {
        const auto last = dnnComponents.components.back();
        if (last.dnnComponent.operation == kDnnConvolutional2dOp) {
            is2DPooling = true;
        } else if (last.dnnComponent.operation == kDnnPiecewiselinearOp && dnnComponents.components.size() > 1) {
            const auto& prev2 = *std::prev(dnnComponents.components.cend(), 2);
            is2DPooling = prev2.dnnComponent.operation == kDnnConvolutional2dOp;
        }
    }

    if (is2DPooling) {
        ValidatePooling2D(layer->name, pooling._kernel_y, pooling._kernel_x, pooling._stride_y, pooling._stride_x);
    }

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
        { c_dim_in, h_dim_in, w_dim_in },
        { c_dim_out, h_dim_out, w_dim_out },
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        { pooling._kernel[X_AXIS], pooling._kernel[Y_AXIS] },
        { pooling._stride[X_AXIS], pooling._stride[Y_AXIS] },
        getScaleFactor(layer, QuantizedDataType::output),
        ptr_inputs,
        ptr_outputs);
    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()));

    // Need to reserve more memory otherwise the compiled model would not be
    // backward compatible with GNA 2.0
    // GNA 2.0 produces more outputs from 1D pooling than later GNA generations (including GNA 3.0)
    // When the model is compiled for some newer GNA generation (than GNA 2.0)
    // but it does not use any specific new GNA features it should be correct to import and run using previous GNA HW
    if (!is2DPooling) {
        const auto hLegacy =
            GNAPluginNS::GNAConvolutionLayer::outputFromPoolingLegacy(h_dim_in, pooling._stride[X_AXIS]);
        const auto wLegacy =
            GNAPluginNS::GNAConvolutionLayer::outputFromPoolingLegacy(w_dim_in, pooling._stride[Y_AXIS]);
        if (num_data_bytes_out < hLegacy * wLegacy * c_dim_out) {
            num_data_bytes_out = hLegacy * wLegacy * c_dim_out;
        }
    }

    num_data_bytes_out *= outputs->getPrecision().size();
    const auto hw_in = h_dim_in * w_dim_in;

    // TODO: Is this really needed?, find out why
    uint32_t num_padding = ALIGN(hw_in, 8) - hw_in;
    size_t num_data_bytes_in = c_dim_in * (hw_in + num_padding) * inputs->getPrecision().size();

    if (dnn->new_num_conv_columns) {
        uint32_t num_rows = 1;
        uint32_t num_columns = c_dim_in * w_dim_in + (ALIGN(c_dim_in * w_dim_in, 8) - c_dim_in * w_dim_in);
        if (dnn->new_num_conv_columns % num_columns == 0) {
            num_rows = dnn->new_num_conv_columns / num_columns;
        } else {
            num_columns = dnn->new_num_conv_columns;
        }
        dnn->new_num_conv_columns = 0;
        num_data_bytes_in = num_rows * num_columns * inputs->getPrecision().size();
    }

    auto fused_to_layer = connectInput(layer, ptr_inputs, num_data_bytes_in);
    // Pooling will be fused with the previous layer and we need to use it's order id
    layer->userValue.v_int = fused_to_layer.input->userValue.v_int;
    connectOutput(layer, ptr_outputs, num_data_bytes_out);
}

void GNAGraphCompiler::CopyPrimitive(InferenceEngine::CNNLayerPtr layer) {
    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();

    auto reshaped_dims = Get2DReshapedData(inputs, GNALimitations::GetMinBatchToFitInBuffer(inputs), 8)->getDims();
    uint32_t num_rows_in = reshaped_dims[1];
    uint32_t num_columns_in = reshaped_dims[0];
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_columns_out = num_columns_in;
    uint32_t num_padding_out = ALIGN(num_rows_out, 8) - num_rows_out;
    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    auto orientation = kDnnInterleavedOrientation;

    auto &currentComponent = dnnComponents.addComponent(layer->name, layer->type);

    dnn->InitCopyComponent(currentComponent,
        orientation,
        ALIGN(num_rows_in, 8),
        num_columns_in,
        ALIGN(num_rows_out, 8),
        num_columns_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        getScaleFactor(layer, QuantizedDataType::output),
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

    // Concat axis validation
    if (!GNALimitations::ValidateConvConcatAxis(concatLayer) && gnaFlags->log_level == ov::log::Level::WARNING) {
        std::ostringstream in_dims_oss;
        auto in_dims = concatLayer->insData[0].lock()->getDims();
        std::copy(in_dims.begin(), in_dims.end(), std::ostream_iterator<size_t>(in_dims_oss, ","));
        std::cout << "[ WARNING ] Topology with layer: " + layer->name + ", type: " + layer->type +
            ", and concatenation axis(" + std::to_string(concatLayer->_axis) +
            ") for input dimensions(" + in_dims_oss.str() + ") not supported\n";
    }

    auto& concatLayerInfo = concat_connection.find(concatLayer->name)->second;
    std::function<InferenceEngine::CNNLayerPtr(InferenceEngine::CNNLayerPtr)> find_cascaded_concat_recursively =
        [&find_cascaded_concat_recursively](InferenceEngine::CNNLayerPtr concat_candidate) {
        if (LayerInfo(concat_candidate).isConcat()) {
            return concat_candidate;
        }

        if (!LayerInfo(concat_candidate).isNonFunctional()) {
            return InferenceEngine::CNNLayerPtr(nullptr);
        }

        for (auto &&child_layer : getInputTo(concat_candidate->outData.front())) {
            auto child_concat = find_cascaded_concat_recursively(child_layer.second);
            if (child_concat) return child_concat;
        }

        return InferenceEngine::CNNLayerPtr(nullptr);
    };

    for (auto &&outLayer : getInputTo(concatLayer->outData.front())) {
        auto concatCandidate = find_cascaded_concat_recursively(outLayer.second);
        if (!concatCandidate) continue;
        gnalog() << "Cascaded concat connection found from: " << layer->name << ", to: " << concatCandidate->name << std::endl;
        connectOutput(layer, &concatLayerInfo.gna_ptr, concatLayerInfo.reserved_size);
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
            connectInput(layer, &concatLayerInfo.gna_ptr, inputLayer.tensorSize, inputLayer.offset, idx, false);
            concatLayerInfo.input_allocated = true;
        } else if (layerInfo.isMemory()) {
            connectInput(layer, &concatLayerInfo.gna_ptr, concatLayerInfo.reserved_size, inputLayer.offset, idx, false);
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

    size_t cropOffset, cropOutputSize;
    std::vector<int32_t> axis;
    std::tie(cropOffset, cropOutputSize, axis) = GetCropParams(cropLayer);
    size_t cropOffsetBytes = cropOffset * cropLayer->precision.size();
    size_t cropOutputSizeBytes = cropOutputSize * cropLayer->precision.size();

    if (!LayerInfo(cropLayer).isCropAffined()) {
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
        connectInput(layer, &cropLayerInfo->second.gna_ptr, cropOutputSizeBytes + cropOffsetBytes, cropOffsetBytes, 0);

        // cases for certain output layers
        for (auto&& outLayer : getInputTo(layer->outData.front())) {
            auto& nextLayer = outLayer.second;
            if (LayerInfo(nextLayer).isConcat()) {
                connectOutput(layer, &cropLayerInfo->second.gna_ptr, cropOutputSizeBytes);
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
        uint32_t num_rows_in = InferenceEngine::details::product(begin(inputs->getDims()), end(inputs->getDims()));
        uint32_t num_columns_in = 1;

        uint32_t num_rows_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()));
        const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
            GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;
        uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;

        void* ptr_inputs = nullptr;
        void* ptr_outputs = nullptr;
        void* ptr_weights = nullptr;
        void* ptr_biases = nullptr;

        auto& currentComponent = dnnComponents.addComponent(layer->name, "crop");

        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
        dnn->InitAffineComponent(currentComponent,
            num_rows_in + num_padding,
            num_columns_in,
            num_rows_out,
            inputs->getPrecision().size(),
            outputs->getPrecision().size(),
            quantized == nullptr ? inputs->getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 2),
            gnaFlags->input_low_precision ? 1 : 4,
            getScaleFactor(layer, QuantizedDataType::weights),
            getScaleFactor(layer, QuantizedDataType::output),
            ptr_inputs,
            ptr_outputs,
            ptr_weights,
            ptr_biases,
            false);
        size_t num_data_bytes_out =
            InferenceEngine::details::product(
                begin(outputs->getDims()), end(outputs->getDims())) * 4;

        size_t num_data_bytes_in = num_columns_in *
            ALIGN(num_rows_in, noOfInputsDivisor) * inputs->getPrecision().size();

        connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
        connectOutput(layer, ptr_outputs, num_data_bytes_out);

        FillWeightOfAligningFilter(layer, ptr_weights, cropOffset, (quantized == nullptr) ? false : true);

        (quantized == nullptr) ?
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, num_rows_out, 64) :
            gnamem->getQueue(REGION_RO)->push_value<int32_t>(layer, ptr_biases, 0, num_rows_out, 64);
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
    const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
        GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;

    // for eltwise sum/sub in 16-bit precision one input should be 4 bytes and one 2 bytes - detecting that below
    // the names of variables are left for clarity although not always reflecting the real precision/size
    auto inputs2Bytes = layer->insData[0].lock();
    auto inputs4Bytes = layer->insData[1].lock();
    auto nonFunctional = [](CNNLayerPtr ptr) {
        return LayerInfo(ptr).isNonFunctional();
    };
    auto inputFunc2Bytes = CNNNetPrevLayerSkipCertain(layer, 0, nonFunctional)->outData[0];
    auto inputFunc4Bytes = CNNNetPrevLayerSkipCertain(layer, 1, nonFunctional)->outData[0];

    int biasesLayerIdx = 1;

    if (quantized) {
        switch (eltwise._operation) {
        case InferenceEngine::EltwiseLayer::Sum:
        case InferenceEngine::EltwiseLayer::Sub:
        {
            if (gnaFlags->input_low_precision == false) {
                if (inputFunc4Bytes->getPrecision().size() != 4) {
                    std::swap(inputFunc4Bytes, inputFunc2Bytes);
                    std::swap(inputs4Bytes, inputs2Bytes);
                    biasesLayerIdx = 0;
                }
                GNA_LAYER_ASSERT(layer, inputFunc2Bytes->getPrecision().size() == 2);
                GNA_LAYER_ASSERT(layer, inputFunc4Bytes->getPrecision().size() == 4);
            } else {
                // for low precision both inputs should be 1 bytes in size
                GNA_LAYER_ASSERT(layer, inputFunc2Bytes->getPrecision().size() == 1);
                GNA_LAYER_ASSERT(layer, inputFunc4Bytes->getPrecision().size() == 1);
            }
            break;
        }
        case InferenceEngine::EltwiseLayer::Prod:
        {
            if (gnaFlags->input_low_precision == false) {
                // for mul both inputs should be 2 bytes precision
                GNA_LAYER_ASSERT(layer, inputFunc2Bytes->getPrecision().size() == 2);
                GNA_LAYER_ASSERT(layer, inputFunc4Bytes->getPrecision().size() == 2);
            } else {
                // for mul both inputs should be 1 byte precision
                GNA_LAYER_ASSERT(layer, inputFunc2Bytes->getPrecision().size() == 1);
                GNA_LAYER_ASSERT(layer, inputFunc4Bytes->getPrecision().size() == 1);
            }

            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Unsupported eltwise operation for quantization: " << eltwise._operation;
        }
    }

    auto outputs = *layer->outData.begin();

    auto in_4b_batch = GetDataDimSize(inputs4Bytes, InferenceEngine::DataDimName::N);
    auto in_4b_channels = GetDataDimSize(inputs4Bytes, InferenceEngine::DataDimName::C);
    auto in_4b_height = GetDataDimSize(inputs4Bytes, InferenceEngine::DataDimName::H);
    auto in_4b_width = GetDataDimSize(inputs4Bytes, InferenceEngine::DataDimName::W);
    auto in_4b_total_size = in_4b_batch * in_4b_channels * in_4b_height * in_4b_width;

    auto in_2b_batch = GetDataDimSize(inputs2Bytes, InferenceEngine::DataDimName::N);
    auto in_2b_channels = GetDataDimSize(inputs2Bytes, InferenceEngine::DataDimName::C);
    auto in_2b_height = GetDataDimSize(inputs2Bytes, InferenceEngine::DataDimName::H);
    auto in_2b_width = GetDataDimSize(inputs2Bytes, InferenceEngine::DataDimName::W);
    auto in_2b_total_size = in_2b_batch * in_2b_channels * in_2b_height * in_2b_width;

    if (in_2b_batch != in_4b_batch) {
        THROW_GNA_LAYER_EXCEPTION(layer) << " Inputs with different batch sizes " << in_2b_batch << " and "
                                         << in_4b_batch << " are not supported";
    }

    if (in_4b_total_size != in_2b_total_size) {
        THROW_GNA_LAYER_EXCEPTION(layer) << " Inputs size mismatch "
            << "(note: For Multiply, Add and Subtract layers, auto broadcasting is only supported for constant inputs) "
            << in_4b_total_size << " != " << in_2b_total_size;
    }

    // If batch size > 1 the data is reshaped to one with batch size = 1
    uint32_t num_rows_in = in_4b_total_size;
    uint32_t num_columns_in = 1;
    uint32_t num_rows_out = num_rows_in;
    uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;

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
        quantized == nullptr ? inputs2Bytes->getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 2),
        quantized == nullptr ? inputs4Bytes->getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 4),
        getScaleFactor(layer, QuantizedDataType::weights),
        getScaleFactor(layer, QuantizedDataType::output),
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
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_weights, -1.0f, num_rows_out, 64);
        } else {
            auto scaledIdentity = -quantized->_weights_quant.GetScale();

            if (gnaFlags->input_low_precision == false) {
                auto quantizedIdentity = FLOAT_TO_INT16(std::min(scaledIdentity, static_cast<float>(INT16_MAX)));
                gnamem->getQueue(REGION_RO)->push_value<int16_t>(layer, ptr_weights, quantizedIdentity, num_rows_out, 64);
            } else {
                auto quantizedIdentity = FLOAT_TO_INT8(std::min(scaledIdentity, static_cast<float>(INT8_MAX)));

                gnamem->getQueue(REGION_RO)->push_value<int8_t>(layer, ptr_weights, quantizedIdentity, num_rows_out, 64);
            }
        }
        connectInput(layer, ptr_biases, num_data_bytes_in, 0, biasesLayerIdx);
        break;
    case EltwiseLayer::Sum:
        if (quantized == nullptr) {
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_weights, 1.0f, num_rows_out, 64);
        } else {
            auto scaledIdentity = quantized->_weights_quant.GetScale();

            if (gnaFlags->input_low_precision == false) {
                auto quantizedIdentity = FLOAT_TO_INT16(std::min(scaledIdentity, static_cast<float>(INT16_MAX)));

                gnamem->getQueue(REGION_RO)->push_value<int16_t>(layer, ptr_weights, quantizedIdentity, num_rows_out, 64);
            } else {
                auto quantizedIdentity = FLOAT_TO_INT8(std::min(scaledIdentity, static_cast<float>(INT8_MAX)));

                gnamem->getQueue(REGION_RO)->push_value<int8_t>(layer, ptr_weights, quantizedIdentity, num_rows_out, 64);
            }
        }
        connectInput(layer, ptr_biases, num_data_bytes_in, 0, biasesLayerIdx);
        break;

    case EltwiseLayer::Prod:
        if (quantized == nullptr) {
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, num_rows_out, 64);
        } else {
            if (gnaFlags->input_low_precision == false) {
                gnamem->getQueue(REGION_RO)->push_value<int32_t>(layer, ptr_biases, 0, num_rows_out, 64);
            } else {
                gnamem->getQueue(REGION_RO)->push_value<int8_t>(layer, ptr_biases, 0, num_rows_out, 64);
            }
        }
        connectInput(layer, ptr_weights, num_data_bytes_in, 0, biasesLayerIdx);
        break;

    default:
        THROW_GNA_EXCEPTION << "Unsupported eltwise operation: " << eltwise._operation;
    }
}

void GNAGraphCompiler::GemmPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(layer->insData.size() == 2);
    auto input_1 = layer->insData[0].lock();
    auto input_2 = layer->insData[1].lock(); // the second input corresponds to ptr_weights in component
    auto outputs = *layer->outData.begin();
    auto inputPrecision = quantized ? Precision(Precision::I16) : input_1->getPrecision();
    uint32_t noOfInputsDivisor = GNALimitations::noOfInputsDivisor;

    auto in_dims = input_1->getDims();
    auto batch_size = (in_dims.size() == 1) ? 1 : in_dims.front();
    uint32_t num_rows_in = InferenceEngine::details::product(in_dims) / batch_size;
    uint32_t num_columns_in = batch_size;
    uint32_t num_rows_out = GetDataDimSize(outputs, 1);
    uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;
    uint32_t num_padding_out = 0;

    // Gemm gets two inputs
    void* ptr_input_1 = nullptr; // the first input
    void* ptr_outputs = nullptr;
    void* ptr_input_2 = nullptr; // the second input corresponds to ptr_weights in component
    void* ptr_biases = nullptr;

    auto& currentComponent = dnnComponents.addComponent(layer->name, ("affine"));

    dnn->InitAffineComponent(currentComponent,
                             num_rows_in + num_padding,
                             num_columns_in,
                             num_rows_out + num_padding_out,
                             inputPrecision.size(),
                             outputs->getPrecision().size(),
                             quantized == nullptr ? input_2->getPrecision().size() : 2,
                             quantized == nullptr ? input_2->getPrecision().size() : 4,
                             getScaleFactor(layer, QuantizedDataType::weights),
                             getScaleFactor(layer, QuantizedDataType::output),
                             ptr_input_1,
                             ptr_outputs,
                             ptr_input_2,
                             ptr_biases,
                             false);

    size_t num_data_bytes_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()))
                                * outputs->getPrecision().size();

    size_t num_data_bytes_in_1 = InferenceEngine::details::product(begin(input_1->getDims()), end(input_1->getDims()))
                               * input_1->getPrecision().size();
    size_t num_data_bytes_in_2 = InferenceEngine::details::product(begin(input_2->getDims()), end(input_2->getDims()))
                                 * input_2->getPrecision().size();

    connectOutput(layer, ptr_outputs, num_data_bytes_out);
    connectInput(layer, ptr_input_1, num_data_bytes_in_1);
    connectInput(layer, ptr_input_2, num_data_bytes_in_2, 0, 1);
    if (gnaFlags->sw_fp32) {
        IE_ASSERT(quantized == nullptr);
        gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, num_rows_out, 64);
    } else {
        gnamem->getQueue(REGION_RO)->push_value<int32_t>(layer, ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAGraphCompiler::AffinePrimitive(InferenceEngine::CNNLayerPtr layer, bool isDiag) {
    auto& weightable = dynamic_cast<WeightableLayer&>(*layer.get());
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

    IE_ASSERT(!layer->insData.empty());
    IE_ASSERT(!layer->outData.empty());
    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();
    Precision inputPrecision;
    uint32_t noOfInputsDivisor = GNALimitations::noOfInputsDivisor;

    if (!quantized) {
        inputPrecision = inputs->getPrecision();
    } else if (gnaFlags->input_low_precision == false) {
        inputPrecision = Precision(Precision::I16);
    } else {
        inputPrecision = Precision(Precision::I8);
        noOfInputsDivisor = GNALimitations::noOfInputsLowPrecDivisor;
    }

    auto input_data = HasTo2DReshapeData(layer) ?
        Get2DReshapedData(inputs, GNALimitations::GetMinBatchToFitInBuffer(inputs), 8) : inputs;
    auto in_dims = input_data->getDims();
    auto batch_size = (in_dims.size() == 1) ? 1 : in_dims.front();
    uint32_t num_rows_in = InferenceEngine::details::product(in_dims) / batch_size;
    uint32_t num_columns_in = batch_size;
    uint32_t num_rows_out = isDiag ? num_rows_in : GetDataDimSize(outputs, 1);
    uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;
    uint32_t num_padding_out = isDiag ? num_padding : 0;

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    // TODO: questionable why for biases that are not in IR we inventing precision
    auto biasPrecisionSize = weightable._biases ?
        weightable._biases->getTensorDesc().getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 4);

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
        biasPrecisionSize,
        getScaleFactor(layer, QuantizedDataType::weights),
        getScaleFactor(layer, QuantizedDataType::output),
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
            IE_THROW() << "[GNA plugin] Unsupported permute order: was " << layer->GetParamAsString("order") <<
                ", but only support 0, 3, 2, 1";
        }

        /**
         * TODO: weights transpose happened after quantisation might result in poor quality for in 8 - move this to passes
         */
        if (weightable._weights->getTensorDesc().getPrecision() == Precision::I8) {
            IE_THROW() << "[GNA plugin] Unsupported permute operation for 8 bit weights for layer: " << layer->name;
        }

        // this affine connected to convolution via pool or activation
        gnalog() << "Transposing weights for layer: " << layer->name << "\n";

        transpose = !isDiag;
        transposedRows = connectionInfo.permute->input()->getDims()[3];
        transposedCols = connectionInfo.permute->input()->getDims()[1];
    }

    auto wpSize = weightable.precision.size();
    const auto weightsBuffer = weightable._weights->cbuffer().as<const uint8_t*>();

    if (num_padding == 0) {
        if (!transpose) {
            gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_weights,
                weightable._weights->cbuffer().as<const void*>(),
                weightable._weights->byteSize(),
                64);
        } else {
            gnamem->getQueue(REGION_RO)->push_initializer(layer, ptr_weights, weightable._weights->byteSize(),
                [isDiag, num_rows_in, num_rows_out, num_padding, transposedRows, transposedCols, weightsBuffer, wpSize](void* data, size_t size) {
                for (uint32_t k = 0; k < (isDiag ? 1 : num_rows_out); k++) {
                    auto rowOffset = k * transposedRows * transposedCols * wpSize;
                    auto cbuffer = weightsBuffer + rowOffset;
                    auto u8Data = reinterpret_cast<uint8_t*>(data) + rowOffset;
                    for (int j = 0; j < transposedCols; j++) {
                        for (int i = 0; i < transposedRows; i++) {
                            auto offsetWrite = (transposedRows * j + i) * wpSize;
                            auto offsetRead = (i * transposedCols + j) * wpSize;
                            if (size < rowOffset + offsetWrite) {
                                // zero out dest if error detected
                                memset(data, 0, size);
                                THROW_GNA_EXCEPTION << "Size error";
                            }
                            ie_memcpy(u8Data + offsetWrite, size - rowOffset - offsetWrite,
                                cbuffer + offsetRead, wpSize);
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

        gnamem->getQueue(REGION_RO)->push_initializer(layer, ptr_weights, paddedWeightsSize,
            [isDiag, num_rows_in, num_rows_out, num_padding, weightsBuffer, wpSize](void* data, size_t size) {
            for (uint32_t i = 0; i < (isDiag ? 1 : num_rows_out); i++) {
                ie_memcpy(data, size,
                    weightsBuffer + num_rows_in * i * wpSize,
                    num_rows_in * wpSize);
                data = reinterpret_cast<uint8_t*>(data) + (num_rows_in + num_padding) * wpSize;
            }
            }, 64);
    }

    if (weightable._biases) {
        gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_biases,
            weightable._biases->cbuffer().as<const void*>(),
            weightable._biases->byteSize(),
            64);
    } else {
        // in that case input from previous layer goes into biases, so we have to initialize input pointer by zero
        if (useBiasConnection) {
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_inputs, 0.0f, num_rows_in + num_padding, 64);
        } else {
            gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, num_rows_out + num_padding_out, 64);
        }
    }
}

void GNAGraphCompiler::FillWeightOfAligningFilter(InferenceEngine::CNNLayerPtr layer, void* ptrWeights, size_t offset, bool isQuantized) {
    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    uint32_t num_rows_in = InferenceEngine::details::product(begin(inputs->getDims()), end(inputs->getDims()));
    uint32_t num_rows_out = InferenceEngine::details::product(begin(outputs->getDims()), end(outputs->getDims()));

    if (!ptrWeights) {
        THROW_GNA_EXCEPTION << "Weights memory is not allocated!!!";
    }

    gnamem->getQueue(REGION_RO)->push_initializer(layer,
                                                  ptrWeights,
                                                  num_rows_out * ALIGN(num_rows_in, 8) * layer->precision.size(),
                                                  [=](void* data, size_t size) {
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

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
        GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;
    uint32_t num_columns_in = GetDataDimSize(inputs, 2);
    uint32_t num_rows_out = GetDataDimSize(outputs, 1);
    uint32_t num_rows_in = filterLayer->_weights->size() / num_rows_out;
    uint32_t num_padding = ALIGN(num_rows_in, noOfInputsDivisor) - num_rows_in;

    auto numRowsPadded = filterLayer->GetParamAsInt("num_rows_padded");
    // number of rows we handled by inserting copy layer
    uint32_t num_rows_copied = 0;
    // in case of left alignment succeed, but due to number of elements not multiple of 8 we need to insert align_filter
    // we are improving it by inserting copy layer of size that covers most of elements - remained max of 32x31 affine filter
    if (0 == numRowsPadded && ALIGN(num_rows_in, 32) > 32) {
        // can we use copy at all
        num_rows_copied = ALIGN(num_rows_in, 32) - 32;

        auto orientation = kDnnInterleavedOrientation;

        auto& copyComponent = dnnComponents.addComponent(layer->name + "_synthetic_copy", CopyLayerName);

        dnn->InitCopyComponent(copyComponent,
                               orientation,
                               num_rows_copied,
                               num_columns_in,
                               num_rows_copied,
                               num_columns_in,
                               inputs->getPrecision().size(),
                               inputs->getPrecision().size(),
                               getScaleFactor(layer, QuantizedDataType::output),
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

    // TODO: questionable why for biases that are not in IR we inventing precision
    auto biasPrecisionSize = filterLayer->_biases ?
        filterLayer->_biases->getTensorDesc().getPrecision().size() : (gnaFlags->input_low_precision ? 1 : 4);
    auto& currentComponent = dnnComponents.addComponent(layer->name, "affine");

    dnn->InitAffineComponent(currentComponent,
        num_rows_in + num_padding,
        num_columns_in,
        num_rows_out,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        filterLayer->_weights->getTensorDesc().getPrecision().size(),
        biasPrecisionSize,
        getScaleFactor(layer, QuantizedDataType::weights),
        getScaleFactor(layer, QuantizedDataType::output),
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases,
        false);

    size_t num_data_bytes_out = num_rows_out * num_columns_in * outputs->getPrecision().size();
    size_t num_data_bytes_in = num_columns_in *
        ALIGN(num_rows_in, noOfInputsDivisor) * inputs->getPrecision().size();

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

        gnamem->getQueue(REGION_RO)->push_initializer(layer, ptr_weights, paddedWeightsSize, [=](void* data, size_t size) {
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
        gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_biases,
            filterLayer->_biases->cbuffer().as<const void*>(),
            filterLayer->_biases->byteSize(),
            64);
    } else {
        gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, num_rows_out, 64);
    }
}

void GNAGraphCompiler::ConvolutionFilterPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto filterLayer = dynamic_cast<InferenceEngine::ConvolutionLayer*> (layer.get());

    if (filterLayer == nullptr) {
        return;
    }

    auto prevLayer = CNNNetPrevLayer(layer.get(), 0);
    if (!LayerInfo(prevLayer).isSplit() && !LayerInfo(prevLayer).isSlice()) {
        THROW_GNA_EXCEPTION << "Case with Affine Aligning Filter for not Split/Slice layers is not implemented yet!";
    }

    void* ptr_inputs = nullptr;
    void* ptr_outputs = nullptr;
    void* ptr_weights = nullptr;
    void* ptr_biases = nullptr;

    IE_ASSERT(!layer->outData.empty());
    IE_ASSERT(!layer->insData.empty());
    auto outputs = *layer->outData.begin();
    auto inputs = layer->insData.begin()->lock();

    const auto noOfInputsDivisor = gnaFlags->input_low_precision ?
        GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;
    const uint32_t orginalInputSize =
        InferenceEngine::details::product(std::next(inputs->getDims().begin()), inputs->getDims().end());
    const uint32_t orginalOutputSize =
       InferenceEngine::details::product(std::next(outputs->getDims().begin()), outputs->getDims().end());
    if (orginalInputSize != orginalOutputSize) {
        THROW_GNA_LAYER_EXCEPTION(filterLayer) << "Number in inputs (" << orginalInputSize <<
            ") should be equal to number of outputs (" << orginalOutputSize << ")!";
    }
    const auto numberOfFilters = filterLayer->_out_depth;
    const auto convolutionStride = numberOfFilters;
    const auto filterWidth = filterLayer->_kernel_x;
    const auto minOutputsPerFilter = ALIGN(orginalOutputSize, numberOfFilters) / numberOfFilters;
    const auto minInputsNeeded = (minOutputsPerFilter - 1) * convolutionStride + filterWidth;
    const auto numInputsFullyPadedAndAligned = ALIGN(minInputsNeeded, noOfInputsDivisor);

    auto numOutputs = GNAConvolutionLayer::outputFromConv(numInputsFullyPadedAndAligned, filterWidth, convolutionStride);
    numOutputs *= numberOfFilters;
    const auto& biasPrecision = filterLayer->_biases ? filterLayer->_biases->getTensorDesc().getPrecision() : outputs->getPrecision();
    auto& currentComponent = dnnComponents.addComponent(layer->name, "affine");

    layer->params["num_rows_for_pwl"] = std::to_string(numOutputs);
    dnn->InitConvolutional1DComponent(currentComponent,
        numInputsFullyPadedAndAligned,
        numOutputs,
        inputs->getPrecision().size(),
        outputs->getPrecision().size(),
        filterLayer->_weights->getTensorDesc().getPrecision().size(),
        biasPrecision.size(),
        numberOfFilters,
        filterWidth,
        convolutionStride,
        getScaleFactor(layer, QuantizedDataType::weights),
        getScaleFactor(layer, QuantizedDataType::output),
        ptr_inputs,
        ptr_outputs,
        ptr_weights,
        ptr_biases);

    size_t num_data_bytes_out =
        InferenceEngine::details::product(
            begin(outputs->getDims()), end(outputs->getDims())) * 4;

    size_t num_data_bytes_in = numInputsFullyPadedAndAligned * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in, 0, 0);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_weights,
        filterLayer->_weights->cbuffer().as<const void*>(),
        filterLayer->_weights->byteSize(),
        64);

    if (filterLayer->_biases) {
        gnamem->getQueue(REGION_RO)->push_ptr(layer, ptr_biases,
            filterLayer->_biases->cbuffer().as<const void*>(),
            filterLayer->_biases->byteSize(),
            64);
    } else {
        gnamem->getQueue(REGION_RO)->push_value(layer, ptr_biases, 0.0f, numberOfFilters, 64);
    }
}

void GNAGraphCompiler::PWLPrimitive(InferenceEngine::CNNLayerPtr layer) {
    auto* generic = dynamic_cast<GenericLayer*>(layer.get());
    std::string type;
    std::vector<gna_pwl_segment_t> ptr_pwl_segments;
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

    GNA_LAYER_ASSERT(layer, !layer->insData.empty());
    GNA_LAYER_ASSERT(layer, !layer->outData.empty());

    auto inputs = layer->insData.begin()->lock();
    auto outputs = *layer->outData.begin();
    float output_pwl_scale_factor = getScaleFactor(layer, QuantizedDataType::output);
    float input_pwl_scale_factor = getScaleFactor(layer, QuantizedDataType::input);

    auto orientation = kDnnInterleavedOrientation;

    uint32_t w_dim_in = GetDataDimSize(inputs, DataDimName::W);
    uint32_t h_dim_in = GetDataDimSize(inputs, DataDimName::H);
    uint32_t c_dim_in = GetDataDimSize(inputs, DataDimName::C);
    uint32_t n_dim_in = GetDataDimSize(inputs, DataDimName::N);
    num_columns = n_dim_in;
    num_rows = w_dim_in * h_dim_in * c_dim_in;

    if (dnn->new_num_conv_columns) {
        if (dnn->new_num_conv_columns % num_columns == 0) {
            num_rows = dnn->new_num_conv_columns / num_columns;
        } else {
            num_columns = dnn->new_num_conv_columns;
            num_rows = 1;
        }
        dnn->new_num_conv_columns = 0;
    }

    // TODO: solve this by layer level transformations
    auto prevLayer = CNNNetPrevLayer(layer, 0);
    if (LayerInfo(prevLayer).isConcatAlignFilter()) {
        auto rowsCopiedOffset = prevLayer->GetParamAsInt("rows_copied_offset");
        if (rowsCopiedOffset != 0) {
            num_rows -= rowsCopiedOffset / outputs->getPrecision().size();
            layer->params["output_offset"] = std::to_string(rowsCopiedOffset);
        }
    } else if (LayerInfo(prevLayer).isConvolutionFilter()) {
        const auto num_rows_for_pwl = prevLayer->GetParamAsInt("num_rows_for_pwl", 0);
        if (num_rows_for_pwl != 0) {
            num_rows = num_rows_for_pwl;
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
        {"softsign", kActSoftSign},
        {"fakequantize", kActFakeQuantize},
        {"pwl", kActPwl}
    };

    auto it = supportedActivations.find(type);
    if (it == supportedActivations.end()) {
        THROW_GNA_EXCEPTION << "Activation function type not yet supported: " << type;
    }
    auto activation_type = DnnActivation::fromType(it->second);
    activation_type.fqParams.set = false;
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    if (quantized != nullptr && quantized->_dst_quant.IsStatsSet()) {
        activation_type.fqParams.set = true;
        activation_type.fqParams.levels = quantized->_dst_quant.GetLevels();
        activation_type.fqParams.inputPerChannel = false;
        activation_type.fqParams.input_low = &(quantized->_dst_quant.GetMinValues(true).front());
        activation_type.fqParams.input_high = &(quantized->_dst_quant.GetMaxValues(true).front());
    }

    activation_type.srcFQParams.set = false;
    if (quantized != nullptr && quantized->_src_quant.IsStatsSet()) {
        activation_type.srcFQParams.set = true;
        activation_type.srcFQParams.levels = quantized->_src_quant.GetLevels();
        activation_type.srcFQParams.inputPerChannel = false;
        activation_type.srcFQParams.input_low = &(quantized->_src_quant.GetMinValues(true).front());
        activation_type.srcFQParams.input_high = &(quantized->_src_quant.GetMaxValues(true).front());
    }

    if (it->second == kActRelu) {
        auto reluLayer = dynamic_cast<ReLULayer*>(layer.get());
        activation_type.args.lrelu.negative_slope = reluLayer != nullptr ? reluLayer->negative_slope : 0.0f;
    } else {
        activation_type.args.lrelu.negative_slope = 0.0f;
    }

    if (quantized == nullptr && it->second == kActFakeQuantize) {
        activation_type = GNAFakeQuantizeLayer(layer).parseAsActivation();
    } else if (it->second == kActKaldiLstmClipping) {
        auto clamp_layer = dynamic_cast<ClampLayer*>(layer.get());
        if (clamp_layer) {
            if (clamp_layer->min_value == 0 && clamp_layer->max_value == 0) {
                // Clamp layer may be not initialized due to backward compatibility
                // use in such case old default values
                activation_type.args.clamp.low = KALDI_LSTM_CLIP_LOWER;
                activation_type.args.clamp.high = KALDI_LSTM_CLIP_UPPER;
            } else {
                activation_type.args.clamp.low = clamp_layer->min_value;
                activation_type.args.clamp.high = clamp_layer->max_value;
            }
        } else {
            activation_type.args.clamp.low = KALDI_LSTM_CLIP_LOWER;
            activation_type.args.clamp.high = KALDI_LSTM_CLIP_UPPER;
        }
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
        GET_ACTIVATION_NAME(kActSign);
        GET_ACTIVATION_NAME(kActAbs);
        GET_ACTIVATION_NAME(kActNegLog);
        GET_ACTIVATION_NAME(kActNegHalfLog);
    default: break;
    }
#endif

    auto& currentComponent = dnnComponents.addComponent(layer->name, actName);
    gna_pwl_segment_t* ptr_pwl_segments_target = nullptr;

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
            PwlDesign(activation_type,
                &*ptr_pwl_segments.begin(),
                static_cast<uint32_t>(ptr_pwl_segments.size()),
                input_pwl_scale_factor,
                output_pwl_scale_factor,
                gnaFlags->input_low_precision);
        } else {
            PwlDesignOpt(activation_type,
                         input_pwl_scale_factor,
                         output_pwl_scale_factor,
                         gnaFlags->input_low_precision,
                         layer->getNode(),
                         CheckIFLastComponentIsPrecededByConv2D(dnnComponents.components),
                         ptr_pwl_segments);
        }
        ptr_pwl_segments_target = reinterpret_cast<gna_pwl_segment_t*>(&ptr_pwl_segments_target);
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

    auto fused_to_layer = connectInput(layer, ptr_inputs, num_data_bytes_in);
    // PWL will be fused with the previous layer and we need to use it's order id
    layer->userValue.v_int = fused_to_layer.input->userValue.v_int;
    connectOutput(layer, ptr_outputs, num_data_bytes_out);

    if (ptr_pwl_segments_target != nullptr) {
        gnamem->getQueue(REGION_RO)->push_local_ptr(layer, ptr_pwl_segments_target,
            &ptr_pwl_segments.front(),
            ptr_pwl_segments.size() * sizeof(gna_pwl_segment_t),
            64);
    }
}

void GNAGraphCompiler::PermutePrimitive(InferenceEngine::CNNLayerPtr layer) {
    if (LayerInfo(layer).isTrivialPermute()) {
        return;
    }
    auto layerOrder = layer->GetParamAsInts("order");
    if (layer->insData.empty()) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "Input layer pointer is unexpectedly absent";
    }
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

    const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
        GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;

    // now this can be run on GNA
    if (squeezedInputOrder[0] < squeezedInputOrder[1]) {  // interleave case
        if (ALIGN(squeezedInputOrder[1], noOfInputsDivisor) != squeezedInputOrder[1]) {
            THROW_GNA_LAYER_EXCEPTION(layer) << "unsupported permute (row size not a multiple of " << noOfInputsDivisor << ")";
        } else {
            auto& currentComponent = dnnComponents.addComponent(layer->name, "interleave");
            dnn->InitInterleaveComponent(currentComponent,
                                         squeezedInputOrder[0],
                                         squeezedInputOrder[1],
                                         inputs->getPrecision().size(),
                                         outputs->getPrecision().size(),
                                         getScaleFactor(layer, QuantizedDataType::output),
                                         ptr_inputs,
                                         ptr_outputs);
        }

    } else {  // deinterleave case
        if (ALIGN(squeezedInputOrder[0], noOfInputsDivisor) != squeezedInputOrder[0]) {
            THROW_GNA_LAYER_EXCEPTION(layer) << "[GNA plugin] unsupported permute (column size not a multiple of " << noOfInputsDivisor << ")";
        } else {
            auto& currentComponent = dnnComponents.addComponent(layer->name, "deinterleave");
            dnn->InitDeinterleaveComponent(currentComponent,
                                           squeezedInputOrder[0],
                                           squeezedInputOrder[1],
                                           inputs->getPrecision().size(),
                                           outputs->getPrecision().size(),
                                           getScaleFactor(layer, QuantizedDataType::output),
                                           ptr_inputs,
                                           ptr_outputs);
        }
    }

    size_t num_data_bytes_out = ALIGN(InferenceEngine::details::product(
            begin(outputs->getDims()), end(outputs->getDims())), noOfInputsDivisor)
                                * outputs->getPrecision().size();
    size_t num_data_bytes_in = squeezedInputOrder[0] * squeezedInputOrder[1] * inputs->getPrecision().size();

    connectInput(layer, ptr_inputs, num_data_bytes_in);
    connectOutput(layer, ptr_outputs, num_data_bytes_out);
}

inline void SKIP(GNAGraphCompiler*, CNNLayerPtr) {}

void GNAGraphCompiler::CreateLayerPrimitive(CNNLayerPtr layer) {
    static const LayersBuilder layersBuilder[] = {
        {{"Input"}, [](GNAGraphCompiler*, CNNLayerPtr l) {}},  // skip input layers they are not used in GNA lib, only as a memory blobs
        {{"FullyConnected", "InnerProduct"}, CREATE(AffinePrimitive)},
        {{"Gemm"}, CREATE(GemmPrimitive)},
        {{"ScaleShift"}, CREATE(DiagonalPrimitive)},
        {{"ConvolutionFilter"}, CREATE(ConvolutionFilterPrimitive)},
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
          "neghalflog",
          "pwl"},
          CREATE(PWLPrimitive)},
        {{"Convolution"}, CREATE(ConvolutionPrimitive)},
        {{"Permute"}, CREATE(PermutePrimitive)},  // permute of certain form (2D transpose) can be assimilated in followed FC layer
        {{"Pooling"}, CREATE(PoolingPrimitive)},
        {{"Power"} , CREATE(PowerPrimitive)},
        {{"Concat"}, CREATE(ConcatPrimitive)},
        {{"Reshape"}, SKIP},  // TODO: handled not in GNA but rather in GNA plugin
        {{"Squeeze"}, SKIP},  // TODO: handled not in GNA but rather in GNA plugin
        {{"Crop"}, CREATE(CropPrimitive)},
        {{CopyLayerName}, CREATE(CopyPrimitive)},
        {{DelayedCopyLayerName}, CREATE(CopyPrimitive)},
        {{"TensorIterator"}, SKIP},
        {{"LSTMCell"}, SKIP},
        {{"FakeQuantize"}, CREATE(PWLPrimitive)}
    };
    (void)layersBuilder;
    auto it = LayersBuilder::getStorage().find(layer->type);
    if (it != LayersBuilder::getStorage().end()) {
        it->second(this, layer);
    } else {
        THROW_GNA_EXCEPTION << "Unsupported layer: " << layer->name << ":" << layer->type;
    }
}

void GNAGraphCompiler::connectOutput(InferenceEngine::CNNLayerPtr layer,
                                    void *ptr,
                                    size_t num_data_bytes_out) {
    auto getOffsetForBinding = [](InferenceEngine::CNNLayerPtr layer) {
        int32_t output_offset = 0;
        if (layer->params.find("output_offset") != layer->params.end()) {
            output_offset = layer->GetParamAsInt("output_offset");
        }
        return output_offset;
    };

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
                    gnamem->getQueue(REGION_STATES)->reserve_ptr(nullptr, &nextMemoryLayer.gna_ptr, ALIGN64(memorySize), 64);
                    gnamem->getQueue(REGION_AUTO)->bind_ptr(nullptr, ptr, &nextMemoryLayer.gna_ptr, getOffsetForBinding(layer));
                    nextMemoryLayer.reserved_size = ALIGN64(memorySize);
                } else {
                    // We may need to extend memory buffer if connected input size is bigger, for example for concat connection
                    gnamem->getQueue(REGION_AUTO)->bind_ptr(nullptr, ptr, &nextMemoryLayer.gna_ptr, getOffsetForBinding(layer), ALIGN64(num_data_bytes_out));
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
                        auto outputSize = std::max(concatLayerInfoItem.reserved_size, num_data_bytes_out * 2);
                        gnamem->getQueue(REGION_SCRATCH)->reserve_ptr(layer, &concatLayerInfoItem.gna_ptr, ALIGN64(outputSize), 64);

                        std::function<void(GNAConcatLayer, GNAPluginNS::GnaInputs&, ConcatConnection&)> allocate_input_recursively =
                            [&allocate_input_recursively](GNAConcatLayer clayer,
                                                          GNAPluginNS::GnaInputs &inputs,
                                                          ConcatConnection& concat_connection) {
                            size_t concatInputIdx = 0;
                            for (auto &&inputLayer : clayer.concatInputLayers) {
                                // skipping non functional and reshape layer, as in that case input might be not connected to anything
                                auto realConcatInputs = CNNNetGetPrevLayersSkip(clayer.getConcat(), [](CNNLayerPtr l) {
                                    return !LayerInfo(l).isNonFunctional() && !LayerInfo(l).isSplit();
                                    }, concatInputIdx++);

                                for (auto rInput : realConcatInputs) {
                                    if (LayerInfo(rInput.first).isInput()) {
                                        inputs[rInput.first->name].allocated_size += inputLayer.tensorSize + inputLayer.offset;
                                    }
                                    if (LayerInfo(rInput.first).isConcat()) {
                                        auto concatLayerInfo = concat_connection.find(rInput.first->name);
                                        allocate_input_recursively(concatLayerInfo->second, inputs, concat_connection);
                                    }
                                }
                            }
                            clayer.input_allocated = true;
                        };

                        allocate_input_recursively(concatLayerInfoItem, *inputs_ptr_, concat_connection);
                    }
                    concatLayerInfo->second.output_allocation_flag = true;
                }
                // output offset precalculated to serve GNAAlignment requirements
                auto output_offset = it->offset;
                if (layer->params.find("output_offset") != layer->params.end()) {
                    output_offset = layer->GetParamAsInt("output_offset");
                }
                gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, ptr, &concatLayerInfoItem.gna_ptr, output_offset);
            }
            return;
        }
    }
    // real output should be allocated in separate region.
    auto mem_region = REGION_SCRATCH;

    auto nextLayer = CNNNetCheckNextLayerSkipCertain(layer, 0, 0, true,
        [](CNNLayerPtr l) { return LayerInfo(l).isNonFunctional(); }).first;
    // Check that layer will be an output
    if (LayerInfo(layer).isOutput() || !nextLayer) {
        mem_region = REGION_OUTPUTS;
    }
    if (LayerInfo(layer).isConst()) {
        mem_region = REGION_RO;
    }
    gnamem->getQueue(mem_region)->reserve_ptr(layer, ptr, ALIGN64(num_data_bytes_out), 64);
}

GNAPluginNS::ConnectionDetails GNAGraphCompiler::connectInput(CNNLayerPtr layer,
                                                              void *ptr,
                                                              size_t num_data_bytes_in,
                                                              int32_t offset,
                                                              int idx,
                                                              bool connectTo) {
    // selecting particular input layers
    // auto prevLayer = CNNNetPrevLayer(layer, idx);
    auto prevLayer = CNNNetPrevLayerSkipCertain(layer, idx, [](CNNLayerPtr l) {
        return LayerInfo(l).isNonFunctional();
    });
    if (!prevLayer) {
        THROW_GNA_EXCEPTION << "Input layer was not found";
    }

    gnalog() << "Connecting input " << layer->name << " to " << prevLayer->name << " ...\n";

    // real input not a memory input
    if (LayerInfo(prevLayer).isInput()) {
        auto quantized = getInjectedData<QuantizedLayerParams>(prevLayer);
        if (quantized) {
            if (quantized->lowPrecision) {
                inputs_ptr_->at(prevLayer->name).set_precision(InferenceEngine::Precision::I8);
            } else {
                inputs_ptr_->at(prevLayer->name).set_precision(InferenceEngine::Precision::I16);
            }
        }
        if (0 == inputs_ptr_->at(prevLayer->name).get_allocated_size()) {
            // if request for allocation less that realTensorInput - we need to extend request
            auto minInput = inputs_ptr_->at(prevLayer->name).get_required_size();
            if (num_data_bytes_in < minInput) {
                const uint32_t noOfInputsDivisor = gnaFlags->input_low_precision ?
                    GNALimitations::noOfInputsLowPrecDivisor : GNALimitations::noOfInputsDivisor;
                gnalog() << "[INPUT] : requested bytes: " << num_data_bytes_in << ", extended to" << ALIGN(minInput, noOfInputsDivisor);
                num_data_bytes_in = ALIGN(minInput, noOfInputsDivisor);
            }

            // real allocation pointer will be kept in ptr not in ptr_inputs_global
            if (!connectTo) {
                gnamem->getQueue(REGION_INPUTS)->push_value(layer, ptr,
                                   static_cast<uint8_t>(0),
                                   num_data_bytes_in,
                                   64);
            } else {
                gnamem->getQueue(REGION_INPUTS)->push_value(layer, &inputs_ptr_->at(prevLayer->name).ptrs.front(),
                                   static_cast<uint8_t>(0),
                                   num_data_bytes_in,
                                   64);
            }
            inputs_ptr_->at(prevLayer->name).allocated_size = num_data_bytes_in;
        }
        if (ALIGN(num_data_bytes_in, 64) > ALIGN(inputs_ptr_->at(prevLayer->name).get_allocated_size(), 64)) {
            THROW_GNA_EXCEPTION
                    << "Layer: " << layer->name
                    << " Cannot bind pointer to already allocated input(" << prevLayer->name
                    << "), due to size_allocated=" << inputs_ptr_->at(prevLayer->name).get_allocated_size()
                    << ", and size_requested=" << num_data_bytes_in;
        }

        if (connectTo) {
            gnamem->getQueue(REGION_AUTO)
                ->bind_ptr(layer, ptr, &inputs_ptr_->at(prevLayer->name).ptrs.front(), offset, ALIGN(num_data_bytes_in, 64));
        } else {
            gnamem->getQueue(REGION_AUTO)
                ->bind_ptr(layer, &inputs_ptr_->at(prevLayer->name).ptrs.front(), ptr, offset, ALIGN(num_data_bytes_in, 64));
        }

        return prevLayer;
    }
    // const input
    if (LayerInfo(prevLayer).isConst()) {
        if (connectTo) {
            gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, ptr, const_connections[prevLayer->name], offset);
        } else {
            gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, const_connections[prevLayer->name], ptr, offset);
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
                // splitting layer should take the execution order from the connected layer
                splittingLayer->userValue = layer->userValue;
                auto res = connectInput(splittingLayer, ptr, std::max(splitLayerInfoItem.reserved_size, num_data_bytes_in), it->offset + offset, 0);
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
            gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, ptr, &concatLayerInfoItem.gna_ptr, offset, num_data_bytes_in);
            // return layer over concat
            return CNNNetPrevLayer(prevLayer);
        }
    } else if (layerInfoObj.isCrop()) {
        auto cropLayerInfo = crop_connection.find(
                prevLayer->name);
        if (cropLayerInfo != crop_connection.end()) {
            auto & cropLayerInfoItem = cropLayerInfo->second;
            gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, ptr, &cropLayerInfoItem.gna_ptr, offset);
            return CNNNetPrevLayer(prevLayer);
        }
    }
    auto prevDnnLayer = dnnComponents.findComponent(prevLayer);

    // check for generic prev layer
    if (prevDnnLayer != nullptr) {
        gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, ptr, &prevDnnLayer->ptr_outputs, offset, num_data_bytes_in);
        return prevLayer;
    }

    auto prevMemoryLayer =
            std::find_if(begin(memory_connection), end(memory_connection), [&](MemoryConnection::value_type &comp) {
                return comp.second.getInput()->params.at("id") == prevLayer->params.at("id");
            });
    if (prevMemoryLayer != memory_connection.end()) {
        // dnnLayer that is input for memory output layer
        // TODO: this is duplicate with connect output
        auto& memoryLayer = prevMemoryLayer->second;
        if (memoryLayer.reserved_size == 0) {
            auto memorySize = InferenceEngine::details::product(memoryLayer.getDims()) * memoryLayer.elementSizeBytes();

            // connectTo used for  indicate that memory layer should be bound to given buffer
            if (connectTo) {
                memorySize = std::max(memorySize, num_data_bytes_in);
                gnamem->getQueue(REGION_STATES)->reserve_ptr(nullptr, &memoryLayer.gna_ptr, ALIGN64(memorySize), 64);
                gnamem->getQueue(REGION_AUTO)->bind_ptr(nullptr, ptr, &memoryLayer.gna_ptr, offset);
            } else {
                if (num_data_bytes_in < memorySize + offset) {
                    THROW_GNA_LAYER_EXCEPTION(layer) <<" invalid allocation request of "
                                                     << num_data_bytes_in << " is more then state tensor size of: " << memorySize + offset;
                }
                gnamem->getQueue(REGION_AUTO)->bind_ptr(nullptr, &memoryLayer.gna_ptr, ptr, offset, ALIGN64(num_data_bytes_in));
            }

            memoryLayer.reserved_size = ALIGN64(memorySize);
        } else {
            // We may need to extend memory buffer if connected input size is bigger, for example for concat connection
            gnamem->getQueue(REGION_AUTO)->bind_ptr(nullptr, ptr, &memoryLayer.gna_ptr, offset, ALIGN64(num_data_bytes_in));
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

void GNAGraphCompiler::printTensorDesc(const std::string& name, const InferenceEngine::TensorDesc& desc) {
    gnalog() << name << " layout: " << desc.getLayout() << " shape: ";
    for (auto i = 0; i < desc.getDims().size(); i++) {
        if (i > 0) {
            gnalog() << 'x';
        }
        gnalog() << desc.getDims()[i];
    }
    gnalog() << "\n";
}

void GNAGraphCompiler::printConvolutionLayer(const InferenceEngine::ConvolutionLayer& layer) {
    const char x = 'x';

    gnalog() << "ConvolutionLayer '"
        << layer.name
        << "' Kernel: "
        << layer._kernel_x << x << layer._kernel_y
        << " Padding: "
        << layer._padding_x << x << layer._padding_y
        << " Stride: "
        << layer._stride_x << x << layer._stride_y
        << " Dilation: "
        << layer._dilation_x << x << layer._dilation_y
        << " Auto Padding: '"
        << layer._auto_pad << "'";
    gnalog() << "\n";
    printTensorDesc("Input", layer.input()->getTensorDesc());
    printTensorDesc("Output", layer.outData.front()->getTensorDesc());
}

void GNAGraphCompiler::printPoolingLayer(const InferenceEngine::PoolingLayer& layer) {
    const char x = 'x';

    gnalog() << "PoolingLayer '"
        << layer.name
        << "' Kernel: "
        << layer._kernel_x << x << layer._kernel_y
        << " Padding: "
        << layer._padding_x << x << layer._padding_y
        << " Stride: "
        << layer._stride_x << x << layer._stride_y
        << " Auto Padding: '"
        << layer._auto_pad << "'";
    gnalog() << "\n";
    printTensorDesc("Input", layer.input()->getTensorDesc());
    printTensorDesc("Output", layer.outData.front()->getTensorDesc());
}

std::vector<uint8_t>
GNAGraphCompiler::transposeMatrix(uint8_t* ptr_matrix, size_t element_size, uint32_t num_rows, uint32_t num_cols) {
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
}
