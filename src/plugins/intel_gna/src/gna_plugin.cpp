// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX

#include "gna_plugin.hpp"

#include <debug.h>
#include <gna2-common-api.h>
#include <gna2-model-api.h>
#include <ie_common.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <gna/gna_config.hpp>
#include <layers/gna_fake_quantize_layer.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/graph_tools.hpp>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/am_intel_dnn.hpp"
#include "common/gna_target.hpp"
#include "frontend/model_quantizer.hpp"
#include "frontend/scale_factor_calc.hpp"
#include "frontend/weights_converter.hpp"
#include "gna2-model-api.h"
#include "gna2_model_export_helper.hpp"
#include "gna2_model_helper.hpp"
#include "gna_fused_iterator.hpp"
#include "gna_graph_patterns.hpp"
#include "gna_itt.hpp"
#include "gna_model_serial.hpp"
#include "gna_plugin_config.hpp"
#include "gna_tensor_tools.hpp"
#include "gna_transformations_pipeline.hpp"
#include "layers/gna_layer_type.hpp"
#include "log/log.hpp"
#include "memory/gna_memory_state.hpp"
#include "orientation_helper.hpp"
#include "preprocessing.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/worker_factory.hpp"
#include "request/worker_pool_impl.hpp"
#include "runtime/gna_float_runtime.hpp"
#include "scale_factor_helper.hpp"

using namespace ov::intel_gna::ngraph_util;

inline uint32_t ToByteSize(const Gna2DataType type) {
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

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

using namespace ov::intel_gna::memory;
using namespace ov::intel_gna::frontend;

namespace InferenceEngine {
template <>
InferenceEngine::TBlob<gna_compound_bias_t, std::enable_if<true, void>>::~TBlob() {
    free();
}
}  // namespace InferenceEngine

template <typename T, typename U>
void GNAPlugin::copyInputData(T* dst,
                              const U* src,
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
                    if (!gnaFlags->input_low_precision) {
                        dst[j * num_group + i] = ConvertFloatToInt16(src[i * num_vector_elements + j] * scaleFactor);
                    } else {
                        dst[j * num_group + i] = ConvertFloatToInt8(src[i * num_vector_elements + j] * scaleFactor);
                    }
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
                T* ptr_dst_vec = reinterpret_cast<T*>(dst) + i * num_vector_stride;
                const U* ptr_src_vec = reinterpret_cast<const U*>(src) + i * num_vector_elements;
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                if (!gnaFlags->input_low_precision) {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ConvertFloatToInt16(ptr_src_vec[j] * scaleFactor);
                    }
                } else {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ConvertFloatToInt8(ptr_src_vec[j] * scaleFactor);
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < num_frames; i++) {
                void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
                const void* ptr_src_vec = reinterpret_cast<const uint8_t*>(src) + i * num_vector_elements * sizeof(U);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                ie_memcpy(ptr_dst_vec, num_vector_elements * sizeof(T), ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (uint32_t i = num_frames; i < num_group; i++) {
            void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

void GNAPlugin::ExportScores(void* ptr_dst,
                             const void* ptr_src,
                             intel_dnn_orientation_t orientation,
                             uint32_t num_frames,
                             uint32_t num_group,
                             uint32_t num_vector_elements,
                             uint32_t num_active_elements,
                             uint32_t num_vector_stride,
                             Precision precision_in,
                             Precision precision_out) {
    if (ptr_src == nullptr || ptr_dst == nullptr) {
        THROW_GNA_EXCEPTION << "Received null pointer arguments";
    }
    if (precision_out != Precision::I32 && precision_out != Precision::FP32) {
        THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << precision_out.name();
    }
    // source scores are possibly padded to multiple of 8 and possibly interleaved
    // rotate if necessary and only copy actual scores (not padding)
    if (orientation == kDnnInterleavedOrientation) {
        int32_t* dst = reinterpret_cast<int32_t*>(ptr_dst);
        const int8_t* src = reinterpret_cast<const int8_t*>(ptr_src);
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_active_elements; j++) {
                auto input_ptr = src + (j * num_group + i) * precision_in.size();
                auto dst_ptr = dst + (i * num_vector_elements + j);

                switch (precision_in) {
                case Precision::I8: {
                    *dst_ptr = static_cast<int32_t>(*reinterpret_cast<const int8_t*>(input_ptr));
                    break;
                }
                case Precision::I16: {
                    *dst_ptr = static_cast<int32_t>(*reinterpret_cast<const int16_t*>(input_ptr));
                    break;
                }
                case Precision::I32: {
                    *dst_ptr = *reinterpret_cast<const int32_t*>(input_ptr);
                    break;
                }
                default:
                    THROW_GNA_EXCEPTION << "Unsupported output layer precision: " << precision_in.name();
                }
            }
            for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                dst[i * num_vector_elements + j] = 0;
            }
        }
    } else {
        switch (precision_in) {
        case Precision::I8:
        case Precision::I32: {
            for (uint32_t i = 0; i < num_frames; i++) {
                void* ptr_dst_vec =
                    reinterpret_cast<uint8_t*>(ptr_dst) + i * num_vector_elements * precision_out.size();
                const void* ptr_src_vec =
                    reinterpret_cast<const uint8_t*>(ptr_src) + i * num_vector_stride * precision_in.size();
                memset(ptr_dst_vec, 0, num_vector_elements * precision_out.size());
                ie_memcpy(ptr_dst_vec,
                          num_active_elements * precision_out.size(),
                          ptr_src_vec,
                          num_active_elements * precision_in.size());
            }
            break;
        }
        case Precision::I16: {
            for (uint32_t i = 0; i < num_frames; i++) {
                auto ptr_dst_vec = reinterpret_cast<int32_t*>(ptr_dst) + i * num_vector_elements;
                auto ptr_src_vec = reinterpret_cast<const int16_t*>(ptr_src) + i * num_vector_stride;
                for (uint32_t j = 0; j < num_vector_elements; j++) {
                    ptr_dst_vec[j] = ptr_src_vec[j];
                }
            }
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Unsupported output layer precision: " << precision_in.name();
        }
    }
}

void GNAPlugin::ImportFrames(void* ptr_dst,
                             const void* ptr_src,
                             Precision input_precision,
                             float scaleFactor,
                             intel_dnn_orientation_t orientation,
                             uint32_t num_frames,
                             uint32_t num_group,
                             uint32_t num_vector_elements,
                             uint32_t num_vector_stride) {
    switch (input_precision) {
    case Precision::U8:
    case Precision::I8: {
        auto src = reinterpret_cast<const uint8_t*>(ptr_src);
        if (!gnaFlags->input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copyInputData(dst,
                          src,
                          num_frames,
                          num_group,
                          num_vector_elements,
                          num_vector_stride,
                          orientation,
                          scaleFactor);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copyInputData(dst,
                          src,
                          num_frames,
                          num_group,
                          num_vector_elements,
                          num_vector_stride,
                          orientation,
                          scaleFactor);
        }
        break;
    }
    case Precision::I16: {
        auto src = reinterpret_cast<const int16_t*>(ptr_src);
        if (!gnaFlags->input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copyInputData(dst,
                          src,
                          num_frames,
                          num_group,
                          num_vector_elements,
                          num_vector_stride,
                          orientation,
                          scaleFactor);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copyInputData(dst,
                          src,
                          num_frames,
                          num_group,
                          num_vector_elements,
                          num_vector_stride,
                          orientation,
                          scaleFactor);
        }
        break;
    }
    case Precision::FP32:
    case Precision::I32: {
        auto src = reinterpret_cast<const float*>(ptr_src);
        if (!gnadevice) {
            auto dst = reinterpret_cast<float*>(ptr_dst);
            copyInputData(dst,
                          src,
                          num_frames,
                          num_group,
                          num_vector_elements,
                          num_vector_stride,
                          orientation,
                          scaleFactor);
        } else {
            if (!gnaFlags->input_low_precision) {
                auto dst = reinterpret_cast<int16_t*>(ptr_dst);
                copyInputData(dst,
                              src,
                              num_frames,
                              num_group,
                              num_vector_elements,
                              num_vector_stride,
                              orientation,
                              scaleFactor);
            } else {
                auto dst = reinterpret_cast<int8_t*>(ptr_dst);
                copyInputData(dst,
                              src,
                              num_frames,
                              num_group,
                              num_vector_elements,
                              num_vector_stride,
                              orientation,
                              scaleFactor);
            }
        }
        break;
    }
    default:
        break;
    }
}

GNAPlugin::GNAPlugin() : graphCompiler(config) {
    Init();
    UpdateFieldsFromConfig();
    InitGNADevice();
}

GNAPlugin::GNAPlugin(const std::map<std::string, std::string>& configMap) : graphCompiler(config) {
    Init();
    SetConfig(configMap);
    log::set_log_level(gnaFlags->log_level);
    InitGNADevice();
}

void GNAPlugin::Init() {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "Init");
    dnn = std::make_shared<backend::AMIntelDNN>(backend::AMIntelDNN());
    gnaFlags = std::make_shared<GNAFlags>(GNAFlags());
    inputs_ptr_ = std::make_shared<GnaInputs>(GnaInputs());
    outputs_ = GnaOutputs();

    graphCompiler.setDNNPtr(dnn);
    graphCompiler.setInputsPtr(inputs_ptr_);

    requestWorkerPool_ = std::make_shared<request::WorkerPoolImpl>();
}

void GNAPlugin::InitGNADevice() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InitGNADevice");
    if (gnaFlags->sw_fp32) {
        gnamem.reset(new gna_memory_float(memory::GNAFloatAllocator{}));
    } else {
        gnadevice = std::make_shared<GNADeviceHelper>(config.target,
                                                      gnaFlags->performance_counting,
                                                      !config.embedded_export_path.empty());
        size_t page_size_bytes = 4096;
        gnamem = std::make_shared<gna_memory_device>(memory::GNAAllocator(gnadevice), page_size_bytes);
    }
    graphCompiler.setGNAMemoryPtr(gnamem);
}

void GNAPlugin::UpdateInputScaleFromNetwork(InferenceEngine::CNNNetwork& network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputScaleFromNetwork");
    // fp32 emulation mode dont need any modifications to configuration
    if (config.gnaFlags.sw_fp32)
        return;

    // search for FQ layers
    // only supports cases of int16 or int8
    InputsDataMap inputs = network.getInputsInfo();
    size_t inputIdx = 0;
    for (auto&& input : inputs) {
        auto data = input.second->getInputData();
        for (auto&& nextToInputLayer : getInputTo(data)) {
            if (!LayerInfo(nextToInputLayer.second).isFakeQuantize()) {
                continue;
            }

            // replacing scale factor from this fq layer
            GNAFakeQuantizeLayer fqLayer(nextToInputLayer.second);
            auto inputRange = fqLayer.getInputRange();
            auto outputRange = fqLayer.getOutputRange();
            if (inputRange.second.size() != 1 || inputRange.second.size() != 1 || outputRange.second.size() != 1 ||
                outputRange.second.size() != 1) {
                THROW_GNA_LAYER_EXCEPTION(nextToInputLayer.second)
                    << "unsupported, per-channel quantization for input layer : " << input.second->name();
            }

            // GNA input is always quantized to int16, so number of levels can't be greater than max uint16
            // todo: should be solved in POT (issue 63330)
            size_t levels =
                std::min(fqLayer.getLevels(), static_cast<size_t>(std::numeric_limits<uint16_t>::max() + 1));
            auto scaleInput = ov::intel_gna::frontend::CalculateScaleFactorFromStats(levels,
                                                                                     inputRange.first[0],
                                                                                     inputRange.second[0]);

            if (!config.inputScaleFactorsPerInput.empty() || !config.inputScaleFactors.empty()) {
                log::warning() << "Scale factor calculated during model quantization (" << scaleInput
                               << ") will be used instead of user input (" << (*inputs_ptr_)[input.first].scale_factor
                               << ").\n";
                if ((*inputs_ptr_)[input.first].scale_factor < scaleInput) {
                    log::warning() << "Scale factor calculated based on input values ("
                                   << (*inputs_ptr_)[input.first].scale_factor
                                   << ") is smaller than scale factor used to quantize model (" << scaleInput << "). "
                                   << "Input values will be clamped.\n";
                }
            }
            config.inputScaleFactorsPerInput[input.first] = scaleInput;
            (*inputs_ptr_)[input.first].scale_factor = scaleInput;
        }

        inputIdx++;
    }
}

void GNAPlugin::UpdateInputsAndOutputsInfoFromNetwork(InferenceEngine::CNNNetwork& network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputsAndOutputsInfoFromNetwork");

    // update inputs
    {
        InputsDataMap network_inputs = network.getInputsInfo();
        for (const auto& input : network_inputs) {
            (*inputs_ptr_)[input.first].Update(input.second);
        }
    }

    // update outputs
    {
        OutputsDataMap outputs = network.getOutputsInfo();
        for (const auto& output : outputs) {
            outputs_[output.first].Update(output.second);
        }
    }
}

void GNAPlugin::UpdateInputs(const std::vector<std::shared_ptr<const ov::Node>>& params) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputs");
    for (const auto& param : params) {
        const std::string ie_name = param->get_friendly_name();
        (*inputs_ptr_)[ie_name].name = param->get_friendly_name();
        (*inputs_ptr_)[ie_name].tensor_names = param->get_output_tensor(0).get_names();
    }
}

void GNAPlugin::UpdateOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateOutputs");
    for (const auto& result : results) {
        const std::string ie_name = ov::op::util::create_ie_output_name(result->input_value(0));
        outputs_[ie_name].name = ie_name;
        outputs_[ie_name].tensor_names = result->get_output_tensor(0).get_names();
    }
}

void GNAPlugin::UpdateInputsAndOutputsInfoFromModel(std::shared_ptr<const ov::Model> model) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputsAndOutputsInfoFromFModel");

    // update inputs
    {
        std::vector<std::shared_ptr<const ov::Node>> node_vector;
        for (auto& param : model->get_parameters()) {
            node_vector.emplace_back(param);
        }
        UpdateInputs(node_vector);
    }

    // update outputs
    {
        std::vector<std::shared_ptr<const ov::Node>> node_vector;
        for (auto& result : model->get_results()) {
            node_vector.emplace_back(result);
        }
        UpdateOutputs(node_vector);
    }
}

bool GNAPlugin::TryToInitOutput(const std::string& portName, InferenceEngine::CNNLayerPtr layer) {
    auto initOutput = [this, portName, layer](intel_dnn_orientation_t orientation,
                                              size_t numBytesPerElem,
                                              size_t numElem,
                                              void* outputPtr) {
        auto quantized = InferenceEngine::getInjectedData<ov::intel_gna::frontend::QuantizedLayerParams>(layer);

        outputs_.at(portName).ptrs.resize(gnaFlags->num_requests);
        outputs_.at(portName).orientation = orientation;
        outputs_.at(portName).set_precision(numBytesPerElem);
        outputs_.at(portName).scale_factor =
            quantized != nullptr ? quantized->_dst_quant.GetScale() : kScaleFactorDefault;
        outputs_.at(portName).num_elements = numElem;

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, &outputs_.at(portName).ptrs.front(), outputPtr);
    };

    // probing gna_primitives
    auto irLayerAvatar = std::find_if(graphCompiler.dnnComponents.components.begin(),
                                      graphCompiler.dnnComponents.components.end(),
                                      [&layer](const backend::DnnComponents::storage_type::value_type& value) {
                                          return value.name == layer->name;
                                      });
    if (irLayerAvatar != graphCompiler.dnnComponents.components.end()) {
        initOutput(irLayerAvatar->dnnComponent.orientation_out,
                   irLayerAvatar->dnnComponent.num_bytes_per_output,
                   irLayerAvatar->dnnComponent.num_rows_out,
                   &irLayerAvatar->dnnComponent.ptr_outputs);
        return true;
    }

    // probing concatInfo
    if (LayerInfo(layer).isConcat()) {
        auto concatConnection = graphCompiler.concat_connection.find(layer->name);
        if (concatConnection != graphCompiler.concat_connection.end()) {
            auto precision = layer->outData.front()->getPrecision().size();
            initOutput(kDnnInterleavedOrientation,
                       precision,
                       concatConnection->second.reserved_size / precision,
                       &concatConnection->second.gna_ptr);
            return true;
        }
    }

    // probing a constant info, for constant trivial networks support
    if (LayerInfo(layer).isConst()) {
        auto const_blob = layer->blobs["custom"];
        auto constConnection = graphCompiler.const_connections.find(layer->name);
        if (constConnection != graphCompiler.const_connections.end()) {
            initOutput(kDnnInterleavedOrientation,
                       layer->outData.front()->getPrecision().size(),
                       const_blob->size(),
                       &constConnection->second);
            return true;
        }
    }

    return false;
}

void GNAPlugin::FillInputsAndOutputsTranspositionInfo(const InferenceEngine::CNNNetwork& net) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "FillInputsAndOutputsTranspositionInfo");
    auto printTranspositionInfo = [](const std::vector<TranspositionInfo>& transpositionInfo) {
        for (const auto& transpositionInfoPart : transpositionInfo) {
            log::debug() << "transpose=" << transpositionInfoPart.transpose
                         << " rows_num=" << transpositionInfoPart.num_transpose_rows
                         << " columns_num=" << transpositionInfoPart.num_transpose_columns << "\n";
        }
    };

    auto inputLayers = CNNNetGetAllInputLayers(net);
    for (const auto& inputLayer : inputLayers) {
        // Collect information for inputs transposition
        if (!LayerInfo(inputLayer).isInput())
            continue;
        auto transpositionInfo = FindTranspositionInfoFromNextLayers(inputLayer);
        if (transpositionInfo.empty())
            continue;

        transpose_inputs_info.insert({inputLayer->name, transpositionInfo});
        log::debug() << "Input " << inputLayer->name << " transposition info: \n";
        printTranspositionInfo(transpositionInfo);
    }

    auto outputsMap = net.getOutputsInfo();
    for (const auto& outPort : outputsMap) {
        auto outLayer = getCreatorLayer(outPort.second).lock();
        // Collect information for outputs transposition
        if (!LayerInfo(outLayer).isOutput())
            continue;
        auto transpositionInfo = FindTranspositionInfoFromPrevLayers(outLayer);
        if (transpositionInfo.empty())
            continue;

        // Swap transposition info rows and columns since we need to transpose output back from NHWC to NCHW
        for (auto&& transpositionInfoPart : transpositionInfo) {
            if (transpositionInfoPart.transpose) {
                std::swap(transpositionInfoPart.num_transpose_rows, transpositionInfoPart.num_transpose_columns);
            }
        }
        transpose_outputs_info.insert({outLayer->name, transpositionInfo});
        log::debug() << "Output " << outLayer->name << " transposition info: \n";
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
        printed_properties.emplace_back("weights scale factor", std::to_string(quantized->_weights_quant.GetScale()));
        if (quantized->_weights_quant.IsStatsSet()) {
            for (auto& min : quantized->_weights_quant.GetMinValues()) {
                printed_properties.emplace_back("weights min val", std::to_string(min));
            }
            for (auto& max : quantized->_weights_quant.GetMaxValues()) {
                printed_properties.emplace_back("weights max val", std::to_string(max));
            }
        }

        if (quantized->_bias_quant.IsStatsSet()) {
            for (auto& min : quantized->_bias_quant.GetMinValues()) {
                printed_properties.emplace_back("bias min val", std::to_string(min));
            }
            for (auto& max : quantized->_bias_quant.GetMaxValues()) {
                printed_properties.emplace_back("bias max val", std::to_string(max));
            }
        }
    }
    printed_properties.emplace_back("src scale factor", std::to_string(quantized->_src_quant.GetScale()));
    if (quantized->_src_quant.IsStatsSet()) {
        for (auto& min : quantized->_src_quant.GetMinValues()) {
            printed_properties.emplace_back("src min val", std::to_string(min));
        }
        for (auto& max : quantized->_src_quant.GetMaxValues()) {
            printed_properties.emplace_back("src max val", std::to_string(max));
        }
    }

    printed_properties.emplace_back("dst scale factor", std::to_string(quantized->_dst_quant.GetScale()));
    if (quantized->_dst_quant.IsStatsSet()) {
        for (auto& min : quantized->_dst_quant.GetMinValues()) {
            printed_properties.emplace_back("dst min val", std::to_string(min));
        }
        for (auto& max : quantized->_dst_quant.GetMaxValues()) {
            printed_properties.emplace_back("dst max val", std::to_string(max));
        }
    }
}
#endif

void GNAPlugin::LoadNetwork(const CNNNetwork& _network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "LoadNetwork");
    _network_name = _network.getName();
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork;

    const auto effectiveCompileTarget = config.target->get_effective_compile_target();
    graphCompiler.SetValidatorTarget(effectiveCompileTarget);

    auto transformer = TransformationsPipeline(config);

    if (_network.getFunction()) {
        CNNNetwork clonedNetwork = InferenceEngine::cloneNetwork(_network);
        auto model = clonedNetwork.getFunction();
        transformer.apply(model);
        limitations::check_all_ops_supported(model, effectiveCompileTarget, config.gnaPrecision);
        convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(model, clonedNetwork);
    }
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetwork network = convertedNetwork ? InferenceEngine::CNNNetwork{convertedNetwork} : _network;
    IE_SUPPRESS_DEPRECATED_END

    transformer.convert_precision_legacy(network);

    //  Check the network
    std::string error;
    if (!limitations::AreLayersSupported(network, error)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // Set input and output information from ngraph function
    if (_network.getFunction()) {
        UpdateInputsAndOutputsInfoFromModel(_network.getFunction());
    }

    // Set input and output information from orginal network
    UpdateInputsAndOutputsInfoFromNetwork(network);

    // DEPRECATED: To be removed after fully switching to POT optimized models.
    // Set Scale Factors for inputs according to configuration.
    ov::intel_gna::helpers::ApplyInputScaleFactors(*inputs_ptr_, config);

    if (transformer.is_fake_quantized()) {
        UpdateInputScaleFromNetwork(network);
    }

    if (MustBeConvertedFromNCHWToNHWC(CNNNetSortTopologically(network))) {
        FillInputsAndOutputsTranspositionInfo(network);
    }

    InferenceEngine::CNNNetwork newNet;

    if (gnaFlags->sw_fp32) {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            ov::intel_gna::frontend::convert_blobs_precision(*lp);
            return lp;
        };
        newNet = InferenceEngine::CNNNetCopy(network, visitor);
        // to run all passes need to have two calls to pass manager
        transformer.apply_legacy(newNet, true);
        transformer.apply_legacy(newNet, false);
    } else {
        ov::intel_gna::frontend::ModelQuantizer modelQuantizer(transformer);
        newNet = modelQuantizer.quantize(network, *inputs_ptr_);
    }

    auto inputLayers = CNNNetGetAllInputLayers(newNet);

#ifdef PLOT
    std::ofstream file("gna_passes.dot");
    saveGraphToDot(
        newNet,
        file,
        [this](const CNNLayerPtr layer, ordered_properties& printed_properties, ordered_properties& node_properties) {
            AddDebugProperties(layer, printed_properties, node_properties);
        });
#endif

    auto sortedNet = CNNNetSortTopologicallyEx(newNet, make_fuzed_order);

    if (sortedNet.empty()) {
        THROW_GNA_EXCEPTION << "Sorted network is empty";
    }
    // Copy operations connected to memory layer (Assign to state variable) should be executed when all functional
    // layers are calculated. To simplify, just moving these Copy operations at the end of the execution list
    std::stable_partition(sortedNet.begin(), sortedNet.end(), [&](CNNLayerPtr layer) {
        return !LayerInfo(layer).isCopyToMemory();
    });

    std::vector<CNNLayerPtr> sortedNoMem;
    std::unordered_map<std::string, std::vector<InferenceEngine::CNNLayerPtr>> memoryPairs;
    // find all memory layers pairs and mark which one used as outputs
    int id = 0;
    for (auto& layer : sortedNet) {
        // set order id for layers to use it in compact mode
        LayerInfo layerInfo(layer);
        IE_SUPPRESS_DEPRECATED_START
        // Increase lifetime of input data buffer needed by DelayedCopy layer to avoid overwritting buffer
        // by downstream layers.
        layer->userValue.v_int = layerInfo.isCopyDelayed() ? std::numeric_limits<int>::max() : id++;
        IE_SUPPRESS_DEPRECATED_END
        auto generic = dynamic_cast<GenericLayer*>(layer.get());
        if (generic == nullptr) {
            sortedNoMem.push_back(layer);
            continue;
        }

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

    if (!graphCompiler.memory_connection.empty() && gnaFlags->num_requests != 1) {
        gnaFlags->num_requests = 1;
    }

    // keep inputs information and create input primitives
    inputs_data_map_ = newNet.getInputsInfo();
    if (inputs_data_map_.empty()) {
        log::warning() << "No inputs for the topology\n";
    }

    // keep output dims
    outputs_data_map_ = newNet.getOutputsInfo();
    if (outputs_data_map_.empty()) {
        THROW_GNA_EXCEPTION << "No outputs for the topology";
    }

    for (auto&& input : inputs_data_map_) {
        inputs_ptr_->at(input.first).ptrs.resize(gnaFlags->num_requests);
    }

    // Creating Layer primitives
    for (auto& layer : sortedNoMem) {
        graphCompiler.CreateLayerPrimitive(layer);
    }

    for (auto& inputLayer : inputLayers) {
        auto layerInfo = LayerInfo(inputLayer);
        if (layerInfo.isInput() && 0 == inputs_ptr_->at(inputLayer->name).get_allocated_size()) {
            graphCompiler.connectOutput(inputLayer, &inputs_ptr_->at(inputLayer->name).ptrs.front(), 0);
        }
    }

    if (graphCompiler.dnnComponents.components.empty()) {
        log::warning() << "No GNA primitives created based on topology. This might indicate trivial topology\n";
        trivialTopology = true;
    }

    /// setting-up output layers information
    int portId = 0;
    for (auto&& outPort : outputs_data_map_) {
        // gets output layer pointer in original topology not in cloned
        auto outLayer = getCreatorLayer(outPort.second).lock();

        // Memory layers are not dnnComponents hence we need to make switch with identity layer
        if (outLayer->type == "Memory") {
            // traverse memory connection to find corresponding output_memory
            for (auto&& memConnection : graphCompiler.memory_connection) {
                if (memConnection.second.getInput()->name == outLayer->name) {
                    // if connection is found, replace memory input layer with memory output layer
                    outLayer = memConnection.second.getOutput();
                    break;
                }
            }
        }

        // searching for outData represented in GNA blob
        // using ufs - upper first search
        log::debug() << "[UFS] searching for : " << outPort.first << " representation in GNA\n";
        bool stopSearching = false;

        CNNNetDFS(
            outLayer,
            [this, &outPort, &stopSearching](CNNLayerPtr layer) {
                log::debug() << "[UFS] from : " << outPort.first << " reached: " << layer->name << "\n";
                stopSearching = TryToInitOutput(outPort.first, layer);
            },
            true,
            [&stopSearching](InferenceEngine::CNNLayer* from) {
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
    gnamem->getQueue(REGION_OUTPUTS)
        ->reserve_ptr(nullptr, nullptr, ALIGN64(outputs_.Get().begin()->get_required_size()), 64);

    void* pParallelExecutionData = nullptr;

    // reserving more bytes for intermediate data in parallel case
    // TODO: this works incorrectly in compact mode at lest
    rwSegmentSize = gnamem->getRegionBytes(REGION_SCRATCH);
    rwSegmentSize += gnamem->getRegionBytes(REGION_INPUTS);
    rwSegmentSize += gnamem->getRegionBytes(REGION_OUTPUTS);
    if (gnaFlags->num_requests > 1) {
        gnamem->getQueue(REGION_SCRATCH)
            ->reserve_ptr(nullptr, &pParallelExecutionData, rwSegmentSize * (gnaFlags->num_requests - 1), 64);
    }

    gnamem->commit(gnaFlags->compact_mode);

    dnn->Init(gnamem.get(), gnaFlags->sw_fp32 ? kDnnFloat : kDnnInt, 1);

    // TODO: this copy is unneeded; in fact, we can directly create gna structs from list
    auto execOrder = graphCompiler.dnnComponents.getExecutionOrder();
    dnn->component.insert(dnn->component.begin(), execOrder.begin(), execOrder.end());

    // in fp32 mode last PWL cannot be computed without that
    if (!graphCompiler.dnnComponents.components.empty()) {
        dnn->InitActiveList(NULL);
    }

    auto worker = createWorkerForLoadNetwork(trivialTopology, isFP32ModeActive());
    requestWorkerPool_->addModelWorker(std::move(worker));

    // initialize paraler requests model
    // creating same gna RW segment for parallel infer requests
    for (int i = 1; i != gnaFlags->num_requests; i++) {
        auto basePtr = reinterpret_cast<uint8_t*>(pParallelExecutionData) + rwSegmentSize * (i - 1);

        auto relocate = [basePtr, this](void*& ptr_out, void* ptr_in) {
            if (ptr_in == nullptr) {
                ptr_out = nullptr;
            } else {
                const auto found = gnamem->getOffsetForMerged(ptr_in);
                if (!found.first) {
                    THROW_GNA_EXCEPTION << "Relocation offset for parallel infer requests was not found\n";
                }
                ptr_out = basePtr + found.second;
            }
        };

        for (auto& input : inputs_ptr_->Get()) {
            relocate(input.ptrs[i], input.ptrs[0]);
        }

        // relocating all output pointers
        for (auto& output : outputs_.Get()) {
            relocate(output.ptrs[i], output.ptrs[0]);
        }

        auto worker = createWorkerForLoadNetwork(trivialTopology, isFP32ModeActive());
        auto model = worker->model();

        // relocating all operations data pointers
        for (int j = 0; j != model->NumberOfOperations; j++) {
            auto& gnaOperation = model->Operations[j];
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[0])->Data, gnaOperation.Operands[0]->Data);
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[1])->Data, gnaOperation.Operands[1]->Data);
        }
        requestWorkerPool_->addModelWorker(std::move(worker));
    }

    // calculating input orientation without memory layers, since their orientation not changed during infer right now
    std::unordered_map<string, std::vector<string>> skippedLayers;

    // update orientation of model intput layer
    for (auto& inputLayer : inputLayers) {
        if (LayerInfo(inputLayer).isInput()) {
            ov::intel_gna::helpers::updateModelInputOrientationWithoutConvolution(*inputLayer,
                                                                                  graphCompiler.dnnComponents,
                                                                                  *inputs_ptr_);
        }
    }

    // update orientation of model output layer
    for (auto&& outPort : outputs_data_map_) {
        auto outLayer = getCreatorLayer(outPort.second).lock();
        if (outLayer && LayerInfo(outLayer).isOutput()) {
            ov::intel_gna::helpers::updateModelOutputOrientation(outPort.first,
                                                                 outLayer->name,
                                                                 graphCompiler.dnnComponents,
                                                                 outputs_);
        }
    }

    if (dnn->do_rotate_input && transpose_inputs_info.empty()) {
        for (auto& inputLayer : inputLayers) {
            transpose_inputs_info.insert(
                {inputLayer->name,
                 {TranspositionInfo{dnn->do_rotate_input, dnn->num_rotate_rows, dnn->num_rotate_columns}}});
        }
    }
    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob.dot");
#endif
}

bool GNAPlugin::isFP32ModeActive() const {
    return gnaFlags->sw_fp32 || !gnadevice;
}

std::shared_ptr<request::Worker> GNAPlugin::createWorkerForLoadNetwork(bool trivial, bool fp32Mode) {
    // Do not initialize gna model for fp32 mode (create trivial model wraper)
    return createWorker(createModelWrapperForLoadNetwork(trivial || fp32Mode), trivial, fp32Mode);
}

std::shared_ptr<request::Worker> GNAPlugin::createWorker(std::shared_ptr<request::ModelWrapper> modelWrapper,
                                                         bool trivial,
                                                         bool fp32Mode) {
    if (trivial) {
        return request::WorkerFactory::createWorkerTrivialTopology(std::move(modelWrapper));
    }

    if (fp32Mode) {
        if (!dnn) {
            THROW_GNA_EXCEPTION << "dnn is nullptr cannot run fp32 mode";
        }
        return request::WorkerFactory::createWorkerFP32(std::move(modelWrapper), dnn);
    }

    // This shouldn't happend due the fact device is created when gnaFlags->sw_fp32 is false.
    if (!gnadevice) {
        THROW_GNA_EXCEPTION << "device is nullptr cannot run in device mode";
    }

    return request::WorkerFactory::createWorker(std::move(modelWrapper), gnadevice, config.pluginGna2AccMode);
}

std::shared_ptr<request::ModelWrapper> GNAPlugin::createModelWrapperForLoadNetwork(bool trivial) {
    if (trivial) {
        return request::ModelWrapperFactory::createTrivial();
    }

    if (!dnn) {
        THROW_GNA_EXCEPTION << "dnn is nullptr cannot load network";
    }

    std::weak_ptr<backend::AMIntelDNN> weakDnn = dnn;
    auto initializer = [weakDnn](Gna2Model* model) {
        if (auto dnn = weakDnn.lock()) {
            dnn->InitGNAStruct(model);
            return;
        }
        THROW_GNA_EXCEPTION << "dnn is nullptr";
    };

    return request::ModelWrapperFactory::createInitialized(std::move(initializer));
}

std::shared_ptr<request::ModelWrapper> GNAPlugin::createModelWrapperForImportNetwork(uint32_t numberOfOperations) {
    return request::ModelWrapperFactory::createWithNumberOfEmptyOperations(numberOfOperations);
}

void GNAPlugin::DumpXNNToFile() const {
    // TODO: output  precision as well as pointer might be incorrect, LSTM for sure
    // gna looks automatically set layer 0 as output and adjust it's pointer / precision/ size respectively
    if (config.embedded_export_path.empty()) {
        return;
    }

    if (!gnadevice) {
        THROW_GNA_EXCEPTION << "Cannot generate XNNDump for float network";
    }

    if (requestWorkerPool_->empty()) {
        THROW_GNA_EXCEPTION << "Cannot generate XNNDump for not exsisting model";
    }

    std::ofstream dumpStream(config.embedded_export_path, std::ios::out | std::ios::binary);

    auto model = const_cast<Gna2Model*>(requestWorkerPool_->firstWorker().model());

    auto const modelId = gnadevice->createModel(*model);
    const auto& inputsDesc = inputs_ptr_->Get();
    const auto& outputsDesc = outputs_.Get();

    if (config.target->get_effective_compile_target() == target::DeviceVersion::GNA1_0) {
        auto dump = gnadevice->dumpXnn(modelId);
        dump.header.RwRegionSize = gnamem->getRegionBytes(REGION_SCRATCH);
        dump.header.InputScalingFactor = inputsDesc.begin()->scale_factor;
        dump.header.OutputScalingFactor = outputsDesc.begin()->scale_factor;
        dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(Gna2ModelSueCreekHeader));
        dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.ModelSize);
    } else {
        const auto inputsForTlv = GnaEndpoint::CreateFromDescriptorContainer(inputsDesc);
        const auto outputsForTlv = GnaEndpoint::CreateFromDescriptorContainer(outputsDesc);
        gnadevice->dumpTLVForDeviceVersion(modelId, dumpStream, inputsForTlv, outputsForTlv);
    }
    gnadevice->releaseModel(modelId);
}

uint32_t GNAPlugin::QueueInference(const InferenceEngine::BlobMap& inputs, InferenceEngine::BlobMap& result) {
    auto freeWorker = requestWorkerPool_->findFreeModelWorker();
    if (freeWorker == nullptr) {
        if (!graphCompiler.memory_connection.empty()) {
            Wait(requestWorkerPool_->firstWorker().representingIndex());
            freeWorker = requestWorkerPool_->findFreeModelWorker();
            if (freeWorker == nullptr) {
                THROW_GNA_EXCEPTION << "could not find free executable network for request" << std::endl;
            }
        } else {
            IE_THROW(RequestBusy) << "GNA executable network has max of "
                                  << static_cast<uint32_t>(gnaFlags->num_requests)
                                  << " parallel infer requests, please sync one of already running";
        }
    }

    auto index = freeWorker->representingIndex();

    int inputNum = 0;
    for (auto& input : inputs) {
        auto inputLayout = input.second->getTensorDesc().getLayout();
        if (inputLayout != InferenceEngine::Layout::C && inputLayout != InferenceEngine::Layout::NC &&
            inputLayout != InferenceEngine::Layout::CN && inputLayout != InferenceEngine::Layout::CHW &&
            inputLayout != InferenceEngine::Layout::NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or "
                                   "Layout::CHW. But was: "
                                << input.second->getTensorDesc().getLayout();
        }

        if (inputLayout == InferenceEngine::Layout::NCHW || inputLayout == InferenceEngine::Layout::CHW) {
            // specific case that can be squeezed to 2d
            inputLayout = InferenceEngine::Layout::NC;
        }

        auto is1D = input.second->getTensorDesc().getLayout() == InferenceEngine::Layout::C;
        auto is3D = input.second->getTensorDesc().getLayout() == InferenceEngine::Layout::CHW;

        if (inputs_ptr_->at(input.first).ptrs.empty()) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (inputs_ptr_->at(input.first).ptrs[index] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << index << " not set";
        }
        const auto inputOrientation = inputs_ptr_->at(input.first).orientation;
        if (inputOrientation == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input orientation for " << input.first << " not set";
        }

        for (auto& output : outputs_.Get()) {
            if (output.orientation == kDnnUnknownOrientation) {
                // should not happen in user code however might happen if there any non executable network based
                // integration of GNAPlugin instance
                THROW_GNA_EXCEPTION << "network not loaded : output orientation not set";
            }
        }

        auto dims = input.second->getTensorDesc().getDims();
        auto importedElements =
            is1D ? dims[0] : InferenceEngine::details::product(std::next(std::begin(dims)), std::end(dims));
        auto importedFrames = (is3D || is1D) ? 1 : dims[0];
        auto targetGroups = is1D ? 1 : dims[0];  // TODO: no proper support for groups yet

        auto importedElementSizeBytes = gnaFlags->sw_fp32 ? 4 : (gnaFlags->input_low_precision ? 1 : 2);
        auto importedBytes = importedElements * importedFrames * importedElementSizeBytes;

        if (inputs_ptr_->at(input.first).get_required_size() < importedBytes) {
            THROW_GNA_EXCEPTION << "Cannot import input frames for :" << input.first
                                << ", allocated size: " << inputs_ptr_->at(input.first).get_required_size()
                                << ", but input blob size: " << importedBytes;
        }

        ImportFrames(inputs_ptr_->at(input.first).ptrs[index],
                     input.second->cbuffer().as<float*>(),
                     input.second->getTensorDesc().getPrecision(),
                     gnaFlags->sw_fp32 ? kScaleFactorDefault : inputs_ptr_->at(input.first).scale_factor,
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
            for (const auto& part_transposition_info : transpose_info->second) {
                transposed_data_size +=
                    part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match input buffer length of " << elementsPerBatch;
            }
            auto input_ptr = reinterpret_cast<uint8_t*>(inputs_ptr_->at(input.first).ptrs[index]);
            ConvertTensorFromNCHWToNHWC(gnadevice ? 2 : 4,
                                        batchSize,
                                        elementsPerBatch,
                                        input_ptr,
                                        true,
                                        transpose_info->second);
        }
        ++inputNum;
    }

    if (!freeWorker->enqueueRequest()) {
        THROW_GNA_EXCEPTION << "Error with enqueueing inference request";
    }

    freeWorker->setResult(result);

#ifdef PLOT
    dnn->BeginNewWrite(dnn_dump_write_index);
    if (dnn->num_components() != 0) {
        dnn->WriteDnnText("Net_.txt", kDnnFloat);
    }
    dnn_dump_write_index++;
#endif

    return index;
}

bool GNAPlugin::Wait(uint32_t request_idx) {
    auto result = WaitFor(request_idx, MAX_TIMEOUT);

    if (result == RequestStatus::kCompletedWithError) {
        THROW_GNA_EXCEPTION << "Error when waiting for inference results!";
    }

    return result == RequestStatus::kCompleted;
}

RequestStatus GNAPlugin::WaitFor(uint32_t request_idx, int64_t millisTimeout) {
    // TODO: GNA2: check whether
    if (requestWorkerPool_->size() <= request_idx) {
        return RequestStatus::kCompleted;
    }

    auto& worker = requestWorkerPool_->worker(request_idx);

    if (worker.isFree()) {
        return RequestStatus::kCompleted;
    }

    const auto waitStatus = worker.wait(millisTimeout);

    if (waitStatus == RequestStatus::kCompletedWithError) {
        return waitStatus;
    }

    if (waitStatus == RequestStatus::kAborted) {
        return waitStatus;
    }

    if (waitStatus == RequestStatus::kPending) {
        return waitStatus;
    }

    auto& requestResult = worker.result();

#ifdef PLOT
    if (dnn->num_components() != 0) {
        dnn->WriteInputAndOutputText();
    }

    // TODO test
    dnn->WriteInputAndOutputTextGNA(*worker.model());
#endif
    for (auto&& outputBlobIt : requestResult) {
        auto& outputBlob = outputBlobIt.second;
        auto& outputDesc = outputs_.at(outputBlobIt.first);
        if (outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::C &&
            outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::NC &&
            outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::CN &&
            outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::NCHW &&
            outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::CHW &&
            outputBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR) {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or "
                                   "Layout::CHW. But was "
                                << outputBlob->getTensorDesc().getLayout();
        }

        auto dims = outputBlob->getTensorDesc().getDims();
        auto is1D = outputBlob->getTensorDesc().getLayout() == InferenceEngine::Layout::C;
        auto isScalar = outputBlob->getTensorDesc().getLayout() == InferenceEngine::Layout::SCALAR;
        auto is3D = outputBlob->getTensorDesc().getLayout() == InferenceEngine::Layout::CHW;
        auto batchSize = (is1D || isScalar || is3D) ? 1 : dims[0];
        auto elementsPerBatch =
            isScalar ? 1
                     : (is1D ? dims.front() : InferenceEngine::details::product(++std::begin(dims), std::end(dims)));

        auto transpose_output_info = transpose_outputs_info.find(outputBlobIt.first);
        if (transpose_output_info != std::end(transpose_outputs_info) &&
            FoundPartToTranspose(transpose_output_info->second)) {
            size_t transposed_data_size = 0;
            for (const auto& part_transposition_info : transpose_output_info->second) {
                transposed_data_size +=
                    part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match output buffer length of " << elementsPerBatch;
            }
            ConvertTensorFromNCHWToNHWC(outputDesc.tensor_precision.size(),
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
                     outputDesc.tensor_precision,
                     outputDesc.model_precision);

        if (gnadevice) {
#ifdef PLOT
            FILE* f = nullptr;
            static int num_infers = 0;
            {
                f = std::fopen("ex_scores.txt", "w");
                if (!f) {
                    THROW_GNA_EXCEPTION << "ex_scores.txt opening failed";
                }
            }
            num_infers++;
            if (f) {
                if (isScalar) {
                    fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t*>()[0]);
                } else {
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                }
                fprintf(f, "\n\n");
            }
#endif
            switch (outputBlob->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                UnscaleAndCast(outputBlob->buffer().as<float*>(),
                               outputBlob->buffer().as<int32_t*>(),
                               elementsPerBatch,
                               batchSize,
                               outputDesc.scale_factor);
                break;

            case InferenceEngine::Precision::I32:
                UnscaleAndCast(outputBlob->buffer().as<int32_t*>(),
                               outputBlob->buffer().as<int32_t*>(),
                               elementsPerBatch,
                               batchSize,
                               outputDesc.scale_factor);
                break;

            default:
                THROW_GNA_EXCEPTION << "Unsupported target precision: " << outputBlob->getTensorDesc().getPrecision()
                                    << std::endl;
                break;
            }

#ifdef PLOT
            if (f) {
                if (isScalar) {
                    fprintf(f, "%.7f ", outputBlob->cbuffer().as<float*>()[0]);
                } else {
                    auto dims = outputBlob->getTensorDesc().getDims();
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%.7f ", outputBlob->cbuffer().as<float*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                }
                fclose(f);
            }
#endif
        }
    }
    return RequestStatus::kCompleted;
}

void GNAPlugin::Reset() {
    graphCompiler.Reset();
}

bool GNAPlugin::Infer(const InferenceEngine::Blob& input, InferenceEngine::Blob& output) {
    BlobMap bmInput;
    BlobMap bmOutput;
    if (inputs_data_map_.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"
                            << "model accepts " << inputs_data_map_.size() << " inputs";
    }

    IE_ASSERT(!inputs_data_map_.empty());
    bmInput[inputs_data_map_.begin()->first] = std::shared_ptr<Blob>(const_cast<Blob*>(&input), [](Blob*) {});
    IE_ASSERT(!outputs_data_map_.empty());
    bmOutput[outputs_data_map_.begin()->first] = std::shared_ptr<Blob>(&output, [](Blob*) {});
    return Infer(bmInput, bmOutput);
}

bool GNAPlugin::Infer(const InferenceEngine::BlobMap& input, InferenceEngine::BlobMap& result) {
    return Wait(QueueInference(input, result));
}

static InferenceEngine::Layout GetLayoutForDims(const InferenceEngine::SizeVector& dims) {
    switch (dims.size()) {
    case 0:
        return SCALAR;
    case 1:
        return C;
    case 2:
        return NC;
    case 3:
        return CHW;
    case 4:
        return NCHW;
    default:
        THROW_GNA_EXCEPTION << "Unsupported dimensions size in GNA: " << dims.size();
    }
}

Blob::Ptr GNAPlugin::GetOutputBlob(const std::string& name, InferenceEngine::Precision precision) {
    // need to have intermediate blob for interleave conversion
    InferenceEngine::Blob::Ptr outputBlob;
    auto outputDataIt = outputs_data_map_.find(name);
    if (outputDataIt == std::end(outputs_data_map_)) {
        THROW_GNA_EXCEPTION << "Output " << name << " isn't found";
    }
    auto outputDims = outputDataIt->second->getTensorDesc().getDims();
    outputBlob = make_blob_with_precision(TensorDesc(precision, outputDims, GetLayoutForDims(outputDims)));
    outputBlob->allocate();
    return outputBlob;
}

Blob::Ptr GNAPlugin::GetInputBlob(const std::string& name, InferenceEngine::Precision precision) {
    InferenceEngine::Blob::Ptr inputBlob;
    // need to have intermediate blob for interleave conversion
    // TODO: NCHW format support is experimental = c++ MO did insert reshape, while TF mo - not
    auto inputDataIt = inputs_data_map_.find(name);
    if (inputDataIt == std::end(inputs_data_map_)) {
        THROW_GNA_EXCEPTION << "Input " << name << " isn't found";
    }
    auto inputDims = inputDataIt->second->getTensorDesc().getDims();
    inputBlob = make_blob_with_precision(TensorDesc(precision, inputDims, GetLayoutForDims(inputDims)));
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::IVariableStateInternal::Ptr> GNAPlugin::QueryState() {
    if (memoryStates.size() != graphCompiler.memory_connection.size()) {
        memoryStates.clear();
        for (auto& connection : graphCompiler.memory_connection) {
            auto state =
                std::make_shared<memory::GNAVariableState>(connection.first,
                                                           std::make_shared<GNAMemoryLayer>(connection.second));
            memoryStates.emplace_back(state);
        }
    }
    return memoryStates;
}

std::string GNAPlugin::GetName() const noexcept {
    return _pluginName;
}

void GNAPlugin::SetName(const std::string& pluginName) noexcept {
    _pluginName = pluginName;
}

InferenceEngine::IExecutableNetworkInternal::Ptr GNAPlugin::ImportNetwork(std::istream& networkModel) {
    auto header = GNAModelSerial::ReadHeader(networkModel);

    void* basePtr = nullptr;
    std::string modelLibVersion;  //!< OpenVINO and GNA Library versions read from GNA model file

    gnamem->getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, &basePtr, header.gnaMemSize);

    gnamem->commit();

    auto model = createModelWrapperForImportNetwork(header.layersCount);
    GNAModelSerial::MemoryType mt;
    auto serial = GNAModelSerial(&model->object(), mt);

    serial.setHeader(header);
    serial.Import(basePtr,
                  header.gnaMemSize,
                  networkModel,
                  *(inputs_ptr_),
                  outputs_,
                  transpose_inputs_info,
                  transpose_outputs_info,
                  modelLibVersion);

    // Print OV and GNA Lib versions used for model export
    if (gnaFlags->log_level >= ov::log::Level::DEBUG) {
        if (modelLibVersion.length()) {
            std::cout << modelLibVersion << std::endl;
        } else {
            std::cout
                << "Unable to read OpenVINO or GNA Library version from model file, consider model export with current "
                   "version of GNA plugin"
                << std::endl;
        }
    }

    trivialTopology = (model->object().NumberOfOperations == 0);

    requestWorkerPool_->addModelWorker(createWorker(model, trivialTopology, isFP32ModeActive()));

    SetNetworkInputs();
    SetNetworkOutputs();

    ov::intel_gna::helpers::ApplyInputScaleFactors(*inputs_ptr_, config, header);

    auto getOrientation = [](Gna2Operation& gnaOperation) {
        return gnaOperation.Type == Gna2OperationTypeConvolution ? kDnnNonInterleavedOrientation
                                                                 : kDnnInterleavedOrientation;
    };
    (void)getOrientation;

    if (header.doRotateInput) {
        for (auto&& input : inputs_data_map_) {
            transpose_inputs_info.insert(
                {input.first, {{header.doRotateInput, header.nRotateRows, header.nRotateColumns}}});
        }
    }
    if (header.doRotateOutput) {
        for (auto&& output : outputs_data_map_) {
            transpose_outputs_info.insert(
                {output.first, {{header.doRotateOutput, header.nRotateOutputRows, header.nRotateOutputColumns}}});
        }
    }

    for (auto&& memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr, gnaFlags->sw_fp32 ? 4 : 2);
        std::string name;
        std::tie(memoryLayer.gna_ptr, memoryLayer.reserved_size, name, memoryLayer.scale_factor) = memory;
        graphCompiler.memory_connection.emplace_back(make_pair(name, memoryLayer));
    }

    // TODO update documenation to allow exporting tlv with importing cep only for sue creek
    // TODO tlv + cep import + export
    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob-imported.dot");
#endif
    return {};
}

void GNAPlugin::Export(const std::string& fileName) {
    std::fstream outStream(fileName, ios_base::out | ios_base::binary);
    Export(outStream);
}

void GNAPlugin::Export(std::ostream& outStream) {
    if (inputs_ptr_->empty() || outputs_.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

    // TODO: nnet group parameter looks only used in application - so can we move this line into load network.
    IE_ASSERT(!inputs_data_map_.empty());
    auto inputDims = inputs_data_map_.begin()->second->getTensorDesc().getDims();

    Gna2Model* model_to_serial = requestWorkerPool_->firstWorker().model();
    auto serial = GNAModelSerial(model_to_serial, *(inputs_ptr_), outputs_)
                      .SetInputRotation(transpose_inputs_info)
                      .SetOutputRotation(transpose_outputs_info);

    for (auto&& memoryConnection : graphCompiler.memory_connection) {
        auto state =
            std::make_shared<memory::GNAVariableState>(memoryConnection.first,
                                                       std::make_shared<GNAMemoryLayer>(memoryConnection.second));
        log::debug() << "Scale factor Memory layer " << state->GetScaleFactor() << std::endl;
        serial.AddState(memoryConnection.second.gna_ptr,
                        memoryConnection.second.reserved_size,
                        memoryConnection.first,
                        state->GetScaleFactor());
    }

    serial.Export(gnadevice->getAllAllocations(), outStream);
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

void GNAPlugin::AddExtension(const InferenceEngine::IExtensionPtr& extension) {}

void GNAPlugin::SetConfig(const std::map<std::string, std::string>& config_map) {
    config.UpdateFromMap(config_map);
    UpdateFieldsFromConfig();
}

void GNAPlugin::UpdateFieldsFromConfig() {
    *gnaFlags = config.gnaFlags;
}

void GNAPlugin::SetNetworkInputs() {
    inputs_data_map_.clear();
    for (auto& input : inputs_ptr_->Get()) {
        inputs_data_map_[input.name] = input.ToIEInputInfo();
    }
}

void GNAPlugin::SetNetworkOutputs() {
    outputs_data_map_.clear();
    for (auto& output : outputs_.Get()) {
        outputs_data_map_[output.name] = output.to_ie_data();
    }
}

std::vector<std::shared_ptr<const ov::Node>> GNAPlugin::GetInputs() {
    std::vector<std::shared_ptr<const ov::Node>> params;
    params.reserve(inputs_ptr_->size());
    for (auto&& input : inputs_ptr_->Get()) {
        auto param = std::make_shared<ov::op::v0::Parameter>(convertPrecision(input.model_precision),
                                                             ov::PartialShape(input.dims));
        param->set_friendly_name(input.name);
        param->get_output_tensor(0).add_names(input.tensor_names);
        params.emplace_back(move(param));
    }
    return params;
}

std::vector<std::shared_ptr<const ov::Node>> GNAPlugin::GetOutputs() {
    std::vector<std::shared_ptr<const ov::Node>> results;
    results.reserve(outputs_.size());
    for (auto&& output : outputs_.Get()) {
        auto param = std::make_shared<ov::op::v0::Parameter>(convertPrecision(output.model_precision),
                                                             ov::PartialShape(output.dims));
        param->set_friendly_name(output.name);
        auto result = std::make_shared<ov::op::v0::Result>(param);
        result->get_output_tensor(0).add_names(output.tensor_names);
        results.emplace_back(std::move(result));
    }
    return results;
}

InferenceEngine::QueryNetworkResult GNAPlugin::QueryNetwork(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config_map) const {
    InferenceEngine::QueryNetworkResult res;

    Config qn_config(config);
    qn_config.UpdateFromMap(config_map);

    const auto effectiveCompileTarget = qn_config.target->get_effective_compile_target();
    auto model = network.getFunction();
    if (model) {
        auto supported = GetSupportedNodes(
            model,
            [&](std::shared_ptr<ov::Model>& model) {
                TransformationsPipeline(qn_config).apply(model);
            },
            [&](const std::shared_ptr<ngraph::Node>& op) {
                return limitations::is_op_supported(op, effectiveCompileTarget, qn_config.gnaPrecision);
            });
        for (auto&& op_name : supported) {
            res.supportedLayersMap.emplace(op_name, GetName());
        }
        return res;
    }

    std::unordered_set<CNNLayer*> allLayers;
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);

    if (inputs.empty()) {
        THROW_GNA_EXCEPTION << "Network is empty (GNA)\n";
    }

    auto const& secondLayers = getInputTo(inputs.begin()->second->getInputData());
    if (secondLayers.empty()) {
        THROW_GNA_EXCEPTION << "Network consists of input layer only (GNA)\n";
    }

    InferenceEngine::details::UnorderedDFS(
        allLayers,
        secondLayers.begin()->second,
        [&](CNNLayerPtr const& layer) {
            if (LayerTypeFromStr(layer->type) != LayerType::NO_TYPE) {
                res.supportedLayersMap.insert({layer->name, GetName()});
            }
        },
        false);

    return res;
}

GNAPlugin::~GNAPlugin() {
    if (gnadevice)
        gnadevice->close();
}
