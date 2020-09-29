// Copyright (C) 2018-2020 Intel Corporation
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
#include <legacy/ie_util_internal.hpp>
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
            for (uint32_t i = 0; i < num_frames; i++) {
                void *ptr_dst_vec = reinterpret_cast<uint8_t *>(ptr_dst) + i * num_vector_elements * sizeof(float);
                const void *ptr_src_vec = reinterpret_cast<const uint8_t *>(ptr_src) + i * num_vector_stride * sizeof(float);
                memset(ptr_dst_vec, 0, num_vector_elements * sizeof(float));
                ie_memcpy(ptr_dst_vec, num_active_elements * sizeof(float),
                    ptr_src_vec, num_active_elements * sizeof(float));
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

void GNAPlugin::LoadNetwork(ICNNNetwork & _network) {
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork;
    if (_network.getFunction()) {
        convertedNetwork = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(_network);
    }
    InferenceEngine::ICNNNetwork &network = convertedNetwork ? *convertedNetwork : _network;

    NetPass::ConvertPrecision(network, Precision::I64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U32, Precision::I32);

    //  Check the input network
    std::string error;
    if (!AreLayersSupported(network, error)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // network optimisation phases
    int passIdx = 0;
    auto run_passes = [&] (const CNNNetPtr& network, bool runBeforeCopy) {
        auto passes = make_shared<PassManager>(PassManagerSettings{policy, runBeforeCopy}, network);
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<UnrollTIPass>();
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<InsertIdentityToLSTMCellPass>();
        passes->registerPass<UnrollLSTMCellPass>();
        passes->registerPass<RemoveSingleInputConcatPass>();

        passes->registerPass<SubstitutePReluPass>();
        passes->registerPass<SubstituteSoftSignPass>();

        passes->registerPass<ReorderMaxPoolPass>();
        passes->registerPass<EltwiseSplitOverChannelsPass>();
        passes->registerPass<InsertSplitAligningFilterPass>();

        passes->registerPass<InsertConcatAligningFilterPass>();
        passes->registerPass<ReorderConcatInputsPass>();
        if (policy.PermutePolicy != Policy::Permute::DISABLED) {
            passes->registerPass<ReversePermutationsPass>();
        }
        if (policy.NHWCToNCHWPolicy != Policy::NHWCToNCHW::DISABLED) {
            passes->registerPass<RemovePermutationsNHWCToNCHWPass>();
        }
        passes->registerPass<InsertIdentityLayerPass>();
        passes->registerPass<InsertCopyLayerPass>();
        passes->registerPass<InsertDiagonalLayerPass>();
        passes->registerPass<HandleMultipleActivationsForTheLayerPass>();
        passes->registerPass<SubstituteScaleShiftBroadCastPass>();
        passes->registerPass<FuseMultipleIdentitiesPass>();
        passes->registerPass<BroadcastConstPass>();
        passIdx = passes->run(passIdx);
    };

    ICNNNetwork::Ptr newNet;
    if (gnaFlags->sw_fp32) {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            transformLayer(lp, WeightsConverter());
            return lp;
        };
        newNet = InferenceEngine::CNNNetCopy(network, visitor);
        // to run all passes need to have two calls to pass manager
        run_passes(newNet, true);
        run_passes(newNet, false);
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
                THROW_GNA_EXCEPTION << "no mans land for GNA precision";
                break;
        }
    }

    auto inputLayers = CNNNetGetAllInputLayers(*newNet);

#ifdef PLOT
    std::ofstream file("gna_passes.dot");
    saveGraphToDot(*newNet, file, [](const CNNLayerPtr layer,
                                           ordered_properties &printed_properties,
                                           ordered_properties &node_properties) {
        // printing quantized params
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
        if (!quantized) {
            return;
        }
        printed_properties.emplace_back(
            "scale factor", std::to_string(quantized->_dst_quant.scale));
    });
#endif

    auto sortedNet = CNNNetSortTopologicallyEx(*newNet, make_fuzed_order);

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
    newNet->getInputsInfo(inputsDataMap);
    if (inputsDataMap.empty()) {
        THROW_GNA_EXCEPTION << " No inputs for the topology";
    }

    // keep output dims
    newNet->getOutputsInfo(outputsDataMap);
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
    // TODO: graph might be static - should we support that
    if (graphCompiler.dnnComponents.components.empty()) {
        THROW_GNA_EXCEPTION << "No GNA primitives created based on topology. This might indicate trivial topology";
    }

    /// setting-up output layers information
    outputsDesc.resize(outputsDataMap.size());

    auto initOutput = [this]
            (int idx, const intel_dnn_component_t & component, CNNLayerPtr layer) {
        // auto idx = std::distance(outputsDataMap.begin(), outputPort);
        auto & desc = outputsDesc[idx];
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

        desc.ptrs.resize(gnaFlags->gna_lib_async_threads_num);
        desc.orientation = component.orientation_out;
        desc.num_bytes_per_element = component.num_bytes_per_output;
        desc.scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
        // TODO: this need to be fixed
        desc.num_elements = component.num_rows_out;

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->bind_ptr(&desc.ptrs.front(), &component.ptr_outputs);
    };

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

        CNNNetDFS(outLayer, [this, &outPort, portId, &stopSearching, &initOutput](CNNLayerPtr layer) {
            auto irLayerAvatar = std::find_if(
                graphCompiler.dnnComponents.components.begin(),
                graphCompiler.dnnComponents.components.end(),
                [&layer](const backend::DnnComponents::storage_type::value_type & value) {
                    return value.name == layer->name;
            });

            gnalog() << "[UFS] from : "<< outPort.first <<" reached: " << layer->name << "\n";

            // probing gna_primitives
            if (irLayerAvatar != graphCompiler.dnnComponents.components.end()) {
                initOutput(portId, irLayerAvatar->dnnComponent, layer);
                stopSearching = true;
            }

            // probing concatInfo
            if (!stopSearching && LayerInfo(layer).isConcat()) {
                auto concatConnection  = graphCompiler.concat_connection.find(layer->name);
                if (concatConnection != graphCompiler.concat_connection.end()) {
                    //initOutput(portId, irLayerAvatar->second, layer);

                    auto &desc = outputsDesc[portId];
                    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

                    desc.ptrs.resize(gnaFlags->gna_lib_async_threads_num);
                    // TODO: what is orientation for concat
                    desc.orientation = kDnnInterleavedOrientation;
                    desc.num_bytes_per_element = layer->outData.front()->getPrecision().size();
                    desc.scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
                    desc.num_elements = concatConnection->second.reserved_size / desc.num_bytes_per_element;

                    // binding ptr for first infer request - then others will be setup during relocation
                    gnamem->bind_ptr(&desc.ptrs.front(), &concatConnection->second.gna_ptr);
                    stopSearching = true;
                }
            }
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
    dnn->InitActiveList(NULL);

#if GNA_LIB_VER == 2
    gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>()));
#else
    nnets.emplace_back(make_shared<CPPWrapper<intel_nnet_type_t>>(), -1, InferenceEngine::BlobMap());
#endif
    if (!gnaFlags->sw_fp32) {
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

    do_rotate_input = dnn->do_rotate_input;
    num_rotate_rows = dnn->num_rotate_rows;
    num_rotate_columns = dnn->num_rotate_columns;

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
    if (!gnadevice) {
        gnaRequestConfigToRequestIdMap.push_back(std::make_tuple(FAKE_REQUEST_CONFIG_ID, -1, InferenceEngine::BlobMap()));
        return;
    }
    for (auto& model : gnaModels) {
        const auto& gnaNnet = std::get<0>(model).get()->obj;
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
            ie_memcpy(ptr_in, num_feature_vector_elements * element_size,
                &temp.front(), num_feature_vector_elements * element_size);
        }
    } else {
        THROW_GNA_EXCEPTION << "Rotate dimensions (" << num_rotate_rows << "," << num_rotate_columns
                           <<") do not match buffer length of "<< num_feature_vector_elements <<" in RotateFeatures()!";
    }
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
        if (inputLayout != Layout::NC && inputLayout != Layout::CN && inputLayout != NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::NC or Layout::CN, but was: "
                                << input.second->getTensorDesc().getLayout();
        }
        if (inputLayout == NCHW) {
            inputLayout = NC;
        }
        auto is2D = input.second->getTensorDesc().getLayout() == Layout::NC || input.second->getTensorDesc().getLayout() == Layout::CN;

        if (!inputsDesc->ptr_inputs_global_id.count(input.first)) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (inputsDesc->getPtrInputsGlobal(input.first)[idx] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << idx << " not set";
        }

        if (inputsDesc->getOrientation(input.first) == kDnnUnknownOrientation) {
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

        auto  importedElements = is2D ? dims[dims.size() - 1] : dims[dims.size() - 1] * dims[dims.size() - 2] * dims[dims.size() - 3];
        auto  importedFrames = dims[0];
        auto  targetGroups = is2D ? dims[dims.size() - 2] : dims[0]; // TODO: no proper support for groups yet

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
                     inputsDesc->getOrientation(input.first),
                     importedFrames,
                     targetGroups,
                     importedElements,
                     importedElements);

        bool isOneChannel = input.second->getTensorDesc().getDims()[1] == 1;
        if (do_rotate_input && ((inputLayout == Layout::NC || inputLayout == Layout::NCHW)
            != (inputsDesc->getOrientation(input.first) == kDnnInterleavedOrientation))
            && !isOneChannel) {
            RotateFeatures(reinterpret_cast<uint8_t *>(inputsDesc->getPtrInputsGlobal(input.first)[idx]),
                           gnadevice ? 2 : 4,
                           // TODO: only works for cnn4a and google command so far
                           dims[0],
                           is2D ? dims[dims.size() - 1] : dims[dims.size() - 1] * dims[dims.size() - 3],  // num_feature_vectors looks batch should be there
                           num_rotate_rows,
                           num_rotate_columns);
        }
        ++inputNum;
    }

    if (!gnadevice) {
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

    if (gnadevice) {
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
        if (outputBlob->getTensorDesc().getLayout() == Layout::NC || outputBlob->getTensorDesc().getLayout() == Layout::CN
            || outputBlob->getTensorDesc().getLayout() == Layout::NCHW || outputBlob->getTensorDesc().getLayout() == Layout::NHWC) {
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
            auto is2D = outputBlob->getTensorDesc().getLayout() == Layout::NC || outputBlob->getTensorDesc().getLayout() == Layout::CN;
            auto& exportOutputDims = outputBlob->getTensorDesc().getDims();
            auto batchSize = exportOutputDims[0];
            auto elementsPerBatch = is2D ? exportOutputDims[exportOutputDims.size() - 1]
                : exportOutputDims[exportOutputDims.size() - 1]
                  * exportOutputDims[exportOutputDims.size() - 2]
                  * exportOutputDims[exportOutputDims.size() - 3];

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
                    auto dims = outputBlob->getTensorDesc().getDims();
                    for (int i = 0; i < dims[dims.size() - 2]; i++) {
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
                    for (int i = 0; i < dims[dims.size() - 2]; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%.2f ", outputBlob->cbuffer().as<float*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                    fclose(f);
                }
#endif
            }
        } else {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::NC, Layout::CN, Layout::NCHW or Layout::NHWC. But was "
                << outputBlob->getTensorDesc().getLayout();
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

Blob::Ptr GNAPlugin::GetOutputBlob(const std::string& name, InferenceEngine::Precision precision) {
    // need to have intermediate blob for interleave conversion
    InferenceEngine::Blob::Ptr outputBlob;
    auto outputDims = outputsDataMap[name]->getTensorDesc().getDims();
    outputBlob = make_blob_with_precision(TensorDesc(precision, outputDims, outputDims.size() == 2 ? NC : NCHW));
    outputBlob->allocate();
    return outputBlob;
}

Blob::Ptr GNAPlugin::GetInputBlob(const std::string& name, InferenceEngine::Precision precision) {
    InferenceEngine::Blob::Ptr inputBlob;
    // need to have intermediate blob for interleave conversion
    // TODO: NCHW format support is experimental = c++ MO did insert reshape, while TF mo - not
    auto inputDims = inputsDataMap[name]->getTensorDesc().getDims();
    inputBlob = make_blob_with_precision(TensorDesc(precision, inputDims, inputDims.size() == 2 ? NC : NCHW));
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::MemoryStateInternal::Ptr>  GNAPlugin::QueryState() {
    if (memoryStates.size() != graphCompiler.memory_connection.size()) {
        memoryStates.clear();
        for (auto& connection : graphCompiler.memory_connection) {
            auto state = std::make_shared<memory::GNAMemoryState>(connection.first, std::make_shared <GNAMemoryLayer>(connection.second));
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

InferenceEngine::IExecutableNetwork::Ptr GNAPlugin::ImportNetwork(std::istream& networkModel) {
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

    serial.setHeader(header);
    serial.Import(basePtr,
            header.gnaMemSize,
            networkModel,
            inputsDesc,
            outputsDesc,
            inputsDataMap,
            outputsDataMap);

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

    do_rotate_input = header.doRotateInput;
    num_rotate_rows = header.nRotateRows;
    num_rotate_columns = header.nRotateColumns;

    for (auto && memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr, gnaFlags->sw_fp32 ? 4 : 2);
        memoryLayer.gna_ptr = memory.first;
        memoryLayer.reserved_size = memory.second;

        graphCompiler.memory_connection.emplace_back(make_pair(std::string("noname"), memoryLayer));
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob-imported.dot");
#endif
#if GNA_LIB_VER == 2
    createRequestConfigsForGnaModels();
#endif
    return nullptr;
}

void GNAPlugin::Export(const std::string &fileName) {
    if (inputsDesc->ptr_inputs_global_id.empty() || outputsDesc.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

#if GNA_LIB_VER == 1
    if (inputsDesc->ptr_inputs_global_id.size() != 1) {
        THROW_GNA_EXCEPTION << " exporting network with multiple inputs not supported";
    }
#endif

    std::fstream outStream(fileName, ios_base::out | ios_base::binary);

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
                    .SetInputRotation(dnn->num_rotate_rows, dnn->num_rotate_columns, dnn->do_rotate_input);

    for (auto && memoryConnection : graphCompiler.memory_connection) {
        serial.AddState(memoryConnection.second.gna_ptr, memoryConnection.second.reserved_size);
    }

    serial.Export(gnamem->getBasePtr(), gnamem->getTotalBytes(), outStream);
}

void GNAPlugin::GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) {
    if (gnaFlags->performance_counting) {
        gnadevice->getGnaPerfCounters(perfMap);
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

void GNAPlugin::QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                             const std::map<std::string, std::string>& config,
                             InferenceEngine::QueryNetworkResult& res) const {
    if (network.getFunction()) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << " ngraph::Function is not supported natively";
    }

    std::unordered_set<CNNLayer *> allLayers;
    InferenceEngine::InputsDataMap inputs;

    network.getInputsInfo(inputs);
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
}
