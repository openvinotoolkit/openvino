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

#include <low_precision_transformations/blob_transformation.hpp>
#include <graph_tools.hpp>
#include <debug.h>
#include <gna/gna_config.hpp>
#include <ie_util_internal.hpp>
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
                *(dst_ptr++) = GNAPluginNS::ConvertFloatToInt16(*(src_ptr++) * inputsDesc->inputScaleFactors[idx]);
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
    gnadevice = std::make_shared<GNADeviceHelper>(gna_proc_type,
                                        gnaFlags->gna_lib_async_threads_num,
                                        gnaFlags->gna_openmp_multithreading,
                                        gnaFlags->performance_counting);
#else
    gnadevice = std::make_shared<GNADeviceHelper>(pluginGna2AccMode,
                pluginGna2DeviceConsistent,
                gnaFlags->gna_lib_async_threads_num,
                gnaFlags->gna_openmp_multithreading,
                gnaFlags->performance_counting);
#endif
    size_t page_size_bytes = 4096;
    gnamem = std::make_shared<gna_memory_type>(memory::make_polymorph<memory::GNAAllocator>(gnadevice), page_size_bytes);
    graphCompiler.setGNAMemoryPtr(gnamem);
}

void GNAPlugin::LoadNetwork(ICNNNetwork &network) {
    // move blobs from Constant layers to Convolution, Deconvolution, FullyConnected layers attributes
    BlobTransformation blobsTransformation;
    blobsTransformation.transform(network, true);

    //  Check the input network
    std::string error;
    if (!AreLayersSupported(network, error)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // network optimisation phases
    auto run_passes = [&] (const CNNNetPtr& network, bool runBeforeCopy) {
        auto passes = make_shared<PassManager>(policy, network, runBeforeCopy);
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<UnrollTIPass>();
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<UnrollLSTMCellPass>();

        passes->registerPass<SubstitutePReluPass>();
        passes->registerPass<ReorderMaxPoolPass>();
        passes->registerPass<InsertSplitAligningFilterPass>();

        passes->registerPass<InsertConcatAligningFilterPass>();
        passes->registerPass<ReorderConcatInputsPass>();
        if (policy.PermutePolicy != Policy::Permute::DISABLED) {
            passes->registerPass<ReversePermutationsPass>();
        }
        passes->registerPass<InsertIdentityLayerPass>();
        passes->registerPass<InsertCopyLayerPass>();
        passes->registerPass<InsertDiagonalLayerPass>();
        passes->registerPass<HandleMultipleActivationsForTheLayerPass>();
        passes->registerPass<SubstituteScaleShiftBroadCastPass>();
        passes->run();
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
        switch (gnaPrecision) {
            case Precision::I16:
                ModelQuantizer<QuantI16> q16;
                std::cout << "Config gnaPrecision = I16 ip SF = " << inputsDesc->inputScaleFactors[0] << "\n";
		newNet = q16.quantize(network, run_passes, inputsDesc->inputScaleFactors);
                break;
            case Precision::I8:
                ModelQuantizer<QuantI8> q8;
		std::cout << "Config gnaPrecision = I8 ip SF = " << inputsDesc->inputScaleFactors[0] << "\n";
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

    auto networkPrecision = newNet->getPrecision();

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
        inputsDesc->get_ptr_inputs_global(input.first).resize(gnaFlags->gna_lib_async_threads_num);
    }

    // CreatingLayer primitives
    for (auto & layer : sortedNoMem) {
        graphCompiler.CreateLayerPrimitive(layer);
    }
    for (auto& inputLayer : inputLayers) {
        auto layerInfo = LayerInfo(inputLayer);
        if (layerInfo.isInput() && 0 == inputsDesc->bytes_allocated_for_input[inputLayer->name]) {
            graphCompiler.connectOutput(inputLayer, &inputsDesc->get_ptr_inputs_global(inputLayer->name).front(), 0);
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
        std::cout << "is quantized ? " << quantized << "\n";
        desc.ptrs.resize(gnaFlags->gna_lib_async_threads_num);
        desc.orientation = component.orientation_out;
        desc.num_bytes_per_element = component.num_bytes_per_output;
        desc.scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;
        
        std::cout << "desc.scale_factor " << desc.scale_factor << "\n";
        // TODO: this need to be fixed
        desc.num_elements = component.num_rows_out;

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->bind_ptr(&desc.ptrs.front(), &component.ptr_outputs);
    };

    int portId = 0;
    for (auto && outPort : outputsDataMap) {
        // gets output layer pointer in original topology not in cloned
        auto outLayer = outPort.second->getCreatorLayer().lock();

        // searching for outData represented in GNA blob
        // using ufs - upper first search
        gnalog() << "[UFS] searching for : "<< outPort.first << " representation in GNA\n";
        bool stopSearching = false;

        CNNNetDFS(outLayer, [this, &outPort, portId, &stopSearching, &initOutput](CNNLayerPtr layer) {
            auto irLayerAvatar = std::find_if(
                graphCompiler.dnnComponents.components.begin(),
                graphCompiler.dnnComponents.components.end(),
                [&layer](std::pair<std::string, intel_dnn_component_t> & value) {
                    return value.first == layer->name;
            });

            gnalog() << "[UFS] from : "<< outPort.first <<" reached: " << layer->name << "\n";

            if (irLayerAvatar != graphCompiler.dnnComponents.components.end()) {
                initOutput(portId, irLayerAvatar->second, layer);
                stopSearching = true;
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
    for (auto &element : graphCompiler.dnnComponents.components) {
        dnn->component.push_back(element.second);
    }

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
    std::unordered_map<string, string> skippedLayers;

    bool withConv = false;
    for (auto &layer : sortedNet) {
        auto layerInfo = LayerInfo(layer);
        if (layerInfo.isConvolution()) {
            withConv = true;
            break;
        }
    }
    if (withConv) {
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

                auto dnnLayer = graphCompiler.dnnComponents.findComponent(layer);
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
                if (!inputsDesc->orientation_in.count(inputName)) {
                    inputsDesc->orientation_in[inputName] = dnnLayer->orientation_in;
                } else {
                    if (inputsDesc->orientation_in[inputName] != dnnLayer->orientation_in) {
                        THROW_GNA_EXCEPTION << "orientation for input layer: " << inputName << "cannot be calculated";
                    }
                }
            }
        }
    } else {
        for (auto& inputLayer : inputLayers) {
            inputsDesc->orientation_in[inputLayer->name] = kDnnInterleavedOrientation;
        }
    }

    num_rotate_rows = dnn->num_rotate_rows;
    num_rotate_columns = dnn->num_rotate_columns;

    for (auto& gnaMemoryConn : graphCompiler.memory_connection) {
        std::string name = gnaMemoryConn.first;
        GNAMemoryLayer memLayer = gnaMemoryConn.second;

        InferenceEngine::CNNLayerPtr layer = memLayer.getInput();
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
        auto scale_factor = quantized != nullptr ? quantized->_dst_quant.scale : 1.0f;

        auto ptr = make_blob_with_precision(TensorDesc(InferenceEngine::Precision::I16,
                                            memLayer.getDims(),
                                            memLayer.getDims().size() == 2 ? NC : NCHW),
                                            memLayer.gna_ptr);
        graphCompiler.memoryStates.emplace_back(std::make_shared<memory::GNAMemoryState>(name, ptr, scale_factor));
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

void GNAPlugin::DumpXNNToFile() const {
    // TODO: output  precision as well as pointer might be incorrect, LSTM for sure
    // gna looks automatically set layer 0 as output and adjust it's pointer / precision/ size respectively
    if (dumpXNNPath.empty()) {
        return;
    }

    if (dumpXNNGeneration != "GNA1" &&
        dumpXNNGeneration != "GNA3" &&
        !dumpXNNGeneration.empty()) {
        THROW_GNA_EXCEPTION << "Wrong GNA generation for embedded model dump: " << dumpXNNGeneration;
    }

    if (!gnadevice) {
        THROW_GNA_EXCEPTION << "Cannot generate XNNDump for float network";
    }
    std::ofstream dumpStream(dumpXNNPath, std::ios::out | std::ios::binary);
#if GNA_LIB_VER == 1
    auto dump = gnadevice->dumpXnn(&std::get<0>(nnets.front())->obj, ptr_active_indices, num_active_indices);
    dump.header.rw_region_size = gnamem->getRWBytes();
    dump.header.input_scaling_factor = inputsDesc->inputScaleFactors.front();
    dump.header.output_scaling_factor = outputsDesc.front().scale_factor;
    dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(intel_gna_model_header));
    dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.model_size);
#else
    auto const modelId = gnadevice->createModel(std::get<0>(gnaModels.front())->obj);
    if (dumpXNNGeneration != "GNA3") {
        auto dump = gnadevice->dumpXnn(modelId);
        dump.header.RwRegionSize = gnamem->getRWBytes();
        dump.header.InputScalingFactor = inputsDesc->inputScaleFactors.front();
        dump.header.OutputScalingFactor = outputsDesc.front().scale_factor;
        dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(Gna2ModelSueCreekHeader));
        dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.ModelSize);
    } else {
        gnadevice->dumpXnnNoMmu(modelId, dumpStream);
    }
    gnadevice->releseModel(modelId);
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
        if (inputLayout != Layout::NC && inputLayout != Layout::CN && inputLayout != NCHW && inputLayout != CHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::NC or Layout::CN or Layout::CHW or Layout::NCHW, but was: "
                                << input.second->getTensorDesc().getLayout();
        }
        if (inputLayout == NCHW || inputLayout == CHW) {
            inputLayout = NC;
        }
        auto is2D = input.second->getTensorDesc().getLayout() == Layout::NC || input.second->getTensorDesc().getLayout() == Layout::CN;
        auto is3D = input.second->getTensorDesc().getLayout() == Layout::CHW;

        if (!inputsDesc->ptr_inputs_global_id.count(input.first)) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (inputsDesc->get_ptr_inputs_global(input.first)[idx] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << idx << " not set";
        }

        if (inputsDesc->orientation_in[input.first] == kDnnUnknownOrientation) {
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

        ImportFrames(inputsDesc->get_ptr_inputs_global(input.first)[idx],
                     input.second->cbuffer().as<float *>(),
                     input.second->getTensorDesc().getPrecision(),
                     gnaFlags->sw_fp32 ? 1.0f : inputsDesc->inputScaleFactors[inputNum],
                     inputsDesc->orientation_in[input.first],
                     dims[0],
                     is2D ? dims[dims.size() - 2] : dims[0],
                     is2D ? dims[dims.size() - 1] : is3D ? dims[dims.size() - 1] * dims[dims.size() - 2] :  dims[dims.size() - 1] * dims[dims.size() - 2] * dims[dims.size() - 3] ,
                     is2D ? dims[dims.size() - 1] : is3D ? dims[dims.size() - 1] * dims[dims.size() - 2] :  dims[dims.size() - 1] * dims[dims.size() - 2] * dims[dims.size() - 3]);

	bool isOneChannel;
	is3D ? isOneChannel = input.second->getTensorDesc().getDims()[0] == 1 : input.second->getTensorDesc().getDims()[1] == 1;
        if (((inputLayout == Layout::NC || inputLayout == Layout::NCHW)
            != (inputsDesc->orientation_in[input.first] == kDnnInterleavedOrientation))
            && !isOneChannel) {
            RotateFeatures(reinterpret_cast<uint8_t *>(inputsDesc->get_ptr_inputs_global(input.first)[idx]),
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
        dnn->Propagate();
        if (freeNnet != nnets.end()) {
            std::get<1>(*freeNnet) = 1;
        }
    } else {
#if GNA_LIB_VER == 1
        auto nnet = std::get<0>(*freeNnet).get();
        std::get<1>(*freeNnet) = gnadevice->propagate(&nnet->obj, ptr_active_indices, num_active_indices);
#else
        const auto reqConfigId = std::get<0>(*freeNnet);
        if (ptr_active_indices != nullptr && num_active_indices > 0 && activeLayerIndex != 0xffffffff)
            gnadevice->setUpActiveList(reqConfigId, activeLayerIndex, ptr_active_indices, num_active_indices);
        std::get<1>(*freeNnet) = gnadevice->propagate(reqConfigId);
#endif
    }

#ifdef PLOT
    dnn->BeginNewWrite(dnn_dump_write_index);
    if (dnn->num_components() != 0) {
        dnn->WriteDnnText("/data/local/tmp/Net_.txt", kDnnFloat);
    }
    dnn_dump_write_index++;
#endif
    if (freeNnet != nnets.end()) {
        // TODO: GNA2: Substitute properly when using GNA 2.0 Library setting and CPU
        std::get<2>(*freeNnet) = result;
    }
    return idx;
}

void GNAPlugin::Wait(uint32_t request_idx) {
#if GNA_LIB_VER == 2
    auto& nnets = gnaRequestConfigToRequestIdMap;
#endif
    if (nnets.size() <= request_idx) return;    // TODO: GNA2: check whether necessary
    // already synced TODO: might be copy required ???
    if (std::get<1>(nnets[request_idx]) == -1) return;

    if (gnadevice) {
        gnadevice->wait(std::get<1>(nnets[request_idx]));
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
        if (outputBlob->getTensorDesc().getLayout() == Layout::NC) {
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
            auto& exportOutputDims = outputBlob->getTensorDesc().getDims();
            ExportScores(outputBlob->buffer(),
                         outputDesc.ptrs[request_idx],
                         outputDesc.orientation,
                         exportOutputDims[0],
                         exportOutputDims[exportOutputDims.size() - 2],
                         exportOutputDims[exportOutputDims.size() - 1],
                         exportOutputDims[exportOutputDims.size() - 1],
                         exportOutputDims[exportOutputDims.size() - 1],
                         outputDesc.num_bytes_per_element,
                         sizeof(float));
        } else if (outputBlob->getTensorDesc().getLayout() != Layout::CN) {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::NC or Layout::CN. But was "
                << outputBlob->getTensorDesc().getLayout();
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
            auto dims = outputBlob->getTensorDesc().getDims();
            for (int i = 0; i < dims[dims.size() - 2]; i++) {
                for (int j = 0; j < dims[dims.size() - 1]; j++) {
                    fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t *>()[dims[dims.size() - 1] * i + j]);
                }
                fprintf(f, "\n");
                }
                fprintf(f, "\n\n");
            }
#endif
            ConvertToFloat(outputBlob->buffer(),
                           outputBlob->buffer(),
                           outputBlob->getTensorDesc().getDims()[outputBlob->getTensorDesc().getDims().size() - 1],
                           outputBlob->getTensorDesc().getDims()[outputBlob->getTensorDesc().getDims().size() - 2],
                           outputDesc.scale_factor);
#ifdef PLOT
        if (f) {
            auto dims = outputBlob->getTensorDesc().getDims();
            for (int i = 0; i < dims[dims.size() - 2]; i++) {
                for (int j = 0; j < dims[dims.size() - 1]; j++) {
                    fprintf(f, "%.2f ", outputBlob->cbuffer().as<float *>()[dims[dims.size() - 1] * i + j]);
                }
                fprintf(f, "\n");
                }
                fclose(f);
            }
#endif
        }
        output_idx++;
    }
}

void GNAPlugin::Reset() {
    graphCompiler.Reset();
}

void GNAPlugin::Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &output) {
    BlobMap bmInput;
    BlobMap bmOutput;
    if (inputsDataMap.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"<< "model accepts " << inputsDataMap.size() << " inputs";
    }

    IE_ASSERT(!inputsDataMap.empty());
    bmInput[inputsDataMap.begin()->first] = std::shared_ptr<Blob>(const_cast<Blob*>(&input), [](Blob*){});
    IE_ASSERT(!outputsDataMap.empty());
    bmOutput[outputsDataMap.begin()->first] = std::shared_ptr<Blob>(&output, [](Blob*){});
    Infer(bmInput, bmOutput);
}

void GNAPlugin::Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result) {
    Wait(QueueInference(input, result));
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
    inputBlob = make_blob_with_precision(TensorDesc(precision, inputDims, inputDims.size() == 2 ? NC : inputDims.size() == 3 ? CHW : NCHW));
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::MemoryStateInternal::Ptr>  GNAPlugin::QueryState() {
    if (graphCompiler.memory_connection.empty()) {
        return {};
    }

    return graphCompiler.memoryStates;
}

std::string GNAPlugin::GetName() const noexcept {
    return _pluginName;
}

void GNAPlugin::SetName(const std::string & pluginName) noexcept {
    _pluginName = pluginName;
}

InferenceEngine::IExecutableNetwork::Ptr GNAPlugin::ImportNetwork(const std::string &modelFileName) {
    // no need to return anything dueto weird design of internal base classes
    std::fstream inputStream(modelFileName, ios_base::in | ios_base::binary);
    if (inputStream.fail()) {
        THROW_GNA_EXCEPTION << "Cannot open file to import model: " << modelFileName;
    }

    auto header = GNAModelSerial::ReadHeader(inputStream);

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
    serial.Import(basePtr, header.gnaMemSize, inputStream);

    inputsDesc->get_ptr_inputs_global("input").push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + header.input.descriptor_offset));
    // TODO: import of multioutput network not supported
    outputsDesc.resize(1);
    auto &outputDesc = outputsDesc.front();
    outputDesc.ptrs.push_back(reinterpret_cast<float*>(reinterpret_cast<uint8_t *> (basePtr) + header.output.descriptor_offset));

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

#if GNA_LIB_VER == 2
    inputsDesc->orientation_in["input"] = getOrientation(std::get<0>(gnaModels.back())->obj.Operations[0]);
    outputDesc.orientation = getOrientation(std::get<0>(gnaModels.back())->obj.Operations[std::get<0>(gnaModels.back())->obj.NumberOfOperations - 1]);
#else
    inputsDesc->orientation_in["input"] = getOrientation(std::get<0>(nnets.back())->obj.pLayers[0]);
    outputDesc.orientation = getOrientation(std::get<0>(nnets.back())->obj.pLayers[std::get<0>(nnets.back())->obj.nLayers - 1]);
#endif
    outputDesc.num_bytes_per_element = header.output.element_size;

    auto outputDims = SizeVector({header.nGroup, header.output.elements_count / header.nGroup});
    auto inputDims = SizeVector({header.nGroup, header.input.elements_count / header.nGroup});

    inputsDataMap["input"] = std::make_shared<InputInfo>();
    inputsDataMap["input"]->setInputData(make_shared<Data>("input",
                                                           TensorDesc(
                                                                   Precision::FP32,
                                                                   inputDims,
                                                                   Layout::NC)));
    outputsDataMap["output"] = make_shared<Data>("output",
                                                 TensorDesc(
                                                         Precision::FP32,
                                                         outputDims,
                                                         Layout::NC));

    outputDesc.scale_factor = header.output.scaleFactor;
    inputsDesc->inputScaleFactors.push_back(header.input.scaleFactor);

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
    dnn->WriteGraphWizModel("gna-blob.dot");
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

    if (inputsDesc->ptr_inputs_global_id.size() != 1) {
        THROW_GNA_EXCEPTION << " exporting network with multiple inputs not supported";
    }

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
    auto serial = GNAModelSerial(&std::get<0>(gnaModels.front())->obj,
#else
    auto serial = GNAModelSerial(&std::get<0>(nnets.front())->obj,
#endif
                   {inputsDesc->inputScaleFactors.front(),
                    inputsDesc->ptr_inputs_global_storage.front()[0],
                    2,
                    static_cast<uint32_t>(InferenceEngine::details::product(inputsDataMap.begin()->second->getTensorDesc().getDims()))},
                   {outputsDesc.front().scale_factor,
                    outputsDesc.front().ptrs.front(),
                    outputsDesc.front().num_bytes_per_element,
                    static_cast<uint32_t>(InferenceEngine::details::product(outputsDataMap.begin()->second->getTensorDesc().getDims()))})
        .SetInputRotation(dnn->num_rotate_rows, dnn->num_rotate_columns);

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

void GNAPlugin::SetConfig(const std::map<std::string, std::string> &config) {
    Init();
    auto supportedConfigOptions = supportedConfigKeys();

    for (auto& item : config) {
        auto keys = std::find_if(supportedConfigOptions.begin(), supportedConfigOptions.end(), [&item](const std::string& supportedConfigOption) {
            return item.first.find(supportedConfigOption) != std::string::npos;
        });
        if (keys == supportedConfigOptions.end()) {
            THROW_GNA_EXCEPTION << as_status << NOT_FOUND << "Incorrect GNA Plugin config. Key " << item.first << " not supported";
        }
    }

    // holds actual value of a found key
    std::string key;
    std::string value;
    auto if_set = [&](const std::string& keyInput, const std::function<void()> & handler) {
        auto keyInMap = config.find(keyInput);
        if (keyInMap != config.end()) {
            value = keyInMap->second;
            handler();
        }
    };

    auto if_start = [&](const std::string& keyInput, const std::function<void()> & handler) {
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
        uint64_t scaleForInput = std::stoul(key, NULL, 10);
        if (scaleForInput > 10) {
            THROW_GNA_EXCEPTION << "input scale factor with index(" << key << ") unsupported";
        }
        auto scaleFactor = std::stod(value);
        if (fp32eq(scaleFactor, 0.0f)) {
            THROW_GNA_EXCEPTION << "input scale factor of 0.0f not supported";
        }
        // not appeared scale factors are to be 1.0f
        if (inputsDesc->inputScaleFactors.size() <= scaleForInput) {
            inputsDesc->inputScaleFactors.resize(scaleForInput + 1, 1.f);
        }
        inputsDesc->inputScaleFactors[scaleForInput] = InferenceEngine::CNNLayer::ie_parse_float(value);
    });

    if (inputsDesc->inputScaleFactors.empty()) {
        if_set(GNA_CONFIG_KEY(SCALE_FACTOR), [&] {
            auto scaleFactor = InferenceEngine::CNNLayer::ie_parse_float(value);
            if (fp32eq(scaleFactor, 0.0f)) {
                THROW_GNA_EXCEPTION << "input scale factor of 0.0f not supported";
            }
            inputsDesc->inputScaleFactors.push_back(scaleFactor);
        });
    }

    if (inputsDesc->inputScaleFactors.empty()) {
        inputsDesc->inputScaleFactors.push_back(1.f);
    }

    if_set(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), [&] {
        dumpXNNPath = value;
    });

    if_set(GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION), [&] {
        dumpXNNGeneration = value;
    });

    if_set(GNA_CONFIG_KEY(DEVICE_MODE), [&] {
#if GNA_LIB_VER == 1
        static caseless_unordered_map <std::string, uint32_t> supported_values = {
                {GNAConfigParams::GNA_AUTO, GNA_AUTO},
                {GNAConfigParams::GNA_HW, GNA_HARDWARE},
                {GNAConfigParams::GNA_SW, GNA_SOFTWARE},
                {GNAConfigParams::GNA_SW_EXACT, GNA_SOFTWARE & GNA_HARDWARE}
        };
        static std::vector <std::string> supported_values_on_gna2 = {
            GNAConfigParams::GNA_GEN,
            GNAConfigParams::GNA_GEN_EXACT,
            GNAConfigParams::GNA_SSE,
            GNAConfigParams::GNA_SSE_EXACT,
            GNAConfigParams::GNA_AVX1,
            GNAConfigParams::GNA_AVX1_EXACT,
            GNAConfigParams::GNA_AVX2,
            GNAConfigParams::GNA_AVX2_EXACT
        };
#else
        static caseless_unordered_map <std::string, std::pair<Gna2AccelerationMode, Gna2DeviceVersion> > supported_values = {
            {GNAConfigParams::GNA_AUTO, {Gna2AccelerationModeAuto, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_HW, {Gna2AccelerationModeHardware, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_SW, {Gna2AccelerationModeSoftware, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_SW_EXACT, {Gna2AccelerationModeSoftware, Gna2DeviceVersion1_0}},
            {GNAConfigParams::GNA_GEN, {Gna2AccelerationModeGeneric, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_GEN_EXACT, {Gna2AccelerationModeGeneric, Gna2DeviceVersion1_0}},
            {GNAConfigParams::GNA_SSE, {Gna2AccelerationModeSse4x2, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_SSE_EXACT, {Gna2AccelerationModeSse4x2, Gna2DeviceVersion1_0}},
            {GNAConfigParams::GNA_AVX1, {Gna2AccelerationModeAvx1, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_AVX1_EXACT, {Gna2AccelerationModeAvx1, Gna2DeviceVersion1_0}},
            {GNAConfigParams::GNA_AVX2, {Gna2AccelerationModeAvx2, Gna2DeviceVersionSoftwareEmulation}},
            {GNAConfigParams::GNA_AVX2_EXACT, {Gna2AccelerationModeAvx2, Gna2DeviceVersion1_0}},
        };
#endif
        auto procType = supported_values.find(value);
        if (procType == supported_values.end()) {
            if (value == GNA_CONFIG_VALUE(SW_FP32)) {
                gnaFlags->sw_fp32 = true;
            } else {
#if GNA_LIB_VER == 1
                auto is_gna2_mode = std::find(
                        supported_values_on_gna2.begin(),
                        supported_values_on_gna2.end(),
                        value);
                if (is_gna2_mode != supported_values_on_gna2.end()) {
                    THROW_GNA_EXCEPTION << "This GNA device mode require GNA2 library: " << value;
                }
#endif
                THROW_GNA_EXCEPTION << "GNA device mode unsupported: " << value;
            }
        } else {
#if GNA_LIB_VER == 1
            gna_proc_type = static_cast<intel_gna_proc_t>(procType->second);
#else
            pluginGna2AccMode = procType->second.first;
            pluginGna2DeviceConsistent = procType->second.second;
#endif
        }
    });

    if_set(GNA_CONFIG_KEY(COMPACT_MODE), [&] {
        if (value == PluginConfigParams::YES) {
            gnaFlags->compact_mode = true;
        } else if (value == PluginConfigParams::NO) {
            gnaFlags->compact_mode = false;
        } else {
            log << "GNA compact mode should be YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "GNA compact mode should be YES/NO, but not" << value;
        }
    });

    if_set(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), [&] {
        if (value == PluginConfigParams::YES) {
            gnaFlags->exclusive_async_requests  = true;
        } else if (value == PluginConfigParams::NO) {
            gnaFlags->exclusive_async_requests  = false;
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
            gnaFlags->uniformPwlDesign = true;
        } else if (value == PluginConfigParams::NO) {
            gnaFlags->uniformPwlDesign = false;
        } else {
            log << "GNA pwl uniform algorithm parameter "
                << "should be equal to YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "GNA pwl uniform algorithm parameter "
                                << "should be equal to YES/NO, but not" << value;
        }
    });

    if_set(CONFIG_KEY(PERF_COUNT), [&] {
        if (value == PluginConfigParams::YES) {
            gnaFlags->performance_counting = true;
        } else if (value == PluginConfigParams::NO) {
            gnaFlags->performance_counting = false;
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
        gnaFlags->gna_lib_async_threads_num = lib_threads;
    });

    if_set(CONFIG_KEY(SINGLE_THREAD), [&] {
        if (value == PluginConfigParams::YES) {
            gnaFlags->gna_openmp_multithreading  = false;
        } else if (value == PluginConfigParams::NO) {
            gnaFlags->gna_openmp_multithreading  = true;
        } else {
            log << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
            THROW_GNA_EXCEPTION << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
        }
    });

    if (gnaFlags->sw_fp32 && gnaFlags->gna_lib_async_threads_num > 1) {
        THROW_GNA_EXCEPTION << "GNA plugin not support async mode on GNA_SW_FP32!";
    }
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
                                           [&](CNNLayerPtr const& layer) {
                                                if (LayerTypeFromStr(layer->type) != LayerType::NO_TYPE) {
                                                    res.supportedLayersMap.insert({ layer->name, GetName() });
                                                }
                                            }, false);
    }
