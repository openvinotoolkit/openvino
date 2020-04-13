// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <list>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "descriptions/gna_input_desc.hpp"
#include "descriptions/gna_flags.hpp"
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "connection_details.hpp"
#include "backend/dnn.hpp"
#include "memory/polymorph_allocator.hpp"
#include "memory/gna_memory.hpp"
#include "layers/gna_memory_layer.hpp"
#include "layers/gna_concat_layer.hpp"
#include "layers/gna_crop_layer.hpp"
#include "layers/gna_split_layer.hpp"
#include "backend/dnn_components.hpp"
#include "backend/am_intel_dnn.hpp"
#include "gna_device.hpp"
#include "gna_data_types.hpp"

namespace GNAPluginNS {
class GNAGraphCompiler {
private:
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn;
    std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem;
    std::shared_ptr<GNAPluginNS::InputDesc> inputDesc;
    std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlags;

    // layers with extra storage for connections and additional
    // non trivial processing

    SplitConnection  split_connection;
    CropConnection   crop_connection;
    ConstConnections const_connections;

    intel_dnn_component_t * find_first_unused_input(InferenceEngine::CNNLayerPtr current);

public:
    GNAPluginNS::backend::DnnComponents dnnComponents;
    MemoryConnection memory_connection;
    ConcatConnection concat_connection;

    void setGNAMemoryPtr(std::shared_ptr<GNAPluginNS::gna_memory_type> gnaMemPtr);
    void setDNNPtr(std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnnPtr);
    void setInputDescPtr(std::shared_ptr<GNAPluginNS::InputDesc> inputDescPtr);
    void setGNAFlagsPtr(std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlagsPtr);

    void fillMemoryConnections(std::unordered_map<std::string,
            std::vector<InferenceEngine::CNNLayerPtr>> &memoryPairs);

    void fillConcatConnections(InferenceEngine::CNNLayerPtr layer);
    void fillSplitConnections(InferenceEngine::CNNLayerPtr layer);

    /**
    * Connects either memory output, or generic output to a layer
     * @param layer - layer pointer
     * @param ptr_outputs - pointer to pointer where to store  output layer information
     * @param ptr_inputs - sizeof output blob
     * @param sz - sizeof output blob
     * @param ptr_inputs - sizeof output blob
     */
    void connectOutput(InferenceEngine::CNNLayerPtr layer, void *ptr_outputs, size_t sz);
    /**
     * Connects certain input to this layer
     * @param layer - layer that we connect input to
     * @param pVoid - pointer that  holds current layer pointer in gna_mem request
     * @param num_data_bytes_in - size
     * @param offset - num bytes to advance in buffer
     * @param idx - index of input port that we are connecting
     * @return layer used as input
     */
    GNAPluginNS::ConnectionDetails connectInput(InferenceEngine::CNNLayerPtr layer,
                                                void *pVoid,
                                                size_t num_data_bytes_in,
                                                int32_t offset = 0,
                                                int idx = 0);

    /**
     * Fill in the Affine layer weights
    * @param layer - affine layer pointer
    * @param ptrWeights - pointer to weights memory
    * @param offset - memory before offset value will be zeroed
    * @param isQuantized - information about layer quantization
    */
    void FillWeightOfAligningFilter(InferenceEngine::CNNLayerPtr layer, void* ptrWeights, size_t offset, bool isQuantized = false);

    void CreateLayerPrimitive(InferenceEngine::CNNLayerPtr);

    void AffinePrimitive(InferenceEngine::CNNLayerPtr, bool isDiag = false);
    void AffineFilterPrimitive(InferenceEngine::CNNLayerPtr);
    void ConcatAlignFilterPrimitive(InferenceEngine::CNNLayerPtr);
    void DiagonalPrimitive(InferenceEngine::CNNLayerPtr);
    void ConstPrimitive(InferenceEngine::CNNLayerPtr);
    void ConvolutionPrimitive(InferenceEngine::CNNLayerPtr);
    void PermutePrimitive(InferenceEngine::CNNLayerPtr);
    void PoolingPrimitive(InferenceEngine::CNNLayerPtr);
    void PowerPrimitive(InferenceEngine::CNNLayerPtr);
    void ConcatPrimitive(InferenceEngine::CNNLayerPtr);
    void CropPrimitive(InferenceEngine::CNNLayerPtr);
    void EltwisePrimitive(InferenceEngine::CNNLayerPtr);
    void SplitPrimitive(InferenceEngine::CNNLayerPtr);
    void SlicePrimitive(InferenceEngine::CNNLayerPtr);
    void PWLPrimitive(InferenceEngine::CNNLayerPtr);
    void CopyPrimitive(InferenceEngine::CNNLayerPtr);

    void Reset();
};
}  // namespace GNAPluginNS
