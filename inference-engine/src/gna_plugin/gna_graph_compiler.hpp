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
#include "gna_plugin_policy.hpp"
#include "backend/gna_hash_combine.hpp"

namespace GNAPluginNS {
class GNAGraphCompiler {
private:
    std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn;
    std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem;
    std::shared_ptr<GNAPluginNS::InputDesc> inputDesc;
    std::shared_ptr<GNAPluginNS::GNAFlags> gnaFlags;
    Policy policy;

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
    void setPolicy(GNAPluginNS::Policy policy);

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

    // weights, biases storage for gen gna_primitive
    struct WB {
        void *pWeights;
        void *pBiases;
    };
    std::list<WB> weightsHolder;
    std::unordered_map <std::vector<size_t>, std::list<WB>::iterator , hash_combine_t> cachedFilterWeights;
    size_t genFilters = 0;

    /**
     * @brief generates affine filter that implements left-shift of given layers input
     * @param input_sz   - number of input elements
     * @prama left_shift - number of elements filter will do shifting
     * @param output_sz  - number of outputs
     * @param input_data_offset - offset in input_tensor where to start copy
     * @param output_data_offset - offset in otuput_tensor where to copy to
     */
    void genAffineFilter(InferenceEngine::CNNLayerPtr layer,
        size_t input_sz,
        size_t left_shift,
        size_t output_sz,
        size_t input_data_offset,
        size_t output_data_offset);

    void genFilterWeights(size_t input_sz,
                          int left_shift,
                          void *&pWeights,
                          void *&pBiases,
                          float weightScale);

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
    void genPWLPrimitive(int num_rows,
                         int num_columns,
                         InferenceEngine::Precision input_precision,
                         InferenceEngine::Precision output_precision,
                         std::string type,
                         float negative_slope,
                         std::string activationName,
                         float input_pwl_scale_factor,
                         float output_pwl_scale_factor,
                         void *&ptr_inputs,
                         void *&ptr_outputs);

    void CopyPrimitive(InferenceEngine::CNNLayerPtr);

    void Reset();
};
}  // namespace GNAPluginNS
