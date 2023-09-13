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
#include "backend/gna_limitations.hpp"
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
#include "gna_plugin_config.hpp"
#include "gna_tensor_tools.hpp"
#include "gna_transformations_pipeline.hpp"
#include "layers/gna_layer_type.hpp"
#include "log/log.hpp"
#include "memory/gna_memory_state.hpp"
#include "orientation_helper.hpp"
#include "pre_post_process/converter_factory.hpp"
#include "pre_post_process/transposition_info.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/worker_factory.hpp"
#include "request/worker_pool_impl.hpp"
#include "runtime/gna_float_runtime.hpp"
#include "scale_factor_helper.hpp"
#include "serial/gna_model_serial.hpp"

using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::graph_utils;

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
using namespace ov::intel_gna::pre_post_processing;

void GNAPlugin::PrePostProcess(InferenceEngine::Blob::Ptr input_blob,
                               InferenceEngine::Blob::Ptr output_blob,
                               std::shared_ptr<ov::Model> model) {
    const ov::element::Type input_type = details::convertPrecision(input_blob->getTensorDesc().getPrecision());
    const ov::element::Type output_type = details::convertPrecision(output_blob->getTensorDesc().getPrecision());
    const ov::Shape& output_shape = output_blob->getTensorDesc().getDims();

    for (const auto& param : model->get_parameters()) {
        param->set_element_type(input_type);
    }
    model->validate_nodes_and_infer_types();
    const ov::Shape& input_shape = model->get_parameters()[0]->get_output_shape(0);

    ov::TensorVector inputs = {ov::Tensor(input_type, input_shape, input_blob->cbuffer().as<void*>())};
    ov::TensorVector results = {ov::Tensor(output_type, output_shape, output_blob->buffer().as<void*>())};

    if (!model->evaluate(results, inputs)) {
        THROW_GNA_EXCEPTION << "Failed to evaluate model " << model->get_friendly_name() << std::endl;
    }
}

GNAPlugin::GNAPlugin() {
    Init();
    UpdateFieldsFromConfig();
    InitGNADevice();
    Limitations::init(config.target->get_effective_compile_target());
    InitGNAMemory();
    InitGraphCompiler();
    m_input_output_handler = InputOutputDataHandler(ConverterFactory::create_converter());
}

GNAPlugin::GNAPlugin(const std::map<std::string, std::string>& configMap) {
    Init();
    SetConfig(configMap);
    log::set_log_level(gnaFlags->log_level);
    InitGNADevice();
    Limitations::init(config.target->get_effective_compile_target());
    InitGNAMemory();
    InitGraphCompiler();
    m_input_output_handler = InputOutputDataHandler(ConverterFactory::create_converter());
}

void GNAPlugin::Init() {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "Init");
    dnn = std::make_shared<backend::AMIntelDNN>(backend::AMIntelDNN());
    gnaFlags = std::make_shared<GNAFlags>(GNAFlags());
    inputs_ptr_ = std::make_shared<GnaInputs>(GnaInputs());
    outputs_ = GnaOutputs();
    requestWorkerPool_ = std::make_shared<request::WorkerPoolImpl>();
}

void GNAPlugin::InitGNADevice() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InitGNADevice");

    if (!gnaFlags->sw_fp32) {
        gnadevice = std::make_shared<GNADeviceHelper>(config.target,
                                                      gnaFlags->performance_counting,
                                                      !config.embedded_export_path.empty());
    }
}

void GNAPlugin::InitGNAMemory() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InitGNAMemory");

    if (gnaFlags->sw_fp32) {
        gnamem.reset(new gna_memory_float(memory::GNAFloatAllocator{}));
    } else {
        gnamem = std::make_shared<gna_memory_device>(memory::GNAAllocator(gnadevice),
                                                     Limitations::get_instance()->get_memory_alignment(),
                                                     Limitations::kMemoryPageSize);
    }
}

void GNAPlugin::InitGraphCompiler() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InitGraphCompiler");

    m_graph_compiler = std::make_shared<GNAGraphCompiler>(
        GNAGraphCompiler(config, dnn, inputs_ptr_, Limitations::get_instance()->get_cnn_validator(), gnamem));
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
            CNNLayerPtr next_layer = nextToInputLayer.second;
            // FQ layer can be connected to the input via Reshape/Transpose/Gather and other non-functional layers.
            if (LayerInfo(next_layer).is_fq_non_sensitive()) {
                next_layer = CNNNetCheckNextLayerSkipCertain(nextToInputLayer.second, 0, 0, true, [](CNNLayerPtr l) {
                                 return LayerInfo(l).is_fq_non_sensitive();
                             }).first;
            }

            if (!LayerInfo(next_layer).isFakeQuantize()) {
                continue;
            }

            // replacing scale factor from this fq layer
            GNAFakeQuantizeLayer fqLayer(next_layer);
            auto inputRange = fqLayer.getInputRange();
            auto outputRange = fqLayer.getOutputRange();
            if (inputRange.second.size() != 1 || outputRange.second.size() != 1) {
                THROW_GNA_LAYER_EXCEPTION(next_layer)
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

        // find pre-processing model
        auto subgraph_it = m_input_output_subgraphs.find(ie_name);
        if (subgraph_it != m_input_output_subgraphs.end()) {
            (*inputs_ptr_)[ie_name].pre_post_process_model = subgraph_it->second;
        }
    }
}

void GNAPlugin::UpdateOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateOutputs");
    for (const auto& result : results) {
        const std::string ie_name = ov::op::util::create_ie_output_name(result->input_value(0));
        outputs_[ie_name].name = ie_name;
        outputs_[ie_name].tensor_names = result->get_output_tensor(0).get_names();

        // find postprocessing model
        auto subgraph_it = m_input_output_subgraphs.find(ie_name);
        if (subgraph_it != m_input_output_subgraphs.end()) {
            outputs_[ie_name].pre_post_process_model = subgraph_it->second;
        }
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
        outputs_.at(portName).set_precision(static_cast<uint32_t>(numBytesPerElem));
        outputs_.at(portName).scale_factor =
            quantized != nullptr ? quantized->_dst_quant.GetScale() : kScaleFactorDefault;
        outputs_.at(portName).num_elements = static_cast<uint32_t>(numElem);

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->getQueue(REGION_AUTO)->bind_ptr(layer, &outputs_.at(portName).ptrs.front(), outputPtr);
    };

    // probing gna_primitives
    auto irLayerAvatar = std::find_if(m_graph_compiler->dnnComponents.components.begin(),
                                      m_graph_compiler->dnnComponents.components.end(),
                                      [&layer](const backend::DnnComponents::storage_type::value_type& value) {
                                          return value.name == layer->name;
                                      });
    if (irLayerAvatar != m_graph_compiler->dnnComponents.components.end()) {
        initOutput(irLayerAvatar->dnnComponent.orientation_out,
                   irLayerAvatar->dnnComponent.num_bytes_per_output,
                   irLayerAvatar->dnnComponent.num_rows_out,
                   &irLayerAvatar->dnnComponent.ptr_outputs);
        return true;
    }

    // probing concatInfo
    if (LayerInfo(layer).isConcat()) {
        auto concatConnection = m_graph_compiler->concat_connection.find(layer->name);
        if (concatConnection != m_graph_compiler->concat_connection.end()) {
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
        auto constConnection = m_graph_compiler->const_connections.find(layer->name);
        if (constConnection != m_graph_compiler->const_connections.end()) {
            initOutput(kDnnInterleavedOrientation,
                       layer->outData.front()->getPrecision().size(),
                       const_blob->size(),
                       &constConnection->second);
            return true;
        }
    }

    return false;
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

    auto transformer = TransformationsPipeline(config);

    if (_network.getFunction()) {
        CNNNetwork clonedNetwork = InferenceEngine::cloneNetwork(_network);
        auto model = clonedNetwork.getFunction();
        transformer.apply(model, &m_input_output_subgraphs);
        Limitations::get_instance()->check_all_ops_supported(model, config.gnaPrecision);
        convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(model, clonedNetwork);
    }
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetwork network = convertedNetwork ? InferenceEngine::CNNNetwork{convertedNetwork} : _network;
    IE_SUPPRESS_DEPRECATED_END

    transformer.convert_precision_legacy(network);

    //  Check the network

    std::string error;
    if (!Limitations::are_layers_supported(network, error)) {
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
            m_graph_compiler->fillConcatConnections(layer);
        } else if (layerInfo.isSplit() || layerInfo.isSlice()) {
            m_graph_compiler->fillSplitConnections(layer);
        }
        sortedNoMem.push_back(layer);
    }

    // fill in extra storage with memory layers
    m_graph_compiler->fillMemoryConnections(memoryPairs);

    if (!m_graph_compiler->memory_connection.empty() && gnaFlags->num_requests != 1) {
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
        m_graph_compiler->CreateLayerPrimitive(layer);
    }

    for (auto& inputLayer : inputLayers) {
        auto layerInfo = LayerInfo(inputLayer);
        if (layerInfo.isInput() && 0 == inputs_ptr_->at(inputLayer->name).get_allocated_size()) {
            m_graph_compiler->connectOutput(inputLayer, &inputs_ptr_->at(inputLayer->name).ptrs.front(), 0);
        }
    }

    if (m_graph_compiler->dnnComponents.components.empty()) {
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
            for (auto&& memConnection : m_graph_compiler->memory_connection) {
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

    void* pParallelExecutionData = nullptr;

    // reserving more bytes for intermediate data in parallel case
    // TODO: this works incorrectly in compact mode at lest
    rwSegmentSize = gnamem->getRegionBytes(REGION_SCRATCH);
    rwSegmentSize += gnamem->getRegionBytes(REGION_INPUTS);
    rwSegmentSize += gnamem->getRegionBytes(REGION_OUTPUTS);
    if (gnaFlags->num_requests > 1) {
        gnamem->getQueue(REGION_SCRATCH)
            ->reserve_ptr(nullptr, &pParallelExecutionData, rwSegmentSize * (gnaFlags->num_requests - 1));
    }

    gnamem->commit(gnaFlags->compact_mode);

    dnn->Init(gnamem.get(), gnaFlags->sw_fp32 ? kDnnFloat : kDnnInt, 1);

    // TODO: this copy is unneeded; in fact, we can directly create gna structs from list
    auto execOrder = m_graph_compiler->dnnComponents.getExecutionOrder();
    dnn->component.insert(dnn->component.begin(), execOrder.begin(), execOrder.end());

    // in fp32 mode last PWL cannot be computed without that
    if (!m_graph_compiler->dnnComponents.components.empty()) {
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
        for (uint32_t j = 0; j != model->NumberOfOperations; j++) {
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
                                                                                  m_graph_compiler->dnnComponents,
                                                                                  *inputs_ptr_);
        }
    }

    // update orientation of model output layer
    for (auto&& outPort : outputs_data_map_) {
        auto outLayer = getCreatorLayer(outPort.second).lock();
        if (outLayer && LayerInfo(outLayer).isOutput()) {
            ov::intel_gna::helpers::updateModelOutputOrientation(outPort.first,
                                                                 outLayer->name,
                                                                 m_graph_compiler->dnnComponents,
                                                                 outputs_);
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

    if (config.target->get_effective_compile_target() == target::DeviceVersion::GNAEmbedded1_0) {
        auto dump = gnadevice->dumpXnn(modelId);
        dump.header.RwRegionSize = static_cast<uint32_t>(gnamem->getRegionBytes(REGION_SCRATCH));
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
    if (config.GetParameter(ov::intel_gna::execution_mode.name()).as<std::string>() == "GNA_HW" &&
        !gnadevice->isHwAvailable()) {
        THROW_GNA_EXCEPTION << "Execution mode GNA_HW is set, but hardware acceleration is unavailable";
    }
    auto freeWorker = requestWorkerPool_->findFreeModelWorker();
    if (freeWorker == nullptr) {
        if (!m_graph_compiler->memory_connection.empty()) {
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
        std::string input_name = input.first;
        InferenceEngine::Layout input_layout = input.second->getTensorDesc().getLayout();

        if (input_layout != InferenceEngine::Layout::C && input_layout != InferenceEngine::Layout::NC &&
            input_layout != InferenceEngine::Layout::CN && input_layout != InferenceEngine::Layout::CHW &&
            input_layout != InferenceEngine::Layout::NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or "
                                   "Layout::CHW. But was: "
                                << input_layout;
        }

        if (input_layout == InferenceEngine::Layout::NCHW || input_layout == InferenceEngine::Layout::CHW) {
            // specific case that can be squeezed to 2d
            input_layout = InferenceEngine::Layout::NC;
        }

        auto is1D = input_layout == InferenceEngine::Layout::C;
        auto is3D = input_layout == InferenceEngine::Layout::CHW;

        if (inputs_ptr_->at(input_name).ptrs.empty()) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input_name << " not set";
        }

        if (inputs_ptr_->at(input_name).ptrs[index] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input_name << " at inferRequest #"
                                << index << " not set";
        }
        const auto inputOrientation = inputs_ptr_->at(input_name).orientation;
        if (inputOrientation == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration
            // of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input orientation for " << input_name << " not set";
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

        if (inputs_ptr_->at(input_name).get_required_size() < importedBytes) {
            THROW_GNA_EXCEPTION << "Cannot import input frames for :" << input_name
                                << ", allocated size: " << inputs_ptr_->at(input_name).get_required_size()
                                << ", but input blob size: " << importedBytes;
        }

        // Perform pre-processing on CPU.
        // When we need to perform pre-processing on CPU using ngraph model we copy user input to the buffer,
        // then set preprocessing output blob as gna input blob.
        std::shared_ptr<ov::Model> model = inputs_ptr_->at(input_name).pre_post_process_model;
        Blob::Ptr buff_blob = nullptr;
        TensorDesc buff_tensor_desc(input.second->getTensorDesc());
        buff_tensor_desc.setPrecision(inputs_ptr_->at(input_name).tensor_precision);

        if (model) {
            // WA: evaluate gather with int16 precision as fp16
            if (buff_tensor_desc.getPrecision() == Precision::I16) {
                buff_tensor_desc.setPrecision(Precision::FP16);
            }
            buff_blob = make_blob_with_precision(buff_tensor_desc);
            buff_blob->allocate();
        } else {
            buff_blob = make_blob_with_precision(buff_tensor_desc, inputs_ptr_->at(input_name).ptrs[index]);
        }

        m_input_output_handler.import_frames(
            buff_blob->buffer(),
            input.second->cbuffer().as<float*>(),
            input.second->getTensorDesc().getPrecision(),
            gnaFlags->sw_fp32 ? kScaleFactorDefault : inputs_ptr_->at(input_name).scale_factor,
            inputOrientation,
            importedFrames,
            targetGroups,
            importedElements,
            importedElements,
            gnaFlags->input_low_precision,
            gnadevice.get() != nullptr);

        if (model) {
            Precision output_prc = buff_blob->getTensorDesc().getPrecision();
            SizeVector output_dims = model->get_result()->get_shape();
            TensorDesc output_desc(output_prc, output_dims, InferenceEngine::Layout::ANY);
            Blob::Ptr output_blob = make_blob_with_precision(output_desc, inputs_ptr_->at(input_name).ptrs[index]);
            PrePostProcess(buff_blob, output_blob, model);
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
        const std::string& output_name = outputBlobIt.first;
        Blob::Ptr output_blob = outputBlobIt.second;
        const InferenceEngine::Layout output_layout = output_blob->getTensorDesc().getLayout();

        if (output_layout != InferenceEngine::Layout::C && output_layout != InferenceEngine::Layout::NC &&
            output_layout != InferenceEngine::Layout::CN && output_layout != InferenceEngine::Layout::NCHW &&
            output_layout != InferenceEngine::Layout::CHW && output_layout != InferenceEngine::Layout::SCALAR) {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or "
                                   "Layout::CHW. But was "
                                << output_layout;
        }

        auto dims = output_blob->getTensorDesc().getDims();
        auto is1D = output_layout == InferenceEngine::Layout::C;
        auto isScalar = output_layout == InferenceEngine::Layout::SCALAR;
        auto is3D = output_layout == InferenceEngine::Layout::CHW;
        size_t batchSize = (is1D || isScalar || is3D) ? 1 : dims[0];
        size_t elementsPerBatch =
            isScalar ? 1 : (is1D ? dims.front() : details::product(++std::begin(dims), std::end(dims)));

        OutputDesc& gna_output_desc = outputs_.at(output_name);
        Blob::Ptr gna_output_blob = nullptr;

        // Perform postprocessing on CPU
        std::shared_ptr<ov::Model> model = gna_output_desc.pre_post_process_model;
        if (model) {
            // WA: evaluate gather with int16 precision as fp16
            Precision preproc_prc = (gna_output_desc.tensor_precision == Precision::I16)
                                        ? Precision(Precision::FP16)
                                        : gna_output_desc.tensor_precision;
            const SizeVector& input_dims = model->get_parameters().front()->get_shape();
            TensorDesc input_desc(preproc_prc, input_dims, InferenceEngine::Layout::ANY);
            Blob::Ptr input_blob = make_blob_with_precision(input_desc, gna_output_desc.ptrs[request_idx]);

            const SizeVector& output_dims = model->get_result()->get_shape();
            TensorDesc output_desc(preproc_prc, output_dims, InferenceEngine::Layout::ANY);
            gna_output_blob = make_blob_with_precision(output_desc);
            gna_output_blob->allocate();

            PrePostProcess(input_blob, gna_output_blob, model);
        } else {
            log::debug() << "Postprocessing for output " << output_name << " is not required" << std::endl;
            TensorDesc output_desc(gna_output_desc.tensor_precision,
                                   gna_output_desc.dims,
                                   gna_output_desc.model_layout);
            gna_output_blob = make_blob_with_precision(output_desc, gna_output_desc.ptrs[request_idx]);
        }

        if (!gnadevice) {
            m_input_output_handler.export_scores(output_blob->buffer(),
                                                 gna_output_blob->cbuffer(),
                                                 gna_output_desc.orientation,
                                                 batchSize,
                                                 batchSize,
                                                 elementsPerBatch,
                                                 elementsPerBatch,
                                                 elementsPerBatch,
                                                 gna_output_desc.tensor_precision,
                                                 Precision::I32,
                                                 1.0f);
        } else {
            m_input_output_handler.export_scores(output_blob->buffer(),
                                                 gna_output_blob->cbuffer(),
                                                 gna_output_desc.orientation,
                                                 batchSize,
                                                 batchSize,
                                                 elementsPerBatch,
                                                 elementsPerBatch,
                                                 elementsPerBatch,
                                                 gna_output_desc.tensor_precision,
                                                 gna_output_desc.model_precision,
                                                 gna_output_desc.scale_factor);

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
                    fprintf(f, "%d ", output_blob->cbuffer().as<int32_t*>()[0]);
                } else {
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%d ", output_blob->cbuffer().as<int32_t*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                }
                fprintf(f, "\n\n");
            }
            if (f) {
                if (isScalar) {
                    fprintf(f, "%.7f ", output_blob->cbuffer().as<float*>()[0]);
                } else {
                    auto dims = output_blob->getTensorDesc().getDims();
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%.7f ", output_blob->cbuffer().as<float*>()[dims[dims.size() - 1] * i + j]);
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
    m_graph_compiler->Reset();
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
    if (memoryStates.size() != m_graph_compiler->memory_connection.size()) {
        memoryStates.clear();
        for (auto& connection : m_graph_compiler->memory_connection) {
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

    gnamem->getQueue(REGION_SCRATCH)->reserve_ptr(nullptr, &basePtr, header.gnaMemSize, false);

    gnamem->commit();

    auto model = createModelWrapperForImportNetwork(static_cast<uint32_t>(header.layersCount));
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

    //  Support model versions <= 2.8
    if (!transpose_inputs_info.empty()) {
        ConvertTransposeMapToModel(transpose_inputs_info, inputs_ptr_->Get());
    }
    if (!transpose_outputs_info.empty()) {
        ConvertTransposeMapToModel(transpose_outputs_info, outputs_.Get());
    }

    for (auto&& memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr, gnaFlags->sw_fp32 ? 4 : 2);
        std::string name;
        std::tie(memoryLayer.gna_ptr, memoryLayer.reserved_size, name, memoryLayer.scale_factor) = memory;
        m_graph_compiler->memory_connection.emplace_back(make_pair(name, memoryLayer));
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

    for (auto&& memoryConnection : m_graph_compiler->memory_connection) {
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

    auto model = network.getFunction();
    if (model) {
        auto supported = GetSupportedNodes(
            model,
            [&](std::shared_ptr<ov::Model>& model) {
                TransformationsPipeline(qn_config).apply(model);
            },
            [&](const std::shared_ptr<ngraph::Node>& op) {
                const auto res = Limitations::get_instance()->is_op_supported(op, qn_config.gnaPrecision);
                return res;
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
