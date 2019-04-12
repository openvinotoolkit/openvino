// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <list>
#include <set>
#include <unordered_set>
#include <sstream>
#include <CPP/cldnn_defs.h>
#include <CPP/data.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/reorder.hpp>
#include <CPP/convolution.hpp>
#include <CPP/pooling.hpp>
#include <CPP/lrn.hpp>
#include <CPP/fully_connected.hpp>
#include <CPP/softmax.hpp>
#include <CPP/activation.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/proposal.hpp>
#include <CPP/roi_pooling.hpp>
#include <CPP/scale.hpp>
#include <CPP/crop.hpp>
#include <CPP/deconvolution.hpp>
#include <CPP/prior_box.hpp>
#include <CPP/detection_output.hpp>
#include <CPP/normalize.hpp>
#include <CPP/reshape.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/permute.hpp>
#include <CPP/split.hpp>
#include <CPP/upsampling.hpp>
#include <CPP/network.hpp>
#include <CPP/profiling.hpp>
#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/reorg_yolo.hpp>
#include <CPP/region_yolo.hpp>
#include <CPP/mutable_data.hpp>
#include <CPP/max_unpooling.hpp>
#include <CPP/arg_max_min.hpp>
#include <CPP/mvn.hpp>
#include <CPP/tile.hpp>
#include <CPP/border.hpp>
#include <CPP/lstm.hpp>
#include <CPP/gather.hpp>
#include <CPP/depth_to_space.hpp>
#include <CPP/shuffle_channels.hpp>
#include <CPP/strided_slice.hpp>
#include <CPP/reverse_sequence.hpp>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cldnn_graph.h"
#include "simple_math.h"
#include <description_buffer.hpp>
#include <cldnn/cldnn_config.hpp>
#include <graph_tools.hpp>
#include <ie_layers_internal.hpp>
#include <net_pass.h>
#include "cldnn_infer_request.h"
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "details/caseless.hpp"
#include <fstream>
#include <utility>
#include <sys/types.h>
#include <sys/stat.h>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

#ifndef NDEBUG
#include <iostream>
#include <iomanip>
#define THROW_CLDNN_EXCEPTION(desc)\
do { \
InferenceEngineException ex(__FILE__, __LINE__);\
std::cout << desc << "\n---\nException detected at " << __FILE__ << ":" << \
__LINE__ << " (" << __FUNCTION__ << ")\n---\n" << std::endl; THROW_IE_EXCEPTION << desc; } while (0);
#else
#define THROW_CLDNN_EXCEPTION(desc) THROW_IE_EXCEPTION << desc;
#endif  // NDEBUG
#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

namespace CLDNNPlugin {

const cldnn::primitive_id CLDNNGraph::m_preProcessTag("_cldnn_input_preprocess");
const cldnn::primitive_id CLDNNGraph::m_weightsTag("_cldnn_weights");
const cldnn::primitive_id CLDNNGraph::m_biasesTag("_cldnn_biases");
const cldnn::primitive_id CLDNNGraph::m_meanValuesTag("_cldnn_mean_values");
const cldnn::primitive_id CLDNNGraph::m_postProcessTag("_cldnn_output_postprocess");
const cldnn::primitive_id CLDNNGraph::m_scalesTag("_cldnn_scales");
const cldnn::primitive_id CLDNNGraph::m_workaroundTag("_cldnn_workaround");
const cldnn::primitive_id CLDNNGraph::m_preCustomLayerTag("_cldnn_custom_preprocess");
const cldnn::primitive_id CLDNNGraph::m_postCustomLayerTag("_cldnn_custom_postprocess");

static void ValidateLayer(const InferenceEngine::CNNLayerPtr& layer, unsigned inputs) {  // todo: add more checks
    if (inputs && layer->insData.size() != inputs) {
        THROW_CLDNN_EXCEPTION("Invalid number of inputs for layer: " << layer->name);
    }
    if (layer->_fusedWith) {
        THROW_CLDNN_EXCEPTION("Unsupported fuse in layer: " << layer->name << " with: " << layer->_fusedWith->name);
    }
}

static void ValidateEltwiseLayer(const InferenceEngine::CNNLayerPtr& layer) {
    if (layer->_fusedWith) {
        THROW_CLDNN_EXCEPTION("Unsupported fuse in layer: " << layer->name << " with: " << layer->_fusedWith->name);
    }
}

#if defined(_WIN32)
#define mkdir(dir, mode) _mkdir(dir)
#endif

void CLDNNGraph::Config::LoadFromMap(const std::map<std::string, std::string>& configMap) {
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;

        // TODO: refactor if-else to map?
        if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DUMP_KERNELS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                dumpCustomKernels = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                dumpCustomKernels = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
            case 0:
                queuePriority = cldnn::priority_mode_types::disabled;
                break;
            case 1:
                queuePriority = cldnn::priority_mode_types::low;
                break;
            case 2:
                queuePriority = cldnn::priority_mode_types::med;
                break;
            case 3:
                queuePriority = cldnn::priority_mode_types::high;
                break;
            default:
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue priority value: " << uVal;
                break;
            }

        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
            case 0:
                queueThrottle = cldnn::throttle_mode_types::disabled;
                break;
            case 1:
                queueThrottle = cldnn::throttle_mode_types::low;
                break;
            case 2:
                queueThrottle = cldnn::throttle_mode_types::med;
                break;
            case 3:
                queueThrottle = cldnn::throttle_mode_types::high;
                break;
            default:
                THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Unsupported queue throttle value: " << uVal;
                break;
            }
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CLDNNCustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_MODE) == 0) {
            if (val.compare(PluginConfigParams::TUNING_DISABLED) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_disabled;
            } else if (val.compare(PluginConfigParams::TUNING_CREATE) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_tune_and_cache;
            } else if (val.compare(PluginConfigParams::TUNING_USE_EXISTING) == 0) {
                tuningConfig.mode = cldnn::tuning_mode::tuning_use_cache;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported tuning mode value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_TUNING_FILE) == 0) {
            tuningConfig.cache_file_path = val;
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_MEM_POOL) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                memory_pool_on = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                memory_pool_on = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported memory pool flag value: " << val;
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_GRAPH_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                graph_dumps_dir = val;
                mkdir(graph_dumps_dir.c_str(), 0755);
            }
        } else if (key.compare(CLDNNConfigParams::KEY_CLDNN_SOURCES_DUMPS_DIR) == 0) {
            if (!val.empty()) {
                sources_dumps_dir = val;
                mkdir(sources_dumps_dir.c_str(), 0755);
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property value by plugin: " << val;
            }
        } else {
            THROW_IE_EXCEPTION << NOT_FOUND_str << "Unsupported property key by plugin: " << key;
        }
    }
}

void CLDNNGraph::changeInputBatch(size_t batch) {
    m_curBatch = batch;
}

bool CLDNNGraph::CanProcessDynBatch(InferenceEngine::ICNNNetwork &network) const {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return false;

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty())
        return false;

    bool check_result = true;
    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
        auto type = LayerTypeFromStr(layer->type);
        if (SimplerNMS == type ||
            ROIPooling == type ||
            PriorBox == type ||
            DetectionOutput == type ||
            Reshape == type ||
            Permute == type ||
            Flatten == type ||
            Proposal == type ||
            PSROIPooling == type ) {
            check_result = false;
        }

        // check for custom layer
        auto customLayer = m_config.customLayers.find(layer->type);
        if (customLayer != m_config.customLayers.end()) {
            check_result = false;
        }
    }, false);

    return check_result;
}

CLDNNGraph::CLDNNGraph(InferenceEngine::ICNNNetwork& network, const Config& config, int max_batch) : m_config(config),
    m_defaultFormat(cldnn::format::bfyx),
    m_curBatch(-1) {
    m_env.engine = std::make_shared<cldnn::engine>(cldnn::engine_configuration(
        (config.useProfiling || (config.tuningConfig.mode != cldnn::tuning_mode::tuning_disabled)),
        false,
        config.dumpCustomKernels,
        std::string(),
        std::string(),
        true,
        std::string(),
        config.sources_dumps_dir,
        config.queuePriority,
        config.queueThrottle,
        config.memory_pool_on));
#if 0
        m_env.debugOptions.PrintOptions();
#endif
    if (config.exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor(TargetDeviceInfo::name(TargetDevice::eGPU));
    }

    bool res = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    res &= NetPass::UnrollRNN_if(network, [] (RNNCellBase rnn) -> bool {
        if (rnn.clip != 0.0f)
            return true;
        if (rnn.type == "GRUCell" ||
            rnn.type == "GRUSequence" ||
            rnn.type == "RNNCell" ||
            rnn.type == "RNNSequence")
            return true;
        if (!(rnn.type == "LSTMCell" || rnn.type == "LSTMSequence") ||
            rnn.activations == std::vector<std::string>{"sigmoid", "tanh", "tanh"})
            return false;
        return true;
    });

    if (!res)
        THROW_CLDNN_EXCEPTION("Plugin doesn't support Tensor Iterator in pure form. "
                              "No one TI optimization pattern was not applied successfully");

    if (max_batch > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(network)) {
            THROW_CLDNN_EXCEPTION("Such topology cannot be compiled for dynamic batch!");
        }

        // calculate number of networks necessary based on binary log
        unsigned int tmp = max_batch;
        unsigned int mask = 1 << 31;
        unsigned int ldigit = 31;

        while (!(tmp & mask)) {
            mask >>= 1;
            ldigit--;
        }

        m_env.m_bv_sz = ldigit + 1;
    } else {
        m_env.m_bv_sz = 0;
    }

    m_env.m_max_batch = max_batch;

    // Handle workarounds
    char networkName[128] = { 0 };
    network.getName(networkName, 127);
    m_env.debugOptions.EnableWA(networkName);
    m_env.debugOptions.AddTimedEvent("Loading Begin");

    if (max_batch > 1) {
        for (int b = m_env.m_bv_sz - 1; b >= 0; b--) {
            m_topology = std::make_shared<cldnn::topology>(cldnn::topology());
            m_env.network.reset();
            m_env.inputLayouts.clear();
            m_env.outputDims.clear();
            m_env.primitiveIDs.clear();

            changeInputBatch(1 << b);
            Load(network);
            CompileNetwork();
            m_env.batchNetworks.insert(m_env.batchNetworks.begin(), m_env.network);

            m_topology.reset();
            m_env.engine->release_pending_memory();
        }
    } else {
        m_topology = std::make_shared<cldnn::topology>(cldnn::topology());
        Load(network);
        CompileNetwork();
        m_topology.reset();
        m_env.engine->release_pending_memory();
    }

    m_env.debugOptions.AddTimedEvent("Loading", "Loading Begin");
    m_env.debugOptions.PrintTimedEvents();
    m_env.debugOptions.ClearTimedEvents();
}

inline std::string layer_type_name_ID(InferenceEngine::CNNLayer* layer) {
    return layer->type + ":" + layer->name;
}

inline std::string layer_type_name_ID(InferenceEngine::CNNLayerPtr layer) {
    return layer_type_name_ID(layer.get());
}

std::vector<InferenceEngine::CNNLayerPtr> CLDNNGraph::GetNextLayers(const InferenceEngine::DataPtr data) {
    std::vector<InferenceEngine::CNNLayerPtr> nextLayers;
    if (data == nullptr) {
        return nextLayers;
    }
    for (auto nl : data->getInputTo()) {
        nextLayers.push_back(nl.second);
    }
    return nextLayers;
}

std::vector<InferenceEngine::CNNLayerPtr> CLDNNGraph::GetNextLayers(const InferenceEngine::CNNLayerPtr layer) {
    std::vector<InferenceEngine::CNNLayerPtr> nextLayers;
    if (layer == nullptr) {
        return nextLayers;
    }
    for (auto od : layer->outData) {
        auto nextLayersVec = GetNextLayers(od);
        for (auto nl : nextLayersVec) {
            nextLayers.push_back(nl);
        }
    }
    return nextLayers;
}

InferenceEngine::CNNLayerPtr CLDNNGraph::GetNextSingleLayer(const InferenceEngine::DataPtr data) {
    if (data == nullptr) {
        return nullptr;
    }
    auto nextLayers = GetNextLayers(data);
    IE_ASSERT(nextLayers.size() == 1);
    return nextLayers[0];
}

InferenceEngine::CNNLayerPtr CLDNNGraph::GetNextSingleLayer(const InferenceEngine::CNNLayerPtr layer) {
    if (layer == nullptr) {
        return nullptr;
    }
    auto nextLayers = GetNextLayers(layer);
    IE_ASSERT(nextLayers.size() == 1);
    return nextLayers[0];
}

void CLDNNGraph::InitFormat(InferenceEngine::ICNNNetwork &network) {
    m_defaultFormat    = FormatFromLayout(InferenceEngine::Layout::NCHW);
}

void CLDNNGraph::CompileNetwork() {
    m_env.debugOptions.AddTimedEvent("Network Build Begin");
    cldnn::build_options options;
    if (!m_config.graph_dumps_dir.empty()) {
        options.set_option(cldnn::build_option::graph_dumps_dir(m_config.graph_dumps_dir));
    }
    options.set_option(cldnn::build_option::optimize_data(true));
    options.set_option(cldnn::build_option::tuning_config(m_config.tuningConfig));

    m_env.network.reset();
    m_env.network = std::make_shared<cldnn::network>(cldnn::network(*(m_env.engine), *m_topology, options));
    m_env.debugOptions.AddTimedEvent("Network Build", "Network Build Begin");
}

void CLDNNGraph::Load(InferenceEngine::ICNNNetwork &network) {
    InitFormat(network);
    auto _networkPrecision = network.getPrecision();

    // 1. create inputs
    InferenceEngine::InputsDataMap networkInputs;
    network.getInputsInfo(networkInputs);
    p_currentInputs = &networkInputs;

    InferenceEngine::OutputsDataMap networkOutputs;
    network.getOutputsInfo(networkOutputs);
    p_currentOutputs = &networkOutputs;

    if (networkInputs.size() == 0) {
        THROW_CLDNN_EXCEPTION("No inputs detected.");
    }

    using LayerVect = std::vector<InferenceEngine::CNNLayerPtr>;
    std::list<InferenceEngine::CNNLayerPtr> layersToHandle;

    auto push_if = [&](const LayerVect& clist) {
        for (auto& l : clist) {
            if ( (std::find_if( layersToHandle.begin(),
                            layersToHandle.end(),
                            [&](const CNNLayerPtr& x) { return layer_type_name_ID(x) == layer_type_name_ID(l); } )) == layersToHandle.end() )
                layersToHandle.push_back(l);
        }
    };

    auto allInputs = CNNNetGetAllInputLayers(network);
    for (auto input : allInputs) {
        if (LayerTypeFromStr(input->type) == ConstantBlob) {
            AddConstantBlobInput(input);
        } else {
            auto iter = networkInputs.find(input->name);    // regular input
            if (iter != networkInputs.end()) {
                AddInputPrimitive(iter->second, input->precision);
            }
        }
        // collect next layers to process
        push_if(GetNextLayers(input));
    }

    // 2. traverse layers
    unsigned infLoopProtection = 0;
    while (!layersToHandle.empty()) {
        if (infLoopProtection++ >= layersToHandle.size()) {
            THROW_CLDNN_EXCEPTION("Infinite loop during network creation");
            break;
        }
        InferenceEngine::CNNLayerPtr currLayer = layersToHandle.front();
        layersToHandle.pop_front();
        auto layerName = layer_type_name_ID(currLayer);

        if (m_env.primitiveIDs.find(layerName) != m_env.primitiveIDs.end()) {
            infLoopProtection = 0;
            continue;  // this layer was already added (had multiple inputs)
        }

        bool missingInput = false;
        try {
            GetPrevLayersPrimitives(currLayer);
        } catch (std::exception) {
            missingInput = true;
        }

        if (missingInput) {  // some inputs aren't created yet
            layersToHandle.push_back(currLayer);  // push the current layer to the end of the line
            continue;  // move on to the next layer
        }

        infLoopProtection = 0;  // found a layer with all inputs already existing
        CreateSingleLayerPrimitive(currLayer);  // currLayer will be advanced if layer was skipped or merged
        m_env.prevPrimitiveIDs[layerName] = GetPrevLayersPrimitives(currLayer);

        push_if(GetNextLayers(currLayer));
    }

    // 3. Handle output reordering
    for (auto output : networkOutputs) {
        // always reorder and let clDNN remove unneeded reorders
        AddOutputPrimitive(output.first, output.second);
    }

    // 4. ???
    // 5. profit
    p_currentInputs = nullptr;
    p_currentOutputs = nullptr;
}

CLDNNGraph::LayerType CLDNNGraph::LayerTypeFromStr(const std::string &str) {
    static const caseless_map<std::string, CLDNNGraph::LayerType> LayerNameToType = {
        { "Convolution" , Convolution },
        { "ReLU" , ReLU },
        { "ReLU6" , ReLU6 },
        { "Sigmoid" , Sigmoid },
        { "Logistic" , Sigmoid },
        { "TanH" , TanH },
        { "ELU" , ELU },
        { "Activation" , Activation },
        { "Exp" , Exp },
        { "Not" , Not },
        { "Norm" , LRN },
        { "Pooling" , Pooling },
        { "FullyConnected" , FullyConnected },
        { "SoftMax" , SoftMax },
        { "Power" , Power },
        { "Split" , Split },
        { "Slice" , Split },
        { "Concat" , Concatenate },
        { "Eltwise" , Eltwise },
        { "SimplerNMS" , SimplerNMS },
        { "ROIPooling" , ROIPooling },
        { "Crop" , Crop },
        { "Deconvolution" , Deconvolution },
        { "PriorBox" , PriorBox },
        { "DetectionOutput" , DetectionOutput },
        { "Normalize" , Normalize },
        { "Reshape" , Reshape },
        { "Permute" , Permute },
        { "Flatten" , Flatten },
        { "BatchNormalization" , BatchNormalization },
        { "PReLU" , PReLU },
        { "ScaleShift" , ScaleShift },
        { "Proposal" , Proposal },
        { "PSROIPooling" , PSROIPooling },
        { "Clamp" , Clamp },
        { "Copy" , Copy },
        { "Upsampling" , Upsampling },
        { "Resample" , Resample },
        { "RegionYolo" , RegionYolo },
        { "ReorgYolo" , ReorgYolo },
        { "Const" , ConstantBlob },
        { "ArgMax" , ArgMax },
        { "MVN" , MVN },
        { "Unpooling" , Unpooling },
        { "Tile" , Tile },
        { "Pad" , Pad },
        { "LSTMCell" , LSTMCell },
        { "LSTMSequence" , RNN },
        { "RNNSequence" , RNN },
        { "Gather" , Gather },
        { "DepthToSpace" , DepthToSpace },
        { "ShuffleChannels" , ShuffleChannels },
        { "StridedSlice" , StridedSlice },
        { "ReverseSequence" , ReverseSequence }
    };
    auto it = LayerNameToType.find(str);
    if (it != LayerNameToType.end())
        return it->second;
    else
        return NO_TYPE;
}

cldnn::pooling_mode CLDNNGraph::PoolingModeFromIEPooling(InferenceEngine::PoolingLayer::PoolType pt, bool excludePadding) {
    switch (pt) {
        case InferenceEngine::PoolingLayer::PoolType::MAX:
            return cldnn::pooling_mode::max;
        case InferenceEngine::PoolingLayer::PoolType::AVG:
            return excludePadding ? cldnn::pooling_mode::average_no_padding : cldnn::pooling_mode::average;
        default: IE_ASSERT(0);  // unhandled pool mode
            THROW_CLDNN_EXCEPTION("Unsupported pooling type: " << pt);
            break;
    }

    return cldnn::pooling_mode::max;  // shouldn't get here
}

cldnn::eltwise_mode CLDNNGraph::EltwiseModeFromIEEltwise(InferenceEngine::EltwiseLayer::eOperation op) {
    switch (op) {
        case InferenceEngine::EltwiseLayer::Sum:
            return cldnn::eltwise_mode::sum;
        case InferenceEngine::EltwiseLayer::Prod:
            return cldnn::eltwise_mode::prod;
        case InferenceEngine::EltwiseLayer::Max:
            return cldnn::eltwise_mode::max;
        case InferenceEngine::EltwiseLayer::Sub:
            return cldnn::eltwise_mode::sub;
        case InferenceEngine::EltwiseLayer::Min:
            return cldnn::eltwise_mode::min;
        case InferenceEngine::EltwiseLayer::Div:
            return cldnn::eltwise_mode::div;
        case InferenceEngine::EltwiseLayer::Squared_diff:
            return cldnn::eltwise_mode::squared_diff;
        case InferenceEngine::EltwiseLayer::Equal:
            return cldnn::eltwise_mode::eq;
        case InferenceEngine::EltwiseLayer::Not_equal:
            return cldnn::eltwise_mode::ne;
        case InferenceEngine::EltwiseLayer::Less:
            return cldnn::eltwise_mode::lt;
        case InferenceEngine::EltwiseLayer::Less_equal:
            return cldnn::eltwise_mode::le;
        case InferenceEngine::EltwiseLayer::Greater:
            return cldnn::eltwise_mode::gt;
        case InferenceEngine::EltwiseLayer::Greater_equal:
            return cldnn::eltwise_mode::ge;
        case InferenceEngine::EltwiseLayer::Logical_AND:
            return cldnn::eltwise_mode::logic_and;
        case InferenceEngine::EltwiseLayer::Logical_OR:
            return cldnn::eltwise_mode::logic_or;
        case InferenceEngine::EltwiseLayer::Logical_XOR:
            return cldnn::eltwise_mode::logic_xor;
        default: THROW_CLDNN_EXCEPTION("Unsupported eltwise operation: " << op);
            break;
    }

    return cldnn::eltwise_mode::max;  // shouldn't get here
}

cldnn::concatenation::concatenation_axis CLDNNGraph::ConcatAxisFromIEAxis(unsigned axis) {
    switch (axis) {
    case 0:
        return cldnn::concatenation::concatenation_axis::along_b;
    case 1:
        return cldnn::concatenation::concatenation_axis::along_f;
    case 2:
        return cldnn::concatenation::concatenation_axis::along_y;
    case 3:
        return cldnn::concatenation::concatenation_axis::along_x;
    default: THROW_CLDNN_EXCEPTION("Unsupported concatenation axis: " << axis);
        break;
    }

    return cldnn::concatenation::concatenation_axis::along_f;  // shouldn't get here
}

void CLDNNGraph::CreatePrimitiveFromBlob(cldnn::primitive_id primID,
                                         const InferenceEngine::Blob::Ptr pBlob,
                                         cldnn::layout blobLayout,
                                         size_t blobByteOffset,
                                         WeightRearrangeType rearrange) {
    auto mem = cldnn::memory::allocate(*(m_env.engine), blobLayout);
    auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
    auto buf = tmpPointer.data();
    auto bufSize = blobLayout.bytes_count();
// The condition below is not valid once we use groups - todo: think of some other size check here
//     if ((pBlob != nullptr) &&
//         (pBlob->size() * (broadcastFeatures ? blobLayout.size.feature[0] : 1)) != blobLayout.count()) {
//         THROW_CLDNN_EXCEPTION("Unexpected blob size");
//     }
    if (pBlob == nullptr) {
        THROW_CLDNN_EXCEPTION("Missing blob data: " << primID);
    } else if ((pBlob->layout() != InferenceEngine::OIHW) &&
               (pBlob->layout() != InferenceEngine::NCHW) &&
               (pBlob->layout() != InferenceEngine::CHW) &&
               (pBlob->layout() != InferenceEngine::NC) &&
               (pBlob->layout() != InferenceEngine::C)) {
        // TODO: support more layouts
        THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(pBlob->layout()) << ") in blob: " << primID);
    } else if (rearrange == BroadcastFeatures) {
        size_t features = static_cast<size_t>(blobLayout.size.feature[0]);
        if (pBlob->size() != features) {
            THROW_CLDNN_EXCEPTION("Invalid blob dimensions to broadcast: " << primID);
        }
        auto data = static_cast<const char *>(pBlob->buffer());
        auto elementSize = cldnn::data_type_traits::size_of(blobLayout.data_type);
        size_t featureElements = blobLayout.count() / static_cast<size_t>(blobLayout.size.feature[0]);
        IE_ASSERT(blobLayout.format == cldnn::format::bfyx);
        for (size_t f = 0; f < features; f++) {
            for (size_t e = 0; e < featureElements; e++) {
                for (size_t b = 0; b < elementSize; b++) {
                    buf[(f*featureElements + e)*elementSize + b] = data[f*elementSize + b];
                }
            }
        }
    } else if (rearrange == FlipDeconvDims) {
        auto data = static_cast<const char *>(pBlob->buffer());
        auto elementSize = cldnn::data_type_traits::size_of(blobLayout.data_type);

        size_t inputFeatureElements = static_cast<size_t>(blobLayout.size.feature[0]);
        size_t outputFeatureElements = static_cast<size_t>(blobLayout.size.batch[0]);

        size_t featureSize = elementSize * static_cast<size_t>(blobLayout.size.spatial[0] * blobLayout.size.spatial[1]);

        for (size_t i = 0; i < inputFeatureElements; i++) {
            for (size_t o = 0; o < outputFeatureElements; o++) {
                size_t outputShift = (o*inputFeatureElements + i)*featureSize;
                size_t inputShift = (i*outputFeatureElements + o)*featureSize;

                for (size_t b = 0; b < featureSize; b++) {
                    buf[outputShift + b] = data[inputShift + b];
                }
            }
        }
    } else {
        auto data = static_cast<const char *>(pBlob->buffer());
        for (size_t i = 0; i < bufSize; i++) {
            buf[i] = data[i + blobByteOffset];
        }
    }
    m_topology->add(cldnn::data(primID, mem));
}

void CLDNNGraph::CreateWeightAndBiasPrimitives(const InferenceEngine::CNNLayerPtr& layer,
                                                   std::vector<cldnn::primitive_id>& weightsPrimID,
                                                   std::vector<cldnn::primitive_id>& biasesPrimID) {
    cldnn::tensor::value_type inFeatures = 1;  // todo: workaround for xyf input, handle general case (xf, xyzf etc...)
    std::shared_ptr<Data> insData0 = layer->insData[0].lock();
    IE_ASSERT(insData0 != nullptr);
    if (insData0->dims.size() > 2) {
        inFeatures = TensorValue(insData0->dims[2]);
    }
    cldnn::tensor::value_type outFeatures(0);
    std::vector<cldnn::tensor::value_type> weightDimsVec;
    InferenceEngine::Blob::Ptr pWeightsBlob, pBiasBlob;
    unsigned groupSize = 1;
    WeightRearrangeType rearrange = NO_REARRANGE;

    switch (LayerTypeFromStr(layer->type)) {
    case Convolution: {
        auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer *> (layer.get());
        if ((inFeatures % groupSize) || (convLayer->_out_depth % groupSize)) {
            THROW_CLDNN_EXCEPTION("Invalid group size in layer " << convLayer->name);
        }
        groupSize = convLayer->_group;
        if (groupSize >= 16)  // cldnn optimization for 16 and more groups
            groupSize = 1;
        weightDimsVec = {
            TensorValue(convLayer->_out_depth / groupSize),
            TensorValue(inFeatures / convLayer->_group),
            TensorValue(convLayer->_kernel[X_AXIS]),
            TensorValue(convLayer->_kernel[Y_AXIS])
        };
        outFeatures = convLayer->_out_depth;
        pWeightsBlob = convLayer->_weights;
        pBiasBlob = convLayer->_biases;
    }
        break;
    case Deconvolution: {
        auto deconvLayer = dynamic_cast<InferenceEngine::DeconvolutionLayer *> (layer.get());
        if ((inFeatures % groupSize) || (deconvLayer->_out_depth % groupSize)) {
            THROW_CLDNN_EXCEPTION("Invalid group size in layer " << deconvLayer->name);
        }
        groupSize = deconvLayer->_group;
        if (groupSize >= 16)  // cldnn optimization for 16 and more groups
            groupSize = 1;
        weightDimsVec = {
            TensorValue(deconvLayer->_out_depth / groupSize),
            TensorValue(inFeatures / deconvLayer->_group),
            TensorValue(deconvLayer->_kernel[X_AXIS]),
            TensorValue(deconvLayer->_kernel[Y_AXIS])
        };
        outFeatures = deconvLayer->_out_depth;
        pWeightsBlob = deconvLayer->_weights;
        pBiasBlob = deconvLayer->_biases;

        if ((groupSize < outFeatures) || (groupSize < inFeatures))
            rearrange = FlipDeconvDims;
    }
        break;
    default:
        IE_ASSERT("Wrong weightable layer type");  // shouldn't get here
        break;
    }

    // create weights primitive
    cldnn::layout weightsLayout = cldnn::layout(
        DataTypeFromPrecision(layer->precision),
        m_defaultFormat,
        cldnn::tensor(weightDimsVec));
    size_t bytesPerGroup = weightsLayout.bytes_count();

    for (unsigned g = 0; g < groupSize; g++) {
        cldnn::primitive_id weightID = layer_type_name_ID(layer) + m_weightsTag + std::to_string(g);
        CreatePrimitiveFromBlob(
            weightID,
            pWeightsBlob,
            weightsLayout,
            g * bytesPerGroup,
            rearrange);
        weightsPrimID.push_back(weightID);
    }

    // create bias primitive
    if (pBiasBlob != nullptr) {
        cldnn::layout biasesLayout = cldnn::layout(
            DataTypeFromPrecision(layer->precision),
            m_defaultFormat,
            cldnn::spatial(TensorValue(outFeatures / groupSize)));
        size_t bytesPerGroup = biasesLayout.bytes_count();
        for (unsigned g = 0; g < groupSize; g++) {
            cldnn::primitive_id biasID = layer_type_name_ID(layer) + m_biasesTag + std::to_string(g);
            CreatePrimitiveFromBlob(
                biasID,
                pBiasBlob,
                biasesLayout,
                g * bytesPerGroup);
            biasesPrimID.push_back(biasID);
        }
    }
}

void CLDNNGraph::CreateScaleWeightsAndBiasesFromBN(
    const InferenceEngine::BatchNormalizationLayer* bnLayer,
    cldnn::primitive_id weightsPrimID,
    cldnn::primitive_id biasesPrimID) {

    if (bnLayer->_weights->dims() != bnLayer->_biases->dims()) {
        THROW_CLDNN_EXCEPTION("mean/variance dimensions mismatch in " << bnLayer->name);
    }
    if (bnLayer->_weights->precision() != bnLayer->_biases->precision()) {
        THROW_CLDNN_EXCEPTION("mean/variance precision mismatch in " << bnLayer->name);
    }

    cldnn::tensor blobTensor(0);
    switch (bnLayer->outData[0]->dims.size()) {
    case 2:
        blobTensor = cldnn::feature(TensorValue(bnLayer->outData[0]->dims[0]));
        break;
    case 4:
        blobTensor = cldnn::feature(TensorValue(bnLayer->outData[0]->dims[2]));
        break;
    default:
        THROW_CLDNN_EXCEPTION("Batch normalization input doesn't have 2 or 4 dimensions in " << bnLayer->name);
    }
    cldnn::layout blobLayout(
        DataTypeFromPrecision(bnLayer->precision),
        m_defaultFormat,
        blobTensor);

    switch (bnLayer->_weights->precision()) {
    case Precision::FP16: {
        InferenceEngine::TBlob<uint16_t> weightsBlob(bnLayer->_weights->precision(), bnLayer->_weights->layout(),  bnLayer->_weights->dims());
        weightsBlob.allocate();
        InferenceEngine::TBlob<uint16_t> biasesBlob(bnLayer->_biases->precision(), bnLayer->_weights->layout(), bnLayer->_biases->dims());
        biasesBlob.allocate();

        auto weightsData = weightsBlob.data();
        auto biasesData = biasesBlob.data();
        auto varianceData = static_cast<const uint16_t *>(bnLayer->_weights->buffer());
        auto meanData = static_cast<const uint16_t *>(bnLayer->_biases->buffer());

        cldnn_status status = CLDNN_SUCCESS;
        for (size_t i = 0; i < weightsBlob.size(); i++) {
            auto variance = cldnn_half_to_float(varianceData[i], &status);
            if (status != CLDNN_SUCCESS) THROW_CLDNN_EXCEPTION("Error during fp16 conversion for layer " << bnLayer->name);
            auto mean = cldnn_half_to_float(meanData[i], &status);
            if (status != CLDNN_SUCCESS) THROW_CLDNN_EXCEPTION("Error during fp16 conversion for layer " << bnLayer->name);

            float scale = 1.0f / sqrt(variance + bnLayer->epsilon);
            weightsData[i] = cldnn_float_to_half(scale, &status);
            if (status != CLDNN_SUCCESS) THROW_CLDNN_EXCEPTION("Error during fp16 conversion for layer " << bnLayer->name);
            biasesData[i] = cldnn_float_to_half((-mean) * scale, &status);
            if (status != CLDNN_SUCCESS) THROW_CLDNN_EXCEPTION("Error during fp16 conversion for layer " << bnLayer->name);
        }
        CreatePrimitiveFromBlob(weightsPrimID, std::make_shared<InferenceEngine::TBlob<uint16_t>>(weightsBlob), blobLayout);
        CreatePrimitiveFromBlob(biasesPrimID, std::make_shared<InferenceEngine::TBlob<uint16_t>>(biasesBlob), blobLayout);
    }
        break;
    case Precision::FP32: {
        InferenceEngine::TBlob<float> weightsBlob(bnLayer->_weights->precision(), bnLayer->_weights->layout(), bnLayer->_weights->dims());
        weightsBlob.allocate();
        InferenceEngine::TBlob<float> biasesBlob(bnLayer->_biases->precision(), bnLayer->_weights->layout(), bnLayer->_biases->dims());
        biasesBlob.allocate();

        auto weightsData = weightsBlob.data();
        auto biasesData = biasesBlob.data();
        auto varianceData = static_cast<const float *>(bnLayer->_weights->buffer());
        auto meanData = static_cast<const float *>(bnLayer->_biases->buffer());

        for (size_t i = 0; i < weightsBlob.size(); i++) {
            auto variance = varianceData[i];
            auto mean = meanData[i];
            weightsData[i] = 1.0f / sqrt(variance + bnLayer->epsilon);
            biasesData[i] = (-mean) * weightsData[i];
        }
        CreatePrimitiveFromBlob(weightsPrimID, std::make_shared<InferenceEngine::TBlob<float>>(weightsBlob), blobLayout);
        CreatePrimitiveFromBlob(biasesPrimID, std::make_shared<InferenceEngine::TBlob<float>>(biasesBlob), blobLayout);
    }
        break;
    default:
        THROW_CLDNN_EXCEPTION("Unhandled mean/variance precision in " << bnLayer->name);
        break;
    }
}

void CLDNNGraph::CreateSingleLayerPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    // Initialize a profiling entry
    InitProfileInfo(layer->name, layer->type);

    // First check for custom layer
    auto customLayer = m_config.customLayers.find(layer->type);
    if (customLayer != m_config.customLayers.end()) {
        CreateCustomLayerPrimitive(layer, customLayer->second);
        return;
    }

    // Otherwise move on to built-in layer types
    switch (LayerTypeFromStr(layer->type)) {
        case Convolution: CreateConvolutionPrimitive(layer);
            break;
        case ReLU:
        case ReLU6:
        case Sigmoid:
        case TanH:
        case ELU:
        case Clamp:
        case Activation:
        case Exp:
        case Not:
            CreateActivationPrimitive(layer, LayerTypeFromStr(layer->type));
            break;
        case LRN: CreateLRNPrimitive(layer);
            break;
        case Pooling: CreatePoolingPrimitive(layer);
            break;
        case Unpooling: CreateMaxUnpoolingPrimitive(layer);
            break;
        case FullyConnected: CreateFullyConnectedPrimitive(layer);
            break;
        case SoftMax: CreateSoftMaxPrimitive(layer);
            break;
        case Power: CreatePowerPrimitive(layer);
            break;
        case Split: CreateSplitPrimitive(layer);
            break;
        case Concatenate: CreateConcatenatePrimitive(layer);
            break;
        case Eltwise: CreateEltwisePrimitive(layer);
            break;
        case SimplerNMS: CreateSimplerNMSPrimitive(layer);
            break;
        case ROIPooling: CreateROIPoolingPrimitive(layer);
            break;
        case Crop: CreateCropPrimitive(layer);
            break;
        case Deconvolution: CreateDeconvolutionPrimitive(layer);
            break;
        case PriorBox: CreatePriorBoxPrimitive(layer);
            break;
        case DetectionOutput: CreateDetectionOutputPrimitive(layer);
            break;
        case Normalize: CreateNormalizePrimitive(layer);
            break;
        case Reshape: CreateReshapePrimitive(layer);
            break;
        case Permute: CreatePermutePrimitive(layer);
            break;
        case Flatten: CreateFlattenPrimitive(layer);
            break;
        case BatchNormalization: CreateBatchNormalizationPrimitive(layer);
            break;
        case PReLU: CreatePReLUPrimitive(layer);
            break;
        case ScaleShift: CreateScaleShiftPrimitive(layer);
            break;
        case Proposal: CreateProposalPrimitive(layer);
            break;
        case PSROIPooling: CreatePSROIPoolingPrimitive(layer);
            break;
        case Copy: CreateCopyPrimitive(layer);
            break;
        case Upsampling: CreateUpsamplingPrimitive(layer);
            break;
        case Resample: CreateResamplePrimitive(layer);
            break;
        case ArgMax: CreateArgMaxPrimitive(layer);
            break;
        case MVN: CreateMVNPrimitive(layer);
            break;
        case LSTMCell: CreateLSTMCellPrimitive(layer);
            break;
        case RNN: CreateRNNPrimitive(layer);
            break;
        case RegionYolo: CreateYOLO2RegionPrimitive(layer);
            break;
        case ReorgYolo: CreateYOLO2ReorgPrimitive(layer);
            break;
        case Tile: CreateTilePrimitive(layer);
            break;
        case Pad: CreatePadPrimitive(layer);
            break;
        case Gather: CreateGatherPrimitive(layer);
            break;
        case DepthToSpace: CreateDepthToSpacePrimitive(layer);
            break;
        case ShuffleChannels: CreateShuffleChannelsPrimitive(layer);
            break;
        case StridedSlice: CreateStridedSlicePrimitive(layer);
            break;
        case ReverseSequence: CreateReverseSequencePrimitive(layer);
            break;
        default: THROW_CLDNN_EXCEPTION("Unknown Layer Type: " << layer->type);
    }
}

void CLDNNGraph::CreateScaleShiftPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto scaleShiftLayer = dynamic_cast<InferenceEngine::ScaleShiftLayer*> (layer.get());

    // create scales and biases
    cldnn::primitive_id scalePrimID = scaleShiftLayer->name + m_scalesTag;
    cldnn::primitive_id biasPrimID = scaleShiftLayer->name + m_biasesTag;

    const auto& dims = scaleShiftLayer->_weights->dims();
    cldnn::tensor weightTensor(1);
    switch (dims.size()) {
    case 1: weightTensor = cldnn::feature(TensorValue(dims[0]));  // value per feature (or 1 global value)
        break;
    case 4: weightTensor = cldnn::tensor(TensorValue(dims[0]), TensorValue(dims[1]), TensorValue(dims[3]), TensorValue(dims[2]));  // value per pixel
        break;
    default: THROW_CLDNN_EXCEPTION("Invalid weights dimensions in layer " << layer->name);
        break;
    }
    cldnn::layout blobLayout(DataTypeFromPrecision(layer->precision), m_defaultFormat, weightTensor);
    CreatePrimitiveFromBlob(scalePrimID, scaleShiftLayer->_weights, blobLayout);
    if (scaleShiftLayer->_biases != nullptr) {
        if (scaleShiftLayer->_biases->dims() != dims) {
            THROW_CLDNN_EXCEPTION("Invalid bias blob dimensions in layer " << layer->name);
        }
        CreatePrimitiveFromBlob(biasPrimID, scaleShiftLayer->_biases, blobLayout);
    } else {
        biasPrimID = "";  // 0-bias
    }

    std::string scaleShiftLayerName = layer_type_name_ID(layer);
    auto scaleShiftPrim = cldnn::scale(
        scaleShiftLayerName,
        inputPrimitives[0],
        scalePrimID,
        biasPrimID);

    m_env.primitiveIDs[scaleShiftLayerName] = scaleShiftLayerName;
    m_topology->add(scaleShiftPrim);
    m_env.profilingIDs.push_back(scaleShiftLayerName);
}

void CLDNNGraph::CreateProposalPrimitive(InferenceEngine::CNNLayerPtr & layer) {
    ValidateLayer(layer, 3);
    auto proposalLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    float nms_thresh = proposalLayer->GetParamAsFloat("nms_thresh", 0.7f);
    int min_size = proposalLayer->GetParamAsInt("min_size", 16);
    int feature_stride = proposalLayer->GetParamAsInt("feat_stride", 16);
    int pre_nms_topn = proposalLayer->GetParamAsInt("pre_nms_topn", 6000);
    int post_nms_topn = proposalLayer->GetParamAsInt("post_nms_topn", 300);
    const std::vector<float> ratio = proposalLayer->GetParamAsFloats("ratio");
    const std::vector<float> scale = proposalLayer->GetParamAsFloats("scale");
    float box_coordinate_scale = proposalLayer->GetParamAsFloat("box_coordinate_scale", 1.0f);
    float box_size_scale = proposalLayer->GetParamAsFloat("box_size_scale", 1.0f);
    int base_size = proposalLayer->GetParamAsInt("base_size", 16);
    std::string framework = proposalLayer->GetParamAsString("framework", "");
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    bool normalize = layer->GetParamsAsBool("normalize", false);
    bool clip_before_nms = layer->GetParamsAsBool("clip_before_nms", true);
    bool clip_after_nms = layer->GetParamsAsBool("clip_after_nms", false);

    float coordinates_offset;
    bool swap_xy;
    bool initial_clip;
    bool round_ratios;
    bool shift_anchors;

    if (framework == "tensorflow") {
        coordinates_offset = 0.0f;
        initial_clip = true;
        shift_anchors = true;
        round_ratios = false;
        swap_xy = true;
    } else {
        coordinates_offset = 1.0f;
        initial_clip = false;
        shift_anchors = false;
        round_ratios = true;
        swap_xy = false;
    }

    std::string proposalLayerName = layer_type_name_ID(layer);
    auto proposalPrim = cldnn::proposal(
        proposalLayerName,
        inputPrimitives[0],  // cls_score
        inputPrimitives[1],  // bbox_pred
        inputPrimitives[2],  // im_info
        0,                   // max_num_proposals is unused
        nms_thresh,
        base_size,
        min_size,
        feature_stride,
        pre_nms_topn,
        post_nms_topn,
        ratio,
        scale,
        coordinates_offset,
        box_coordinate_scale,
        box_size_scale,
        swap_xy,
        initial_clip,
        clip_before_nms,
        clip_after_nms,
        round_ratios,
        shift_anchors,
        normalize);

    m_env.primitiveIDs[proposalLayerName] = proposalLayerName;
    m_topology->add(proposalPrim);
    m_env.profilingIDs.push_back(proposalLayerName);
}

void CLDNNGraph::CreatePReLUPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto preluLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    std::string preluLayerName = layer_type_name_ID(layer);
    auto inDataPtr = preluLayer->insData[0].lock();
    if (!inDataPtr) {
        THROW_CLDNN_EXCEPTION("Data inserted into PreLu " << preluLayer->name << " is nullptr");
    }
    auto inputDims = inDataPtr->dims;
    static const std::string blobName("weights");
    ValidateGenericLayerBlobs(preluLayer, { blobName });

    bool channel_shared = preluLayer->GetParamsAsBool("channel_shared", false);

    auto slopeBlob = preluLayer->blobs.at(blobName);
    if (channel_shared) {
        if (slopeBlob->dims()[0] != 1) {
            THROW_CLDNN_EXCEPTION("PReLU slope blob with wrong dimensions in " << preluLayer->name);
        }
        float slope(0.0f);
        switch (slopeBlob->precision()) {
        case InferenceEngine::Precision::FP32:
            slope = *static_cast<const float *>(slopeBlob->buffer());
            break;
        case InferenceEngine::Precision::FP16:
        {
            cldnn_status status = CLDNN_SUCCESS;
            slope = cldnn_half_to_float(*static_cast<const uint16_t *>(slopeBlob->buffer()), &status);
            if (status != CLDNN_SUCCESS) {
                THROW_CLDNN_EXCEPTION("Error converting fp16 value in " << preluLayer->name);
            }
        }
            break;
        default: THROW_CLDNN_EXCEPTION("Invalid PReLU slope blob precision in " << preluLayer->name);
        }
        m_topology->add(cldnn::activation(preluLayerName, inputPrimitives[0], activation_relu_negative_slope, { slope, 0.f }));
    } else {
        CreateGenericLayerBlobPrimitives(preluLayer);
        cldnn::primitive_id slopePrimID(preluLayerName + "_" + blobName + m_weightsTag);
        m_topology->add(cldnn::activation(preluLayerName, inputPrimitives[0], slopePrimID, activation_relu_negative_slope));
    }

    m_env.primitiveIDs[preluLayerName] = preluLayerName;
    m_env.profilingIDs.push_back(preluLayerName);
}

void CLDNNGraph::CreateBatchNormalizationPrimitive(InferenceEngine::CNNLayerPtr & layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    std::string bnLayerName = layer_type_name_ID(layer);

    auto bnLayer = dynamic_cast<InferenceEngine::BatchNormalizationLayer *> (layer.get());
    cldnn::primitive_id weightID = bnLayerName + "_" + m_scalesTag;
    cldnn::primitive_id biasID = bnLayerName + "_" + m_biasesTag;

#define _SCALE_BN_OPT
#ifdef _SCALE_BN_OPT
    // Using scale as an optimization (1 mad instead of mad+rsq)
    // create new blobs for scale shift
    CreateScaleWeightsAndBiasesFromBN(bnLayer, weightID, biasID);
    auto scalePrim = cldnn::scale(bnLayerName, inputPrimitives[0], weightID, biasID);

    m_env.primitiveIDs[bnLayerName] = bnLayerName;
    m_topology->add(scalePrim);
    m_env.profilingIDs.push_back(bnLayerName);
    return;
#endif  // _SCALE_BN_OPT

    cldnn::tensor blobTensor(0);
    switch (bnLayer->outData[0]->dims.size()) {
    case 2:
        blobTensor = cldnn::feature(TensorValue(bnLayer->outData[0]->dims[0]));
        break;
    case 4:
        blobTensor = cldnn::feature(TensorValue(bnLayer->outData[0]->dims[2]));
        break;
    default:
        THROW_CLDNN_EXCEPTION("Batch normalization input doesn't have 2 or 4 dimensions in " << bnLayer->name);
    }
    cldnn::layout blobLayout(
        DataTypeFromPrecision(layer->precision),
        m_defaultFormat,
        blobTensor);

    // Create variance primitive
    cldnn::primitive_id varianceID = bnLayerName + "_" + m_weightsTag;
    CreatePrimitiveFromBlob(varianceID, bnLayer->_weights, blobLayout);

    // Create mean primitive
    cldnn::primitive_id meanID = bnLayerName + "_" + m_biasesTag;
    CreatePrimitiveFromBlob(meanID, bnLayer->_biases, blobLayout);

    auto bnPrim = cldnn::batch_norm(
        bnLayerName,
        inputPrimitives[0],
        meanID,
        varianceID,
        bnLayer->epsilon);

    m_env.primitiveIDs[bnLayerName] = bnLayerName;
    m_topology->add(bnPrim);
    m_env.profilingIDs.push_back(bnLayerName);
}

void CLDNNGraph::CreateFlattenPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto flattenLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    std::string flattenLayerName = layer_type_name_ID(layer);

    auto flattenPrim = cldnn::reshape(
        flattenLayerName,
        inputPrimitives[0],
        CldnnTensorFromIEDims(flattenLayer->outData[0]->dims));

    m_env.primitiveIDs[flattenLayerName] = flattenLayerName;
    m_topology->add(flattenPrim);
    m_env.profilingIDs.push_back(flattenLayerName);
}

void CLDNNGraph::CreatePermutePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto permuteLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    std::vector<uint16_t> ie_order;
    for (auto& a : permuteLayer->GetParamAsInts("order"))
        ie_order.push_back(static_cast<uint16_t>(a));

    // if order size is less than 4 - fill the rest with just copy
    for (auto o = ie_order.size(); o < 4; o++)
        ie_order.push_back((uint16_t)o);

    /*
        Because ofthe cldnn ordering: bfxy, and IE ordering: bfyx
        wee need to adjust the permute order.
    */
    std::vector<uint16_t> cldnn_permute_order;
    // 1. Switch permute order values (x and y)
    for (auto const& o : ie_order) {
        if (o == 2)
            cldnn_permute_order.push_back(3);
        else if (o == 3)
            cldnn_permute_order.push_back(2);
        else
            cldnn_permute_order.push_back(o);
    }
    // 2. Swap x and y positions
    std::swap(cldnn_permute_order[2], cldnn_permute_order[3]);

    std::string permuteLayerName = layer_type_name_ID(layer);

    auto permutePrim = cldnn::permute(
        permuteLayerName,
        inputPrimitives[0],
        cldnn_permute_order);

    m_env.primitiveIDs[permuteLayerName] = permuteLayerName;
    m_topology->add(permutePrim);
    m_env.profilingIDs.push_back(permuteLayerName);
}

void CLDNNGraph::CreateReshapePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto reshapeLayer = dynamic_cast<InferenceEngine::ReshapeLayer*> (layer.get());
    IE_ASSERT(reshapeLayer->outData.size());
    std::string reshapeLayerName = layer_type_name_ID(layer);

    auto reshapePrim = cldnn::reshape(
        reshapeLayerName,
        inputPrimitives[0],
        CldnnTensorFromIEDims(reshapeLayer->outData[0]->dims));

    m_env.primitiveIDs[reshapeLayerName] = reshapeLayerName;
    m_topology->add(reshapePrim);
    m_env.profilingIDs.push_back(reshapeLayerName);
}

void CLDNNGraph::CreateNormalizePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto normLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    ValidateGenericLayerBlobs(normLayer, { "weights" });
    CreateGenericLayerBlobPrimitives(normLayer);

    // params
    bool across_spatial = normLayer->GetParamsAsBool("across_spatial", true);
    float eps = normLayer->GetParamAsFloat("eps", 0.0f);

    // WA for MO outputting %.6f
    if (eps == 0.0f) {
        eps = 1e-10f;
    }

    std::string normLayerName = layer_type_name_ID(layer);
    auto normPrim = cldnn::normalize(
        normLayerName,
        inputPrimitives[0],
        normLayerName + "_weights" + m_weightsTag,
        across_spatial,
        eps);

    m_env.primitiveIDs[normLayerName] = normLayerName;
    m_topology->add(normPrim);
    m_env.profilingIDs.push_back(normLayerName);
}

void CLDNNGraph::CreateDetectionOutputPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 3);
    auto detectionLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    uint32_t num_classes            = detectionLayer->GetParamAsUInt("num_classes", 1);
    bool share_location             = detectionLayer->GetParamsAsBool("share_location", true);
    int background_label_id         = detectionLayer->GetParamAsInt("background_label_id", 0);
    float nms_threshold             = detectionLayer->GetParamAsFloat("nms_threshold", 0.3f);
    int top_k                       = detectionLayer->GetParamAsInt("top_k", -1);
    float confidence_threshold      = detectionLayer->GetParamAsFloat("confidence_threshold", -FLT_MAX);
    float eta                       = detectionLayer->GetParamAsFloat("eta", 1.0f);
    int keep_top_k                  = detectionLayer->GetParamAsInt("keep_top_k", -1);
    bool variance_encoded_in_target = detectionLayer->GetParamsAsBool("variance_encoded_in_target", false);
    int input_width                 = detectionLayer->GetParamAsInt("input_width", -1);
    int input_height                = detectionLayer->GetParamAsInt("input_height", -1);
    bool normalized                 = detectionLayer->GetParamsAsBool("normalized", true);
    std::string code_type           = detectionLayer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CORNER");
    bool clip_before_nms            = detectionLayer->GetParamsAsBool("clip_before_nms", false) ||
                                      detectionLayer->GetParamsAsBool("clip", false);  // For backward compatibility
    bool clip_after_nms             = detectionLayer->GetParamsAsBool("clip_after_nms", false);
    bool decrease_label_id          = detectionLayer->GetParamsAsBool("decrease_label_id", false);

    cldnn::prior_box_code_type cldnnCodeType = PriorBoxCodeFromString(code_type);
    int32_t prior_info_size = normalized != 0 ? 4 : 5;
    int32_t prior_coordinates_offset = normalized != 0 ? 0 : 1;

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    std::string detectionLayerName = layer_type_name_ID(layer);
    auto detectionPrim = cldnn::detection_output(detectionLayerName,
                                                 inputPrimitives[0],
                                                 inputPrimitives[1],
                                                 inputPrimitives[2],
                                                 num_classes,
                                                 keep_top_k,
                                                 share_location,
                                                 background_label_id,
                                                 nms_threshold,
                                                 top_k,
                                                 eta,
                                                 cldnnCodeType,
                                                 variance_encoded_in_target,
                                                 confidence_threshold,
                                                 prior_info_size,
                                                 prior_coordinates_offset,
                                                 normalized,
                                                 input_width,
                                                 input_height,
                                                 decrease_label_id,
                                                 clip_before_nms,
                                                 clip_after_nms);

    m_env.primitiveIDs[detectionLayerName] = detectionLayerName;
    m_topology->add(detectionPrim);
    m_env.profilingIDs.push_back(detectionLayerName);
}

void CLDNNGraph::CreatePriorBoxPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);
    auto priorBoxLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // params
    std::vector<float> min_size = priorBoxLayer->GetParamAsFloats("min_size");
    std::vector<float> max_size = priorBoxLayer->GetParamAsFloats("max_size", {});
    std::vector<float> aspect_ratio = priorBoxLayer->GetParamAsFloats("aspect_ratio", {});
    std::vector<float> variance = priorBoxLayer->GetParamAsFloats("variance");
    bool flip = priorBoxLayer->GetParamsAsBool("flip", true);
    bool clip = priorBoxLayer->GetParamsAsBool("clip", false);
    bool scale_all_sizes = priorBoxLayer->GetParamsAsBool("scale_all_sizes", true);
    float offset = priorBoxLayer->GetParamAsFloat("offset", 0.5f);

    auto step_w = priorBoxLayer->GetParamAsFloat("step_w", 0.0f);
    auto step_h = priorBoxLayer->GetParamAsFloat("step_h", 0.0f);
    auto step   = priorBoxLayer->GetParamAsFloat("step", 0.0f);

    float _step_w = 0.0f;
    float _step_h = 0.0f;
    if (HasParam(priorBoxLayer->params, "step_w") && step_w != 0.0f &&
        HasParam(priorBoxLayer->params, "step_h") && step_h != 0.0f) {
        _step_w = step_w;
        _step_h = step_h;
    } else if (HasParam(priorBoxLayer->params, "step") && step != 0.0f) {
        _step_w = step;
        _step_h = step;
    }

    int img = priorBoxLayer->GetParamAsInt("img_size", 0);
    int img_w = priorBoxLayer->GetParamAsInt("img_w", 0);
    int img_h = priorBoxLayer->GetParamAsInt("img_h", 0);
    if ((img != 0) || (img_w != 0) || (img_h != 0)) {
        // unsupported mode
        THROW_CLDNN_EXCEPTION("Unsupported image sizes in prior box " + layer->name + " (use an image blob instead of dimensions)");
    }

    IE_ASSERT(layer->insData[1].lock());
    auto img_dims = layer->insData[1].lock()->dims;
    cldnn::tensor img_size = cldnn::spatial(TensorValue(img_dims[0]), TensorValue(img_dims[1]));
    std::vector<cldnn::primitive_id> inputPrimitives = GetPrevLayersPrimitives(layer);
    // second input isn't used by value - only dimensions taken from the layer input

    if (_step_w == 0.0f || _step_h == 0.0f) {
        _step_w = static_cast<float>(img_w) / static_cast<float>(img_dims[0]);
        _step_h = static_cast<float>(img_h) / static_cast<float>(img_dims[1]);
    }

    std::string priorBoxLayerName = layer_type_name_ID(layer);
    auto priorBoxPrim = cldnn::prior_box(
        priorBoxLayerName,
        inputPrimitives[0],
        img_size,
        min_size,
        max_size,
        aspect_ratio,
        flip,
        clip,
        variance,
        _step_w,
        _step_h,
        offset,
        scale_all_sizes);

    m_env.primitiveIDs[priorBoxLayerName] = priorBoxLayerName;
    m_topology->add(priorBoxPrim);
    m_env.profilingIDs.push_back(priorBoxLayerName);
}

void CLDNNGraph::CreateDeconvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto deconvLayer = dynamic_cast<InferenceEngine::DeconvolutionLayer *> (layer.get());

    if (deconvLayer->_dilation[X_AXIS] != 1 || deconvLayer->_dilation[Y_AXIS] != 1) {
        THROW_CLDNN_EXCEPTION("Unsupported dilation in deconvolution " << layer->name);
    }

    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(layer, weightPrimID, biasPrimID);
    auto allPads = getPaddings(*deconvLayer);
    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
        cldnn::spatial(deconvLayer->_stride[X_AXIS], deconvLayer->_stride[Y_AXIS]));
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
        cldnn::spatial(-allPads.begin[X_AXIS], -allPads.begin[Y_AXIS]));

    std::string deconvLayerName = layer_type_name_ID(layer);

    if (deconvLayer->_group >= 16) {
        auto deconvPrim = cldnn::deconvolution(deconvLayerName,
            inputPrimitives[0],
            weightPrimID,
            biasPrimID,
            deconvLayer->_group,
            stride,
            padding,
            false,
            0.0f,
            CldnnTensorFromIEDims(deconvLayer->outData[0]->dims));
        m_topology->add(deconvPrim);
    } else {
        auto deconvPrim = cldnn::deconvolution(deconvLayerName,
            inputPrimitives[0],
            weightPrimID,
            biasPrimID,
            stride,
            padding,
            false,
            0.0f,
            CldnnTensorFromIEDims(deconvLayer->outData[0]->dims));
        m_topology->add(deconvPrim);
    }
    m_env.primitiveIDs[deconvLayerName] = deconvLayerName;
    m_env.profilingIDs.push_back(deconvLayerName);
}

void CLDNNGraph::CreateCropPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    if (layer->insData.size() != 1 && layer->insData.size() != 2) {
        THROW_CLDNN_EXCEPTION("Invalid number of inputs for layer: " << layer->name);
    }
    if (layer->_fusedWith) {
        THROW_CLDNN_EXCEPTION("Unsupported fuse in layer: " << layer->name << " with: " << layer->_fusedWith->name);
    }
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto cropLayer = dynamic_cast<InferenceEngine::CropLayer*> (layer.get());
    IE_ASSERT(cropLayer->axis.size() == cropLayer->offset.size());
    // IE_ASSERT(cropLayer->outData[0] && cropLayer->outData[0]->dims.size() == 4);

    std::vector<cldnn::tensor::value_type> offset{ 0, 0, 0, 0 };
    for (size_t i = 0; i < cropLayer->axis.size(); i++) {
        if (cropLayer->axis[i] < 0 || cropLayer->axis[i] > 3) {
            THROW_CLDNN_EXCEPTION("Invalid crop axis: " + std::to_string(cropLayer->axis[i]) + " in layer " + cropLayer->name);
        }
        offset[cropLayer->axis[i]] = cropLayer->offset[i];
    }
    auto outputDims = cropLayer->outData[0]->dims;
    size_t ods = outputDims.size();
    cldnn::tensor refSize(
        TensorValue(ods > 3 ? outputDims[3] : 1),
        TensorValue(ods > 2 ? outputDims[2] : 1),
        TensorValue(outputDims[0]),
        TensorValue(outputDims[1]));

    cldnn::tensor offSize(
        TensorValue(offset[0]),
        TensorValue(offset[1]),
        TensorValue(offset[3]),
        TensorValue(offset[2]));

    std::string cropLayerName = layer_type_name_ID(layer);
    auto cropPrim = cldnn::crop(
        cropLayerName,
        inputPrimitives[0],
        refSize,
        offSize);
    m_env.primitiveIDs[cropLayerName] = cropLayerName;
    m_topology->add(cropPrim);
    m_env.profilingIDs.push_back(cropLayerName);
}

void CLDNNGraph::CreateROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);
    auto roiPoolingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // params
    int pooled_width = roiPoolingLayer->GetParamAsInt("pooled_w", 0);
    int pooled_height = roiPoolingLayer->GetParamAsInt("pooled_h", 0);
    float spatial_scale = roiPoolingLayer->GetParamAsFloat("spatial_scale", 1.0f);
    std::string method = roiPoolingLayer->GetParamAsString("method", "max");
    bool position_sensitive = false;

    cldnn::pooling_mode mode = cldnn::pooling_mode::max;
    if (method == "bilinear") {
        mode = cldnn::pooling_mode::bilinear;
    }
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::string roiPoolingLayerName = layer_type_name_ID(layer);
    auto roiPoolingPrim = cldnn::roi_pooling(roiPoolingLayerName,
                                             inputPrimitives[0],  // input data
                                             inputPrimitives[1],  // input rois
                                             mode,
                                             position_sensitive,
                                             pooled_width,
                                             pooled_height,
                                             spatial_scale);
    m_env.primitiveIDs[roiPoolingLayerName] = roiPoolingLayerName;
    m_topology->add(roiPoolingPrim);
    m_env.profilingIDs.push_back(roiPoolingLayerName);
}

void CLDNNGraph::CreatePSROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);
    auto psROIPoolingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // params
    int group_size = psROIPoolingLayer->GetParamAsInt("group_size");
    int output_dim = psROIPoolingLayer->GetParamAsInt("output_dim");
    float spatial_scale = psROIPoolingLayer->GetParamAsFloat("spatial_scale");
    size_t spatial_bins_x = static_cast<size_t>(psROIPoolingLayer->GetParamAsInt("spatial_bins_x", 1));
    size_t spatial_bins_y = static_cast<size_t>(psROIPoolingLayer->GetParamAsInt("spatial_bins_y", 1));
    std::string mode_str = psROIPoolingLayer->GetParamAsString("mode", "average");
    bool position_sensitive = true;

    cldnn::pooling_mode mode = mode_str == "average" ? cldnn::pooling_mode::average
                                                     : cldnn::pooling_mode::bilinear;

    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::string psROIPoolingLayerName = layer_type_name_ID(layer);
    auto psROIPoolingPrim = cldnn::roi_pooling(psROIPoolingLayerName,
                                               inputPrimitives[0],  // input data
                                               inputPrimitives[1],  // input rois
                                               mode,
                                               position_sensitive,
                                               group_size,
                                               group_size,
                                               spatial_scale,
                                               output_dim,
                                               spatial_bins_x,
                                               spatial_bins_y);

    m_env.primitiveIDs[psROIPoolingLayerName] = psROIPoolingLayerName;
    m_topology->add(psROIPoolingPrim);
    m_env.profilingIDs.push_back(psROIPoolingLayerName);
}

void CLDNNGraph::CreateCustomLayerPrimitive(InferenceEngine::CNNLayerPtr & layer, CLDNNCustomLayerPtr customLayer) {
    ValidateLayer(layer, 0);
    // todo: handling fusing
    auto genericLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    // Handle defines
    std::string layerDefines;
    for (const auto& def : customLayer->Defines()) {
        std::string singleDefine("#define " + def.name + " " + def.prefix);
        if (genericLayer->params.find(def.param) != genericLayer->params.end()) {
            singleDefine += genericLayer->params.at(def.param);
        } else {
            singleDefine += def.default_value;
        }
        singleDefine += def.postfix + "\n";
        layerDefines.append(singleDefine);
    }

    // reserve
    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    // Handle Blobs
    std::map<std::string, size_t> blobIndex;
    for (auto& blob : genericLayer->blobs) {
        // create primitive from blob (always 1d)
        cldnn::primitive_id blobId = genericLayer->name + "_" + blob.first;
        if (blob.second->dims().size() != 1) {
            THROW_CLDNN_EXCEPTION("Invalid dimensions for blob " << blob.first << " in layer " << genericLayer->name);
        }
        CreatePrimitiveFromBlob(blobId, blob.second, cldnn::layout(
            DataTypeFromPrecision(blob.second->precision()),
            m_defaultFormat,
            cldnn::tensor(1, 1, TensorValue(blob.second->dims()[0]), 1)));
        // save index in blobIndex
        blobIndex[blob.first] = reorderedInputs.size();
        // add to reorderedInputs
        reorderedInputs.push_back(blobId);
    }

    // Handle kernel parameters
    std::vector<cldnn_arg> kernelParameters;
    cldnn::format outputFormat(cldnn::format::any);
    for (const auto& param : customLayer->KernelParams()) {
        switch (param.type) {
        case CLDNNCustomLayer::ParamType::Input: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].arg_type = cldnn_arg_type::arg_input;
            kernelParameters[param.paramIndex].index = static_cast<cldnn_arg_index>((param.portIndex >= inputPrimitives.size()) ? -1 : param.portIndex);

            // Handle input reorder
            if (param.portIndex < inputPrimitives.size() && reorderedInputs[param.portIndex].empty()) {
                // todo: add support for multiple reorders of the same input? (read as bfyx for one arg and yxfb for another)
                if (param.format != cldnn::format::any) {
                    auto reorderPrimName = inputPrimitives[param.portIndex] + "_" + layer->name + m_preCustomLayerTag;
                    auto preprocessPrim = cldnn::reorder(
                        reorderPrimName,
                        inputPrimitives[param.portIndex],
                        param.format,
                        DataTypeFromPrecision(layer->precision));
                    m_topology->add(preprocessPrim);
                    m_env.profilingIDs.push_back(reorderPrimName);
                    InitProfileInfo(reorderPrimName, "Reorder");
                    reorderedInputs[param.portIndex] = (reorderPrimName);
                } else {
                    reorderedInputs[param.portIndex] = inputPrimitives[param.portIndex];
                }
            }
        }
            break;
        case CLDNNCustomLayer::ParamType::Output: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].arg_type = cldnn_arg_type::arg_output;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn_arg_index>((param.portIndex >= inputPrimitives.size()) ? -1 : param.portIndex);
            outputFormat = param.format;
        }
            break;
        case CLDNNCustomLayer::ParamType::Data: {
            kernelParameters.resize(kernelParameters.size() > size_t(param.paramIndex + 1) ? kernelParameters.size() : size_t(param.paramIndex + 1));
            kernelParameters[param.paramIndex].arg_type = cldnn_arg_type::arg_input;
            kernelParameters[param.paramIndex].index =
                static_cast<cldnn_arg_index>((blobIndex.find(param.blobName) == blobIndex.end()) ? -1 : blobIndex.at(param.blobName));
        }
            break;
        default:
            THROW_CLDNN_EXCEPTION("Invalid custom layer param type: " << param.type << " in layer: " << genericLayer->name);
        }
    }
    const std::string layerTitle("\n// Layer " + layer->name + " using Custom Layer " + customLayer->Name() + "\n");
    const std::string defineTitle("// Custom Layer User Defines\n");

    auto dims = genericLayer->outData[0]->dims;
    std::reverse(dims.begin(), dims.end());

    size_t N = (dims.size() > 0) ? dims[0] : 1;
    size_t C = (dims.size() > 1) ? dims[1] : 1;
    size_t H = (dims.size() > 2) ? dims[2] : 1;
    size_t W = (dims.size() > 3) ? dims[3] : 1;
    cldnn::tensor outputTensor = cldnn::tensor(cldnn::batch(N), cldnn::feature(C), cldnn::spatial(W, H));

    cldnn::layout outputLayout = cldnn::layout(DataTypeFromPrecision(genericLayer->precision), outputFormat, outputTensor);

    // evaluate work sizes rules
    std::vector<size_t> gws, lws;

    // assume output tensor is dimension source by default
    int batchDim = outputTensor.batch[0];
    int featureDim = outputTensor.feature[0];
    int yDim = outputTensor.spatial[1];
    int xDim = outputTensor.spatial[0];
    int iidx = customLayer->InputDimSourceIndex();

    std::string genericLayerName = layer_type_name_ID(layer);
    // if input index is greater than -1, take dimension from input
    if (iidx >= 0) {
        if (iidx >= genericLayer->insData.size())
            THROW_CLDNN_EXCEPTION("Invalid input tensor for index: " << iidx);
        // get dimensions from one of the input tensors
        auto inDataPtr = genericLayer->insData[iidx].lock();
        if (!inDataPtr) {
            THROW_CLDNN_EXCEPTION("Data inserted into generic layer " << genericLayer->name << " is nullptr");
        }
        auto inputDims = inDataPtr->dims;

        batchDim = featureDim = yDim = 0;
        xDim = inputDims[0];

        if (dims.size() > 1)
            yDim = inputDims[1];
        if (dims.size() > 2)
            featureDim = inputDims[2];
        if (dims.size() > 3)
            batchDim = inputDims[3];
    }
    const std::map<char, int> vars = {
        { 'b', batchDim }  , { 'B', batchDim },
        { 'f', featureDim }, { 'F', featureDim },
        { 'y', yDim },       { 'Y', yDim },
        { 'x', xDim },       { 'X', xDim },
    };
    for (auto rule : customLayer->GlobalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        gws.push_back(expr.Evaluate());
    }
    for (auto rule : customLayer->LocalSizeRules()) {
        SimpleMathExpression expr;
        expr.SetVariables(vars);
        expr.SetExpression(rule);
        lws.push_back(expr.Evaluate());
    }

    auto customPrim = cldnn::custom_gpu_primitive(
        genericLayerName,
        reorderedInputs,
        { layerTitle, defineTitle, layerDefines, customLayer->KernelSource() },
        customLayer->KernelEntry(),
        kernelParameters,
        customLayer->CompilerOptions(),
        outputLayout,
        gws,
        lws);

    if (outputLayout.format != cldnn::format::any &&
        p_currentOutputs->find(genericLayerName) == p_currentOutputs->end()) {
        // Handle output reorder
        auto reorderPrimName = genericLayerName + m_postCustomLayerTag;
        m_topology->add(
            cldnn::reorder(
                reorderPrimName,
                genericLayerName,
                m_defaultFormat,
                customPrim.output_layout.data_type));
        m_env.primitiveIDs[genericLayerName] = reorderPrimName;
        m_env.primitiveIDs[reorderPrimName] = reorderPrimName;
        m_env.profilingIDs.push_back(reorderPrimName);
        InitProfileInfo(reorderPrimName, "Reorder");
    } else {
        m_env.primitiveIDs[genericLayerName] = genericLayerName;
    }
    m_topology->add(customPrim);
    m_env.profilingIDs.push_back(genericLayerName);
}

void CLDNNGraph::CreateSimplerNMSPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 3);
    IE_ASSERT(layer->insData[0].lock()->dims[3] == 1);  // only handling input batch size 1
    IE_ASSERT(layer->insData[1].lock()->dims[3] == 1);  // only handling input batch size 1
    auto simpleNMSLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    int max_num_proposals = simpleNMSLayer->GetParamAsInt("max_num_proposals");
    float iou_threshold = simpleNMSLayer->GetParamAsFloat("iou_threshold", 0.7f);
    int min_bbox_size = simpleNMSLayer->GetParamAsInt("min_bbox_size", 16);
    int feature_stride = simpleNMSLayer->GetParamAsInt("feat_stride", 16);
    int pre_nms_topn = simpleNMSLayer->GetParamAsInt("pre_nms_topn");
    int post_nms_topn = simpleNMSLayer->GetParamAsInt("post_nms_topn");
    std::vector<float> scale = simpleNMSLayer->GetParamAsFloats("scale");
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::string simpleNMSLayerName = layer_type_name_ID(layer);
    auto simpleNMSPrim = cldnn::proposal(
        simpleNMSLayerName,
        inputPrimitives[0],  // cls_score
        inputPrimitives[1],  // bbox_pred
        inputPrimitives[2],  // im_info
        max_num_proposals,
        iou_threshold,
        min_bbox_size,
        feature_stride,
        pre_nms_topn,
        post_nms_topn,
        { 0.5f, 1.0f, 2.0f },  // ratios for the SimplerNMS variant
        scale);

    m_env.primitiveIDs[simpleNMSLayerName] = simpleNMSLayerName;
    m_topology->add(simpleNMSPrim);
    m_env.profilingIDs.push_back(simpleNMSLayerName);
}

void CLDNNGraph::CreateEltwisePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateEltwiseLayer(layer);

    auto eltwiseLayer = dynamic_cast<InferenceEngine::EltwiseLayer *> (layer.get());
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::vector<float> coefficients = eltwiseLayer->coeff;
    if (eltwiseLayer->_operation != InferenceEngine::EltwiseLayer::Sum && !coefficients.empty()) {
        THROW_IE_EXCEPTION << "Only sum operation supports operands coefficients";
    }

    if (!coefficients.empty() && coefficients.size() != inputPrimitives.size()) {
        THROW_IE_EXCEPTION << "Number of provided coefficients is not equal to number of operands";
    }

    std::string eltwiseLayerName = layer_type_name_ID(layer);
    auto eltwisePrim = cldnn::eltwise(
        eltwiseLayerName,
        inputPrimitives,
        EltwiseModeFromIEEltwise(eltwiseLayer->_operation),
        coefficients);
    m_env.primitiveIDs[eltwiseLayerName] = eltwiseLayerName;
    m_topology->add(eltwisePrim);
    m_env.profilingIDs.push_back(eltwiseLayerName);
}

void CLDNNGraph::CreateConcatenatePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 0);
    auto concatLayer = dynamic_cast<InferenceEngine::ConcatLayer *> (layer.get());
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    std::string concatLayerName = layer_type_name_ID(layer);
    auto concatPrim = cldnn::concatenation(
        concatLayerName,
        inputPrimitives,
        ConcatAxisFromIEAxis(concatLayer->_axis));
    m_env.primitiveIDs[concatLayerName] = concatLayerName;
    m_topology->add(concatPrim);
    m_env.profilingIDs.push_back(concatLayerName);
}

void CLDNNGraph::CreateSplitPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto splitLayer = dynamic_cast<InferenceEngine::SplitLayer *> (layer.get());
    if (IsValidSplitConvMerge(splitLayer)) {
        // AlextNet style split->conv*2->merge
        CreateFusedSplitConvMergePrimitive(layer);
    } else {
#ifdef _USE_SPLIT_PRIMITIVE
        auto inputPrimitives = GetPrevLayersPrimitives(layer);
        auto inputDims = splitLayer->insData[0].lock()->dims;
        InferenceEngine::SizeVector startOffset(inputDims.size());
        std::vector<std::pair<cldnn::primitive_id, cldnn::tensor>> outputOffsets;
std::cout << "Splitting layer: " << layer->name << "\n\tSize:" << CldnnTensorFromIEDims(inputDims) << std::endl;
        for (auto& outLayer : splitLayer->outData) {
            if (outLayer->dims.size() != startOffset.size()) {
                THROW_CLDNN_EXCEPTION("Invalid dimesions in split layer: " << splitLayer->name << " output: " << outLayer->name);
            }
            for (size_t i = 0; i < inputDims.size(); i++) {
                if ((outLayer->dims[i] + startOffset[i]) > inputDims[i]) {
                    THROW_CLDNN_EXCEPTION("Invalid dimesions in split layer: " << splitLayer->name << " output: " << outLayer->name);
                }
            }
            auto outTensor = CldnnTensorFromIEDims(outLayer->dims);
            auto cropPrim = cldnn::crop(outLayer->name, inputPrimitives[0], outTensor, CldnnTensorFromIEDims(startOffset));
            m_topology->add(cropPrim);
            m_env.primitiveIDs[outLayer->name] = outLayer->name;
            m_env.profilingIDs.push_back(outLayer->name);
            outputOffsets.push_back({ outLayer->name, CldnnTensorFromIEDims(startOffset) });
            for (size_t i = 0; i < inputDims.size(); i++) {
                if (outLayer->dims[i] != inputDims[i]) {
                    startOffset[i] += outLayer->dims[i];
                }
            }
        }

        auto splitPrim = cldnn::split(
            splitLayer->name,
            inputPrimitives[0],
            outputOffsets);
        m_topology->add(splitPrim);


        // set split as not_run
        InitProfileInfo(layer->name, layer->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);  // Mark this layer as optimized out

#else  // _USE_SPLIT_PRIMITIVE
        // TODO: replace with clDNN split when it's implemented
        auto inputPrimitives = GetPrevLayersPrimitives(layer);
        auto inDataPtr = splitLayer->insData[0].lock();
        if (!inDataPtr) {
            THROW_CLDNN_EXCEPTION("Data inserts into split layer " << splitLayer->name << " is nullptr");
        }
        auto inputDims = inDataPtr->dims;
        InferenceEngine::SizeVector startOffset(inputDims.size());

        auto TensorFromIEDims = [](const InferenceEngine::SizeVector& dims, int def) {
            switch (dims.size()) {
            case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
            case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
            case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
            case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
            default: THROW_CLDNN_EXCEPTION("Invalid dimensions size(" << dims.size() << ") in split layer");
            }
        };

        for (auto& outLayer : splitLayer->outData) {
            std::string outLayerName = splitLayer->type + ":" + outLayer->name;
            if (outLayer->dims.size() != startOffset.size()) {
                THROW_CLDNN_EXCEPTION("Invalid dimesions in split layer: " << splitLayer->name << " output: " << outLayer->name);
            }
            for (size_t i = 0; i < inputDims.size(); i++) {
                if ((outLayer->dims[i] + startOffset[i]) > inputDims[i]) {
                    THROW_CLDNN_EXCEPTION("Invalid dimesions in split layer: " << splitLayer->name << " output: " << outLayer->name);
                }
            }
            SizeVector reverseDims = outLayer->dims;
            std::reverse(reverseDims.begin(), reverseDims.end());
            auto outTensor = TensorFromIEDims(reverseDims, 1);

            SizeVector reverseOffset = startOffset;
            std::reverse(reverseOffset.begin(), reverseOffset.end());
            auto offsetTensor = TensorFromIEDims(reverseOffset, 0);

            auto cropPrim = cldnn::crop(outLayerName, inputPrimitives[0], outTensor, offsetTensor);
            m_env.primitiveIDs[outLayerName] = outLayerName;
            m_topology->add(cropPrim);
            m_env.profilingIDs.push_back(outLayerName);
            InitProfileInfo(outLayerName, "Crop");

            for (size_t i = 0; i < inputDims.size(); i++) {
                if (outLayer->dims[i] != inputDims[i]) {
                    startOffset[i] += outLayer->dims[i];
                }
            }
        }

        // set split as not_run
        InitProfileInfo(layer->name, layer->type, false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);  // Mark this layer as optimized out
#endif  // _USE_SPLIT_PRIMITIVE
    }
}

void CLDNNGraph::CreateFusedSplitConvMergePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    // only handle the split->conv->merge topology for now
    auto splitLayer = dynamic_cast<InferenceEngine::SplitLayer *> (layer.get());
    IE_ASSERT(IsValidSplitConvMerge(splitLayer));

    auto convLayer1 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[0]).get());
    auto convLayer2 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[1]).get());
    auto concatLayer =
        dynamic_cast<InferenceEngine::ConcatLayer *> (GetNextSingleLayer(
            GetNextSingleLayer(splitLayer->outData[0])).get());

    if (convLayer1 == nullptr ||
        convLayer2 == nullptr ||
        concatLayer == nullptr) {
        THROW_CLDNN_EXCEPTION("Expected single layer does not exist");
    }
    // Mark these layers as optimized out
    InitProfileInfo(convLayer1->name, convLayer1->type, false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);
    InitProfileInfo(convLayer2->name, convLayer2->type, false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);
    InitProfileInfo(concatLayer->name, concatLayer->type, false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);

    // build the split conv primitive
    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(GetNextSingleLayer(splitLayer->outData[0]), weightPrimID, biasPrimID);
    CreateWeightAndBiasPrimitives(GetNextSingleLayer(splitLayer->outData[1]), weightPrimID, biasPrimID);

    auto concatLayerPtr = std::make_shared<InferenceEngine::CNNLayer>(*concatLayer);

    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                         cldnn::spatial(convLayer1->_stride[X_AXIS], convLayer1->_stride[Y_AXIS]));
    auto allPad = getPaddings(*convLayer1);
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
                                          cldnn::spatial(-allPad.begin[X_AXIS], -allPad.begin[Y_AXIS]));
    cldnn::tensor dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                           cldnn::spatial(convLayer1->_dilation[X_AXIS], convLayer1->_dilation[Y_AXIS]));

    std::string splitLayerName = layer_type_name_ID(layer);
    auto splitPrim = cldnn::convolution(splitLayerName,
                                        inputPrimitives[0],
                                        weightPrimID,
                                        biasPrimID,
                                        stride,
                                        padding,
                                        dilation,
                                        false,
                                        0.0f,
                                        CldnnTensorFromIEDims(concatLayer->outData[0]->dims));

    layer = concatLayerPtr;

    m_env.primitiveIDs[splitLayerName]  = splitLayerName;
    m_env.primitiveIDs[layer_type_name_ID(convLayer1)]  = splitLayerName;
    m_env.primitiveIDs[layer_type_name_ID(convLayer2)]  = splitLayerName;
    m_env.primitiveIDs[layer_type_name_ID(concatLayer)] = splitLayerName;  // pair the last merged layer (concat or relu) with
                                                               // this primitive name to be used as
                                                              // input prim for subsequent layers
    m_topology->add(splitPrim);
    m_env.profilingIDs.push_back(splitLayerName);
}

void CLDNNGraph::CreatePowerPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer *> (layer.get());
    if (powerLayer->power != 1.0f && powerLayer->power != 0.5f) {
        THROW_CLDNN_EXCEPTION("Power Layer " << layer->name << "uses unsupported power value");
    }

    std::string powerLayerName = layer_type_name_ID(layer);
    if ((powerLayer->scale == 1.0f) && (powerLayer->offset == 0.0f)) {
        if (powerLayer->power == 0.5f) {
            auto activationPrim = cldnn::activation(powerLayerName, inputPrimitives[0], activation_sqrt);
            m_topology->add(activationPrim);
            m_env.profilingIDs.push_back(powerLayerName);
            m_env.primitiveIDs[powerLayerName] = powerLayerName;
        } else {
            // skip this layer
            m_env.primitiveIDs[powerLayerName] = inputPrimitives[0];  // register the previous primID for this layer too
            InitProfileInfo(layer->name, layer->type, false, InferenceEngine::InferenceEngineProfileInfo::NOT_RUN);  // Mark this layer as not run
        }
    } else {
        // create scale primitive
        auto scaleValuePrimName = powerLayerName + m_scalesTag;
        AddSingleValuePrimitive(scaleValuePrimName,
            DataTypeFromPrecision(powerLayer->precision),
            powerLayer->scale);

        cldnn::primitive_id biasValuePrimName = "";
        if (powerLayer->offset != 0.0f) {
            biasValuePrimName = powerLayerName + m_biasesTag;
            AddSingleValuePrimitive(biasValuePrimName,
                DataTypeFromPrecision(powerLayer->precision),
                powerLayer->offset);
        }
        auto scalePrim = cldnn::scale(
            powerLayerName,
            inputPrimitives[0],
            scaleValuePrimName,
            biasValuePrimName);

        m_env.primitiveIDs[powerLayerName] = powerLayerName;
        m_topology->add(scalePrim);
        m_env.profilingIDs.push_back(powerLayerName);

        if (powerLayer->power == 0.5f) {
            auto activationPrim = cldnn::activation(powerLayerName+"_sqrt", powerLayerName, activation_sqrt);
            m_topology->add(activationPrim);
            m_env.profilingIDs.push_back(powerLayerName+"_sqrt");
        }
    }
}

void CLDNNGraph::CreateSoftMaxPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto softmaxLayer = dynamic_cast<InferenceEngine::SoftMaxLayer *> (layer.get());

    // additional WA for clDNN FullyConnected output in BX instead of BF
    int inputOrder = 0;
    auto prevData = layer->insData[0].lock();

    if (prevData == nullptr) {
        THROW_CLDNN_EXCEPTION("SoftMax: nonexistent input for layer: " << layer->name);
    }

    auto prevCreator = prevData->creatorLayer.lock();
    bool isPrevFC = false;

    if (prevCreator && (LayerTypeFromStr(prevCreator->type) == FullyConnected))
        isPrevFC = true;
    // end of WA

    std::string softmaxLayerName = layer_type_name_ID(layer);
    auto softmaxPrim = cldnn::softmax(softmaxLayerName, inputPrimitives[0], SoftmaxDimensionFromIEAxis(softmaxLayer, isPrevFC));
    m_env.primitiveIDs[softmaxLayerName] = softmaxLayerName;
    m_topology->add(softmaxPrim);
    m_env.profilingIDs.push_back(softmaxLayerName);
}

void CLDNNGraph::CreateFullyConnectedPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto fcLayer = dynamic_cast<InferenceEngine::FullyConnectedLayer *> (layer.get());

    std::string fcLayerName = layer_type_name_ID(layer);
    // create bias primitive
    cldnn::primitive_id biasesPrimID = "";
    if (fcLayer->_biases != nullptr) {
        biasesPrimID = fcLayerName + m_biasesTag;
        CreatePrimitiveFromBlob(biasesPrimID,
            fcLayer->_biases,
            cldnn::layout(DataTypeFromPrecision(fcLayer->precision), m_defaultFormat,
                cldnn::spatial(TensorValue(fcLayer->_out_num))));
    }

    // create weights primitive
    // gcc bug to resolve auto, at least for 5.4 version
    std::shared_ptr<Data> insData0 = fcLayer->insData[0].lock();
    IE_ASSERT(insData0 != nullptr);
    cldnn::primitive_id weightsPrimID = fcLayerName + m_weightsTag;
    cldnn::tensor weightsDims;
    switch (insData0->dims.size()) {
    case 4:
        weightsDims = { TensorValue(fcLayer->outData[0]->dims[0]),
                        TensorValue(insData0->dims[2]),
                        TensorValue(insData0->dims[0]),
                        TensorValue(insData0->dims[1]) };
        break;
    case 2:
        weightsDims = { TensorValue(fcLayer->outData[0]->dims[0]), TensorValue(insData0->dims[0]), 1, 1 };
        break;
    default: THROW_CLDNN_EXCEPTION("Invalid data dimensions");
    }
    CreatePrimitiveFromBlob(weightsPrimID,
                            fcLayer->_weights,
                            cldnn::layout(DataTypeFromPrecision(fcLayer->precision), m_defaultFormat, weightsDims));

    auto fcPrim = cldnn::fully_connected(fcLayerName,
                                         inputPrimitives[0],
                                         weightsPrimID,
                                         biasesPrimID,
                                         false,
                                         0.0f);

    m_env.primitiveIDs[fcLayerName] = fcLayerName;
    m_topology->add(fcPrim);
    m_env.profilingIDs.push_back(fcLayerName);
}

void CLDNNGraph::CreatePoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto poolLayer = dynamic_cast<InferenceEngine::PoolingLayer *> (layer.get());

    std::string poolLayerName = layer_type_name_ID(layer);
    auto allPads = getPaddings(*poolLayer);
    if (poolLayer->outData.size() > 1) {
        // max pooling with argmax
        SizeVector argmaxDims;

        std::string realOutputID, argmaxOutputID;
        int outputOrder = 0;

        for (auto out : poolLayer->outData) {
            auto layersMap = out->getInputTo();

            for (auto item : layersMap) {
                bool isUpooling = (LayerTypeFromStr(item.second->type) == Unpooling);
                if (outputOrder == 1 && isUpooling) {
                    argmaxDims = out->dims;
                    argmaxOutputID = out->name;
                } else {
                    realOutputID = out->name;
                }
                outputOrder++;
            }
        }

        // create mutable_data primitive for storing argmax data
        cldnn::tensor mutableTensor;
        switch (argmaxDims.size()) {
        case 4: mutableTensor = cldnn::tensor(TensorValue(argmaxDims[3]), TensorValue(argmaxDims[2]),
            TensorValue(argmaxDims[0]), TensorValue(argmaxDims[1]));
            break;
        case 3: mutableTensor = cldnn::tensor(TensorValue(argmaxDims[2]), TensorValue(argmaxDims[1]),
            1, TensorValue(argmaxDims[0]));
            break;
        case 2: mutableTensor = cldnn::tensor(TensorValue(argmaxDims[1]), TensorValue(argmaxDims[0]), 1, 1);
            break;
        case 1:  // not implemented yet.
        default: THROW_CLDNN_EXCEPTION("Invalid constant blob dimensions");
        }

        cldnn::layout mutableLayout = cldnn::layout(
            cldnn::data_types::f32,
            m_defaultFormat,
            mutableTensor);

        cldnn::primitive_id argmaxPrimID = layer->name + "_argmax_mutable";

        auto mem = cldnn::memory::allocate(*(m_env.engine), mutableLayout);
        auto argmax_mutable_prim = cldnn::mutable_data(argmaxPrimID, mem);
        m_topology->add(argmax_mutable_prim);
        m_env.primitiveIDs[argmaxPrimID] = argmaxPrimID;
        m_env.primitiveIDs[argmaxOutputID] = argmaxPrimID;

        // create pooling primitive itself
        auto poolPrim = cldnn::pooling(poolLayerName,
            inputPrimitives[0],
            argmaxPrimID,
            cldnn::pooling_mode::max_with_argmax,
            cldnn::spatial(TensorValue(poolLayer->_kernel[X_AXIS]), TensorValue(poolLayer->_kernel[Y_AXIS])),  // size
            cldnn::spatial(TensorValue(poolLayer->_stride[X_AXIS]), TensorValue(poolLayer->_stride[Y_AXIS])),  // stride
                                                                                                   // input offset (padding) - explicit tensor for 0 bf
            { 0, 0, -TensorValue(allPads.begin[X_AXIS]), -TensorValue(allPads.begin[Y_AXIS]) },
            CldnnTensorFromIEDims(poolLayer->outData[0]->dims));
        m_topology->add(poolPrim);
        m_env.primitiveIDs[realOutputID] = poolLayerName;
    } else {
        // regular pooling
        auto poolPrim = cldnn::pooling(poolLayerName,
            inputPrimitives[0],
            PoolingModeFromIEPooling(poolLayer->_type, poolLayer->_exclude_pad),
            cldnn::spatial(TensorValue(poolLayer->_kernel[X_AXIS]), TensorValue(poolLayer->_kernel[Y_AXIS])),  // size
            cldnn::spatial(TensorValue(poolLayer->_stride[X_AXIS]), TensorValue(poolLayer->_stride[Y_AXIS])),  // stride
                                                                                                   // input offset (padding) - explicit tensor for 0 bf
            { 0, 0, -TensorValue(allPads.begin[X_AXIS]), -TensorValue(allPads.begin[Y_AXIS]) },
            CldnnTensorFromIEDims(poolLayer->outData[0]->dims));
    m_topology->add(poolPrim);
        m_env.primitiveIDs[poolLayerName] = poolLayerName;
    }

    m_env.profilingIDs.push_back(poolLayerName);
}

void CLDNNGraph::CreateLRNPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto lrnLayer = dynamic_cast<InferenceEngine::NormLayer *> (layer.get());
    std::string lrnLayerName = layer_type_name_ID(layer);
    auto lrnPrim = cldnn::lrn(
        lrnLayerName,
        inputPrimitives[0],
        lrnLayer->_size,
        static_cast<float>(lrnLayer->_k),
        lrnLayer->_alpha,
        lrnLayer->_beta,
        lrnLayer->_isAcrossMaps ? cldnn_lrn_norm_region_across_channel : cldnn_lrn_norm_region_within_channel);

    m_env.primitiveIDs[lrnLayerName] = lrnLayerName;
    m_topology->add(lrnPrim);
    m_env.profilingIDs.push_back(lrnLayerName);
}

void CLDNNGraph::CreateActivationPrimitive(InferenceEngine::CNNLayerPtr &layer, const LayerType type) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    cldnn_activation_additional_params params{ 0.0f, 0.0f };
    cldnn_activation_func func = cldnn_activation_func_t::activation_none;

    LayerType activationType;
    if (type == Activation) {
        std::string activation_type = layer->GetParamAsString("type");
        if (activation_type == "tanh") {
            activationType = TanH;
        } else if (activation_type == "sigmoid" || activation_type == "logistic")  {
            activationType = Sigmoid;
        } else if (activation_type == "elu")  {
            activationType = ELU;
        } else if (activation_type == "relu")  {
            activationType = ReLU;
        } else if (activation_type == "relu6")  {
            activationType = ReLU6;
        } else if (activation_type == "clamp")  {
            activationType = Clamp;
        } else if (activation_type == "exp")  {
            activationType = Exp;
        } else if (activation_type == "not")  {
            activationType = Not;
        } else {
            THROW_CLDNN_EXCEPTION("Unsupported activation type (" + activation_type +
                                  ") in layer " + layer->name);
        }
    } else {
        activationType = type;
    }

    switch (activationType) {
    case TanH:
    {
        func = cldnn_activation_func_t::activation_hyperbolic_tan;
        break;
    }
    case ELU:
    {
        func = cldnn_activation_func_t::activation_elu;
        params.a = layer->GetParamAsFloat("alpha", 1.0f);
        break;
    }
    case Sigmoid:
    {
        func = cldnn_activation_func_t::activation_logistic;
        break;
    }
    case ReLU:
    {
        func = cldnn_activation_func_t::activation_relu_negative_slope;
        params.a = layer->GetParamAsFloat("negative_slope", 0.0f);
        break;
    }
    case ReLU6:
    {
        func = cldnn_activation_func_t::activation_clamp;
        params.b = layer->GetParamAsFloat("n", 6.0f);
        break;
    }
    case Clamp:
    {
        func = cldnn_activation_func_t::activation_clamp;
        params.a = layer->GetParamAsFloat("min");
        params.b = layer->GetParamAsFloat("max");
        break;
    }
    case Exp:
    {
        func = cldnn_activation_func_t::activation_exp;
        break;
    }
    case Not:
    {
        func = cldnn_activation_func_t::activation_not;
        break;
    }
    default:
        THROW_CLDNN_EXCEPTION("Unsupported activation type (" + layer->type +
                              ") in layer " + layer->name);
    }

    std::string layerName = layer_type_name_ID(layer);
    auto activationPrimitive = cldnn::activation(layerName, inputPrimitives[0], func, params);
    m_env.primitiveIDs[layerName] = layerName;
    m_topology->add(activationPrimitive);
    m_env.profilingIDs.push_back(layerName);
}

void CLDNNGraph::CreateCopyPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto copyLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // Optimize out and just update references
    std::string layerName = layer_type_name_ID(layer);
    m_env.primitiveIDs[layerName] = inputPrimitives[0];
    InitProfileInfo(layerName, layer->type, false, InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);  // Mark this layer as optimized out
}

void CLDNNGraph::CreateUpsamplingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    // Assuming multi-input will be handled by prev concat/eltwise layers
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto upsamplingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    uint32_t scale = upsamplingLayer->GetParamAsUInt("scale");
    uint32_t numFilter = upsamplingLayer->GetParamAsUInt("num_filter");
    std::string sampleType = upsamplingLayer->GetParamAsString("sample_type");

    std::string upsamplingLayerName = layer_type_name_ID(layer);
    auto upsamplingPrim = cldnn::upsampling(
        upsamplingLayerName,
        inputPrimitives[0],
        scale,
        numFilter,
        UpsamplingTypeFromString(sampleType));

    m_env.primitiveIDs[upsamplingLayerName] = upsamplingLayerName;
    m_topology->add(upsamplingPrim);
    m_env.profilingIDs.push_back(upsamplingLayerName);
}

void CLDNNGraph::CreateResamplePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto resampleLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    auto outDims = layer->outData[0]->dims;
    size_t inFeatures = 1;
    unsigned int scale = 1;
    std::shared_ptr<Data> insData0 = layer->insData[0].lock();
    IE_ASSERT(insData0 != nullptr);
    if (insData0->dims.size() > 2) {
        inFeatures = insData0->dims[2];
        scale = outDims[0]/insData0->dims[0];
        if (scale < 1) {
            THROW_CLDNN_EXCEPTION("Unsupported scale in layer " + layer->name);
        }
    }
    std::string sampleType = resampleLayer->GetParamAsString("type");

    if (sampleType != "caffe.ResampleParameter.NEAREST") {
        THROW_CLDNN_EXCEPTION("Unsupported resampling type (" + sampleType + ") in layer " + layer->name);
    }

    std::string resampleLayerName = layer_type_name_ID(layer);
    auto upsamplingPrim = cldnn::upsampling(
        resampleLayerName,
        inputPrimitives[0],
        scale,
        inFeatures,
        cldnn::upsampling_sample_type::nearest);

    m_env.primitiveIDs[resampleLayerName] = resampleLayerName;
    m_topology->add(upsamplingPrim);
    m_env.profilingIDs.push_back(resampleLayerName);
}

void CLDNNGraph::CreateYOLO2RegionPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto YOLOregionLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    uint32_t coords = YOLOregionLayer->GetParamAsUInt("coords", 4);
    uint32_t classes = YOLOregionLayer->GetParamAsUInt("classes", 20);
    uint32_t num = YOLOregionLayer->GetParamAsUInt("num", 1);
    bool do_softmax = YOLOregionLayer->GetParamsAsBool("do_softmax", true);

    uint32_t mask_size = 0;
    if (HasParam(YOLOregionLayer->params, "mask")) {
        const auto mask = YOLOregionLayer->GetParamAsInts("mask");
        mask_size = static_cast<uint32_t>(mask.size());
    }

    std::string YOLOregionLayerName = layer_type_name_ID(layer);
    auto regionPrim = cldnn::region_yolo(
        YOLOregionLayerName,
        inputPrimitives[0],
        coords,
        classes,
        num,
        mask_size,
        do_softmax);

    m_env.primitiveIDs[YOLOregionLayerName] = YOLOregionLayerName;
    m_topology->add(regionPrim);
    m_env.profilingIDs.push_back(YOLOregionLayerName);
}

void CLDNNGraph::CreateYOLO2ReorgPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto YOLOreorgLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    uint32_t stride = YOLOreorgLayer->GetParamAsUInt("stride");

    std::string YOLOreorgLayerName = layer_type_name_ID(layer);
    auto reorgPrim = cldnn::reorg_yolo(
        YOLOreorgLayerName,
        inputPrimitives[0],
        stride);

    m_env.primitiveIDs[YOLOreorgLayerName] = YOLOreorgLayerName;
    m_topology->add(reorgPrim);
    m_env.profilingIDs.push_back(YOLOreorgLayerName);
}

void CLDNNGraph::CreateArgMaxPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto ArgMaxLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    const cldnn::arg_max_min::out_type otype = cldnn::arg_max_min::out_type::max;

    if (HasParam(ArgMaxLayer->params, "out_max_val")) {
        int32_t out_max_val_flag = ArgMaxLayer->GetParamAsInt("out_max_val");
        if (out_max_val_flag != 0) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << "ArgMax: out_max_val param is not supported for layer: " << layer->name;
        }
    }

    uint32_t top_k = ArgMaxLayer->GetParamAsUInt("top_k", 1);

    cldnn::arg_max_min::axis_name chosen_axis = cldnn::arg_max_min::axis_name::xyf;

    if (HasParam(ArgMaxLayer->params, "axis")) {
        int32_t axis_param = ArgMaxLayer->GetParamAsInt("axis", 1);

        int32_t axis = axis_param;
        if (-4 <= axis && axis <= -1)
            axis += 4;

        switch (axis) {
        case 0: chosen_axis = cldnn::arg_max_min::axis_name::batch; break;
        case 1: chosen_axis = cldnn::arg_max_min::axis_name::feature; break;
        case 2: chosen_axis = cldnn::arg_max_min::axis_name::y; break;
        case 3: chosen_axis = cldnn::arg_max_min::axis_name::x; break;
        }
    }

    std::string ArgMaxLayerName = layer_type_name_ID(layer);
    auto argmaxPrim = cldnn::arg_max_min(
        ArgMaxLayerName,
        inputPrimitives[0],
        otype,
        top_k,
        chosen_axis);

    m_env.primitiveIDs[ArgMaxLayerName] = ArgMaxLayerName;
    m_topology->add(argmaxPrim);
    m_env.profilingIDs.push_back(ArgMaxLayerName);
}

void CLDNNGraph::CreateMaxUnpoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);

    auto UnpoolingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    cldnn::primitive_id real_input, argmax_mutable;

    // locate ArgMax primitive
    int inputOrder = 0;
    for (auto inputData : layer->insData) {
        auto prevData = inputData.lock();

        if (prevData == nullptr) {
            THROW_CLDNN_EXCEPTION("MaxUnpooling: nonexistent input for layer: " << layer->name);
        }

        auto prevCreator = prevData->creatorLayer.lock();

        if (prevCreator &&
            (LayerTypeFromStr(prevCreator->type) == Pooling) &&
            prevCreator->outData.size() > 1 &&
            inputOrder == 1) {
            argmax_mutable = m_env.primitiveIDs.at(prevCreator->name + "_argmax_mutable");
        } else {
            real_input = m_env.primitiveIDs.at(prevData->name);
        }
        inputOrder++;
    }

    uint32_t stride = UnpoolingLayer->GetParamAsUInt("stride");
    uint32_t kernel_size = UnpoolingLayer->GetParamAsUInt("kernel_size");

    std::string UnpoolingLayerName = layer_type_name_ID(layer);
    auto unpoolingPrim = cldnn::max_unpooling(
        UnpoolingLayerName,
        real_input,
        argmax_mutable,
        cldnn::spatial(kernel_size, kernel_size),  // size
        cldnn::spatial(stride, stride) );          // stride

    m_env.primitiveIDs[UnpoolingLayerName] = UnpoolingLayerName;
    m_topology->add(unpoolingPrim);
    m_env.profilingIDs.push_back(UnpoolingLayerName);
}

void CLDNNGraph::CreateMVNPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto MvnLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    bool across_channels = MvnLayer->GetParamsAsBool("across_channels", false);
    bool normalize_variance = MvnLayer->GetParamsAsBool("normalize_variance", true);
    float eps = MvnLayer->GetParamAsFloat("eps", 1e-10f);

    std::string MvnLayerName = layer_type_name_ID(layer);
    auto mvnPrim = cldnn::mvn(
        MvnLayerName,
        inputPrimitives[0],
        across_channels,
        normalize_variance,
        eps);

    m_env.primitiveIDs[MvnLayerName] = MvnLayerName;
    m_topology->add(mvnPrim);
    m_env.profilingIDs.push_back(MvnLayerName);
}

void CLDNNGraph::CreateTilePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto tileLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    int axis = tileLayer->GetParamAsInt("axis", 1);
    int tiles = tileLayer->GetParamAsInt("tiles");

    auto cldnnAxisFromIE = [](int axis) {
        switch (axis) {
            case 0: return cldnn::tile::tile_axis::along_b;
            case 1: return cldnn::tile::tile_axis::along_f;
            case 2: return cldnn::tile::tile_axis::along_y;
            case 3: return cldnn::tile::tile_axis::along_x;
            default: THROW_CLDNN_EXCEPTION("Unsupported tile axis: " << axis);
        }
    };
    std::string tileLayerName = layer_type_name_ID(layer);
    auto tilePrim = cldnn::tile(
        tileLayerName,
        inputPrimitives[0],
        cldnnAxisFromIE(axis),
        tiles);

    m_env.primitiveIDs[tileLayerName] = tileLayerName;
    m_topology->add(tilePrim);
    m_env.profilingIDs.push_back(tileLayerName);
}

void CLDNNGraph::CreatePadPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto padLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    auto PadTensorFromArgs = [](const std::string &s) -> cldnn::tensor {
        std::stringstream ss(s);
        std::string item;
        std::vector<cldnn::tensor::value_type> elems;
        while (std::getline(ss, item, ',')) {
            elems.push_back(static_cast<cldnn::tensor::value_type>(std::atoll(item.c_str())));
        }

        while (elems.size() < 4) {
            elems.push_back(0);
        }

        // Swap x and y
        auto tmp = elems[2];
        elems[2] = elems[3];
        elems[3] = tmp;

        return cldnn::tensor(elems, 0);
    };

    auto pads_begin = PadTensorFromArgs(padLayer->GetParamAsString("pads_begin"));
    auto pads_end = PadTensorFromArgs(padLayer->GetParamAsString("pads_end"));
    std::string mode = padLayer->GetParamAsString("pad_mode");
    float pad_value = padLayer->GetParamAsFloat("pad_value", 0.0f);

    cldnn::border_type border_mode;
    if (mode == "constant")
        border_mode = cldnn::border_type::constant;
    else if (mode == "edge")
        border_mode = cldnn::border_type::edge;
    else if (mode == "symmetric")
        border_mode = cldnn::border_type::mirror;
    else if (mode == "reflect")
        border_mode = cldnn::border_type::mirror_101;
    else
        THROW_CLDNN_EXCEPTION("Invalid border mode " << mode << " in layer " << padLayer->name);

    std::string padLayerName = layer_type_name_ID(layer);
    auto tilePrim = cldnn::border(
            padLayerName,
            inputPrimitives[0],
            pads_begin,
            pads_end,
            border_mode,
            pad_value);

    m_env.primitiveIDs[padLayerName] = padLayerName;
    m_topology->add(tilePrim);
    m_env.profilingIDs.push_back(padLayerName);
}

std::string get_string_id(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

void CLDNNGraph::CreateLSTMCellPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    int lstm_batch_size, lstm_sequence_len, lstm_input_size, lstm_hidden_size;
    SizeVector in_dims1, in_dims2;
    bool hasBias = false;
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto elementSize = cldnn::data_type_traits::size_of(DataTypeFromPrecision(layer->precision));
    std::string layerName = layer_type_name_ID(layer);
    cldnn::primitive_id weightID = layerName + m_weightsTag;
    cldnn::primitive_id recurrentID = layerName + "_recurrent" + m_weightsTag;
    cldnn::primitive_id biasID = layerName + m_biasesTag;
    auto cellLayer = dynamic_cast<InferenceEngine::LSTMCell*> (layer.get());

    /* check incoming CNN layer and setup required variables */
    {
        auto in_data0 = layer->insData[0].lock();
        if (!in_data0)
            THROW_IE_EXCEPTION << "Missing first input for LSTMCell layer " << layer->name;

        auto in_dims0 = in_data0->dims;
        auto out_dims0 = layer->outData[0]->dims;

        lstm_input_size = in_dims0[0];
        lstm_batch_size = in_dims0[1];
        lstm_hidden_size = out_dims0[0];

        /* do we have initial hidden and cell?
        if blobs are not null, direct the data from them
        into corresponding LSTM inputs */

        auto in_data1 = layer->insData[1].lock();
        if (!in_data1)
            THROW_IE_EXCEPTION << "Missing second input for LSTMCell layer " << layer->name;
        in_dims1 = in_data1->dims;


        auto in_data2 = layer->insData[2].lock();
        if (!in_data2)
            THROW_IE_EXCEPTION << "Missing third input for LSTMCell layer " << layer->name;
        in_dims2 = in_data2->dims;


        if (in_dims0.size() != 2 || in_dims1.size() != 2 || in_dims2.size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for LSTMCell Layer " << layer->name;
    }

    /* Prepare weight/bias memory primitives - split weight blob into W and R */
    {
        cldnn::tensor wTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(lstm_input_size, 4 * lstm_hidden_size));
        cldnn::tensor rTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(lstm_hidden_size, 4 * lstm_hidden_size));
        cldnn::layout WLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, wTensor);
        cldnn::layout RLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, rTensor);

        auto wmem = cldnn::memory::allocate(*(m_env.engine), WLayout);
        auto wtmpPointer = wmem.pointer<char>();  // implicitly maps buffer - unmap in destructor

        auto rmem = cldnn::memory::allocate(*(m_env.engine), RLayout);
        auto rtmpPointer = rmem.pointer<char>();

        auto wLayer = dynamic_cast<InferenceEngine::WeightableLayer *> (layer.get());
        auto pWeightsBlob = wLayer->_weights;
        auto blobBytes = static_cast<const char *>(pWeightsBlob->buffer());
        const size_t WchunkSz = lstm_input_size * elementSize;
        const size_t RchunkSz = lstm_hidden_size * elementSize;

        auto wBytes = wtmpPointer.data();
        auto rBytes = rtmpPointer.data();

        for (int h = 0; h < 4 * lstm_hidden_size; h++) {
            // copy "input size" elements to W
            for (size_t b = 0; b < WchunkSz; b++)
                *wBytes++ = *blobBytes++;

            // copy "lstm_hidden_size" elements to R
            for (size_t b = 0; b < RchunkSz; b++)
                *rBytes++ = *blobBytes++;
        }

        m_topology->add(cldnn::data(weightID, wmem));
        m_topology->add(cldnn::data(recurrentID, rmem));

        /* create bias memory primitive */
        auto pBiasBlob = wLayer->_biases;
        if (pBiasBlob != nullptr) {
            cldnn::tensor bTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(4 * lstm_hidden_size, 1));
            cldnn::layout BLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, rTensor);

            auto bmem = cldnn::memory::allocate(*(m_env.engine), BLayout);
            auto btmpPointer = bmem.pointer<char>();

            auto blobBytes = static_cast<const char *>(pBiasBlob->buffer());
            const size_t BchunkSz = lstm_hidden_size * elementSize;
            auto bBytes = btmpPointer.data();

            for (size_t b = 0; b < 4 * BchunkSz; b++)
                *bBytes++ = *blobBytes++;

            m_topology->add(cldnn::data(biasID, bmem));
            hasBias = true;
        }
    }

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";

    cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), cldnn::format::bfyx, inputShape);
    m_topology->add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    m_topology->add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    std::string hiddenInStr = inHiddenReshapeID + "_1";
    std::string cellInStr = inHiddenReshapeID + "_2";
    m_topology->add(cldnn::reshape(hiddenInStr, inputPrimitives[1], hiddenStateShape));
    m_topology->add(cldnn::reshape(cellInStr, inputPrimitives[2], hiddenStateShape));

    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_gemm_id = layerName + "_lstm_gemm";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    m_topology->add(cldnn::lstm_gemm(lstm_gemm_id, permuteID,
                                     weightID, recurrentID,
                                     hasBias ? biasID : "",
                                     hiddenInStr));
    m_topology->add(cldnn::lstm_elt(lstm_elt_id, lstm_gemm_id, cellInStr,
                                    0, 0, {}, {}, cldnn_lstm_offset_order_fizo));

    cldnn::primitive_id outputHiddenID = layerName;
    m_topology->add(cldnn::crop(outputHiddenID, lstm_elt_id, hiddenSz, cldnn::tensor{0, 0, 0, 0}));
    cldnn::primitive_id outputCellID = layer->type + ":" + layer->outData[1]->name;
    m_topology->add(cldnn::crop(outputCellID, lstm_elt_id, hiddenSz, cellCropSz));

    // output primitive IDs
    m_env.primitiveIDs[outputHiddenID] = outputHiddenID;                                // LSTMCell:LSTMCell - "concat hidden"
    m_env.primitiveIDs[layer->type + ":" + layer->outData[0]->name] = outputHiddenID;   // LSTMCell:LSTMCell:0 - hidden state
    m_env.primitiveIDs[outputCellID] = outputCellID;                                    // LSTMCell:LSTMCell:1 - cell state

    m_env.profilingIDs.push_back(layerName);
}

void CLDNNGraph::CreateRNNPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    int lstm_batch_size, lstm_sequence_len, lstm_input_size, lstm_hidden_size;
    SizeVector in_dims1, in_dims2;
    bool hasInitialHidden = false, hasInitialCell = false, hasBias = false, isForward = true;
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto elementSize = cldnn::data_type_traits::size_of(DataTypeFromPrecision(layer->precision));
    std::string layerName = layer_type_name_ID(layer);
    cldnn::primitive_id weightID = layerName + m_weightsTag;
    cldnn::primitive_id recurrentID = layerName + "_recurrent" + m_weightsTag;
    cldnn::primitive_id biasID = layerName + m_biasesTag;
    auto rnnLayer = dynamic_cast<InferenceEngine::RNNSequenceLayer*> (layer.get());
    bool permute_input = (1 != rnnLayer->axis);

    /* check incoming CNN layer and setup required variables */
    {
        if (rnnLayer->cellType != RNNSequenceLayer::LSTM)
         THROW_IE_EXCEPTION << "RNN layer supports only LSTM like cell";

        auto in_data0 = layer->insData[0].lock();
        if (!in_data0)
            THROW_IE_EXCEPTION << "Missing first input for RNN layer " << layer->name;

        auto in_dims0 = in_data0->dims;
        auto out_dims0 = layer->outData[0]->dims;

        if (!permute_input) {
            lstm_batch_size = in_dims0[2];
            lstm_sequence_len = in_dims0[1];
        } else {
            lstm_batch_size = in_dims0[1];
            lstm_sequence_len = in_dims0[2];
        }

        lstm_input_size = in_dims0[0];
        lstm_hidden_size = out_dims0[0];

        /* do we have initial hidden and cell?
        if blobs are not null, direct the data from them
        into corresponding LSTM inputs */

        auto in_data1 = layer->insData[1].lock();
        if (in_data1) {
            in_dims1 = in_data1->dims;
            hasInitialHidden = true;
        }

        auto in_data2 = layer->insData[2].lock();
        if (in_data2) {
            in_dims2 = in_data2->dims;
            hasInitialCell = true;
        }

        if (rnnLayer->direction != RNNSequenceLayer::FWD && rnnLayer->direction != RNNSequenceLayer::BWD)
            THROW_IE_EXCEPTION << "Support only forward and backward direction for RNN Layer " << layer->name;
        isForward = rnnLayer->direction == RNNSequenceLayer::FWD;

        if (in_dims0.size() != 3 || in_dims1.size() != 2 || in_dims2.size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for RNN Layer " << layer->name;
    }

    /* Prepare weight/bias memory primitives - split weight blob into W and R */
    {
        cldnn::tensor wTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(lstm_input_size, 4 * lstm_hidden_size));
        cldnn::tensor rTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(lstm_hidden_size, 4 * lstm_hidden_size));
        cldnn::layout WLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, wTensor);
        cldnn::layout RLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, rTensor);

        auto wmem = cldnn::memory::allocate(*(m_env.engine), WLayout);
        auto wtmpPointer = wmem.pointer<char>();  // implicitly maps buffer - unmap in destructor

        auto rmem = cldnn::memory::allocate(*(m_env.engine), RLayout);
        auto rtmpPointer = rmem.pointer<char>();

        auto wLayer = dynamic_cast<InferenceEngine::WeightableLayer *> (layer.get());
        auto pWeightsBlob = wLayer->_weights;
        auto blobBytes = static_cast<const char *>(pWeightsBlob->buffer());
        const size_t WchunkSz = lstm_input_size * elementSize;
        const size_t RchunkSz = lstm_hidden_size * elementSize;

        auto wBytes = wtmpPointer.data();
        auto rBytes = rtmpPointer.data();

        for (int h = 0; h < 4 * lstm_hidden_size; h++) {
            // copy "input size" elements to W
            for (size_t b = 0; b < WchunkSz; b++)
                *wBytes++ = *blobBytes++;

            // copy "lstm_hidden_size" elements to R
            for (size_t b = 0; b < RchunkSz; b++)
                *rBytes++ = *blobBytes++;
        }

        m_topology->add(cldnn::data(weightID, wmem));
        m_topology->add(cldnn::data(recurrentID, rmem));

        /* create bias memory primitive */
        auto pBiasBlob = wLayer->_biases;
        if (pBiasBlob != nullptr) {
            cldnn::tensor bTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(4 * lstm_hidden_size, 1));
            cldnn::layout BLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), m_defaultFormat, rTensor);

            auto bmem = cldnn::memory::allocate(*(m_env.engine), BLayout);
            auto btmpPointer = bmem.pointer<char>();

            auto blobBytes = static_cast<const char *>(pBiasBlob->buffer());
            const size_t BchunkSz = lstm_hidden_size * elementSize;
            auto bBytes = btmpPointer.data();

            for (size_t b = 0; b < 4 * BchunkSz; b++)
                *bBytes++ = *blobBytes++;

            m_topology->add(cldnn::data(biasID, bmem));
            hasBias = true;
        }
    }

    std::vector<std::pair<cldnn::primitive_id, cldnn::tensor>> input_ids_offsets;
    std::vector<cldnn::primitive_id> output_ids_offsets;

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";

    cldnn::tensor inputShape;

    if (permute_input) {
        inputShape = { lstm_sequence_len, lstm_batch_size, lstm_input_size, 1 };
    } else {
        inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
    }
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(layer->precision), cldnn::format::bfyx, inputShape);
    m_topology->add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    m_topology->add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    m_topology->add(cldnn::reshape(inHiddenReshapeID+"_1", inputPrimitives[1], hiddenStateShape));
    m_topology->add(cldnn::reshape(inHiddenReshapeID+"_2", inputPrimitives[2], hiddenStateShape));

    for (int i = 0; i < lstm_sequence_len; ++i)
        input_ids_offsets.push_back({ get_string_id(i), {0, i, 0, 0} });

    cldnn::primitive_id inputSplitID = layerName + "_inputSplit";

    if (permute_input) {
        m_topology->add(cldnn::permute(layerName + "_inputSwap", permuteID, { 1, 0, 2, 3 }));
        m_topology->add(cldnn::split(inputSplitID, layerName + "_inputSwap", input_ids_offsets));
    } else {
        m_topology->add(cldnn::split(inputSplitID, permuteID, input_ids_offsets));
    }

    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
    std::string hiddenStr = hasInitialHidden ? inHiddenReshapeID+"_1" : "";
    std::string cellStr = hasInitialCell ? inHiddenReshapeID+"_2" : "";

    for (int i = 0; i < lstm_sequence_len; ++i) {
        std::string lstm_gemm_id = layerName + "_lstm_gemm" + get_string_id(i);
        std::string lstm_elt_id = layerName + "_lstm_elt" + get_string_id(i);
        std::string crop_id = layerName + "_crop" + get_string_id(i);

        int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
        m_topology->add(cldnn::lstm_gemm(lstm_gemm_id, inputSplitID + ":" + get_string_id(seqIdx),
                                            weightID, recurrentID,
                                            hasBias ? biasID : "",
                                            hiddenStr));
        m_topology->add(cldnn::lstm_elt(lstm_elt_id, lstm_gemm_id,
                                            cellStr, 0, 0, {}, {},
                                            cldnn_lstm_offset_order_fizo));

        hiddenStr = crop_id + ":hidden";
        cellStr = crop_id + ":cell";
        m_topology->add(cldnn::crop(hiddenStr, lstm_elt_id, hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }));
        output_ids_offsets.push_back(hiddenStr);

        if (i < lstm_sequence_len - 1) {
            m_topology->add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
        } else {
            // last hidden state crop (output 2)
            if (layer->outData.size() > 1) {
                cldnn::primitive_id outputHiddenID = layer->type + ":" + layer->outData[1]->name;
                m_env.primitiveIDs[hiddenStr] = hiddenStr;
                m_env.primitiveIDs[outputHiddenID] = hiddenStr;
            }

            // last cell state crop (output 3)
            if (layer->outData.size() > 2) {
                m_topology->add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
                cldnn::primitive_id outputCellID = layer->type + ":" + layer->outData[2]->name;
                m_env.primitiveIDs[cellStr] = cellStr;
                m_env.primitiveIDs[outputCellID] = cellStr;
            }
        }
    }

    if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());

    if (permute_input) {
        m_topology->add(cldnn::concatenation(layerName + "_outputConcat", output_ids_offsets, cldnn::concatenation::along_f));
        m_topology->add(cldnn::permute(layerName, layerName + "_outputConcat", { 1, 0, 2, 3 }));
    } else {
        m_topology->add(cldnn::concatenation(layerName, output_ids_offsets, cldnn::concatenation::along_f));
    }

    m_env.primitiveIDs[layerName] = layerName;
    m_env.primitiveIDs[layer->type + ":" + layer->outData[0]->name] = layerName;
    m_env.profilingIDs.push_back(layerName);
}

void CLDNNGraph::AddConstantBlobInput(InferenceEngine::CNNLayerPtr &layer) {
    auto constBlob = layer->blobs.begin()->second;
    auto constDims = layer->outData[0]->dims;

    cldnn::tensor constTensor;
    switch (constDims.size()) {
    case 4: constTensor = cldnn::tensor(TensorValue(constDims[3]), TensorValue(constDims[2]),
            TensorValue(constDims[0]), TensorValue(constDims[1]));
            break;
    case 3: constTensor = cldnn::tensor(TensorValue(constDims[2]), TensorValue(constDims[1]),
            1, TensorValue(constDims[0]));
            break;
    case 2: constTensor = cldnn::tensor(TensorValue(constDims[1]), TensorValue(constDims[0]), 1, 1);
            break;
    case 1: constTensor = cldnn::tensor(TensorValue(constDims[0]), 1, 1, 1);
            break;
        default: THROW_CLDNN_EXCEPTION("Invalid constant blob dimensions");
    }

    cldnn::layout constLayout = cldnn::layout(
        DataTypeFromPrecision(layer->blobs.begin()->second->precision()),
        m_defaultFormat,
        constTensor);

    size_t bytes = constLayout.bytes_count();
    cldnn::primitive_id constPrimID = layer_type_name_ID(layer);

    CreatePrimitiveFromBlob(constPrimID, constBlob, constLayout);
    m_env.primitiveIDs[constPrimID] = constPrimID;
}

void CLDNNGraph::CreateConvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer *> (layer.get());

    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(layer, weightPrimID, biasPrimID);

    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                         cldnn::spatial(convLayer->_stride[X_AXIS], convLayer->_stride[Y_AXIS]));
    auto allPad = getPaddings(*convLayer);
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
                                          cldnn::spatial(-allPad.begin[X_AXIS], -allPad.begin[Y_AXIS]));
    cldnn::tensor dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                           cldnn::spatial(convLayer->_dilation[X_AXIS], convLayer->_dilation[Y_AXIS]));

    std::string convLayerName = layer_type_name_ID(layer);
    if (convLayer->_group >= 16) {
        auto convPrim = cldnn::convolution(convLayerName,
                                           inputPrimitives[0],
                                           weightPrimID,
                                           biasPrimID,
                                           convLayer->_group,
                                           stride,
                                           padding,
                                           dilation,
                                           false,
                                           0.0,
                                           CldnnTensorFromIEDims(convLayer->outData[0]->dims));
        m_topology->add(convPrim);
    } else {
        auto convPrim = cldnn::convolution(convLayerName,
                                           inputPrimitives[0],
                                           weightPrimID,
                                           biasPrimID,
                                           stride,
                                           padding,
                                           dilation,
                                           false,
                                           0.0f,
                                           CldnnTensorFromIEDims(convLayer->outData[0]->dims));
        m_topology->add(convPrim);
    }
    m_env.primitiveIDs[convLayerName] = convLayerName;
    m_env.profilingIDs.push_back(convLayerName);
}

void CLDNNGraph::CreateGatherPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto gatherLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    int axis = gatherLayer->GetParamAsInt("axis", 0);

    // Be careful, TensorFlow consist negative axis interpretation bug. Here: -3 = b, -2 = f, -1 = y, but must be -3 = f, -2 = y, -1 = x
    auto cldnnAxisFromIE = [](int axis) {
        switch (axis) {
            case 0: return cldnn::gather::gather_axis::along_b;
            case 1: return cldnn::gather::gather_axis::along_f;
            case 2: return cldnn::gather::gather_axis::along_y;
            case 3: return cldnn::gather::gather_axis::along_x;
            case -1: return cldnn::gather::gather_axis::along_y;
            case -2: return cldnn::gather::gather_axis::along_f;
            case -3: return cldnn::gather::gather_axis::along_b;
            default: THROW_CLDNN_EXCEPTION("Unsupported gather axis: " << axis);
        }
    };

    std::string gatherLayerName = layer_type_name_ID(layer);
    auto gatherPrim = cldnn::gather(
            gatherLayerName,
            inputPrimitives[0],
            inputPrimitives[1],
            cldnnAxisFromIE(axis),
            CldnnTensorFromIEDims(gatherLayer->outData[0]->dims));

    m_env.primitiveIDs[gatherLayerName] = gatherLayerName;
    m_topology->add(gatherPrim);
    m_env.profilingIDs.push_back(gatherLayerName);
}

void CLDNNGraph::CreateDepthToSpacePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto depthToSpace = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    size_t blockSize = depthToSpace->GetParamAsInt("block_size", 2);

    if (depthToSpace->input().get()->dims.size() != 4)
        THROW_CLDNN_EXCEPTION("Unsupported size of tensor " << depthToSpace->input().get()->dims.size());

    size_t blockSizeSquare = blockSize * blockSize;

    if (depthToSpace->input().get()->dims[2] % blockSizeSquare != 0)
        THROW_CLDNN_EXCEPTION("The depth of the input tensor must be divisible by squared block size = " << blockSizeSquare);

    std::string depthToSpaceName = layer_type_name_ID(layer);
    auto depthToSpacePrim = cldnn::depth_to_space(
            depthToSpaceName,
            inputPrimitives[0],
            blockSize);

    m_env.primitiveIDs[depthToSpaceName] = depthToSpaceName;
    m_topology->add(depthToSpacePrim);
    m_env.profilingIDs.push_back(depthToSpaceName);
}

void CLDNNGraph::CreateShuffleChannelsPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto shuffleChannels = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    const int32_t numberOfDims = shuffleChannels->input()->getDims().size();

    int32_t group = shuffleChannels->GetParamAsInt("group", 1);
    int32_t axis = shuffleChannels->GetParamAsInt("axis", 1);

    if (axis < 0)
        axis += numberOfDims;

    if (axis < 0 || axis >= numberOfDims)
        THROW_CLDNN_EXCEPTION("Incorrect axis value! Actual axis is" + std::to_string(group));

    if (group < 1)
        THROW_CLDNN_EXCEPTION("Invalid group size value (should equal at least one). Actual block size is" +
                                       std::to_string(group));

    if (shuffleChannels->input().get()->getDims()[axis] % group != 0)
        THROW_CLDNN_EXCEPTION("Group parameter must evenly divide the channel dimension. Actual group size is " +
                                       std::to_string(axis));

    std::string shuffleChannelsName = layer_type_name_ID(layer);
    auto shuffleChannelsPrim = cldnn::shuffle_channels(
            shuffleChannelsName,
            inputPrimitives[0],
            group,
            axis);

    m_env.primitiveIDs[shuffleChannelsName] = shuffleChannelsName;
    m_topology->add(shuffleChannelsPrim);
    m_env.profilingIDs.push_back(shuffleChannelsName);
}

void CLDNNGraph::CreateStridedSlicePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto stridedSliceLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    auto tmp = stridedSliceLayer->GetParamAsUInts("end_mask");
    std::vector<uint8_t> end_mask(tmp.begin(), tmp.end());
    tmp = stridedSliceLayer->GetParamAsUInts("begin_mask");
    std::vector<uint8_t> begin_mask(tmp.begin(), tmp.end());
    tmp = stridedSliceLayer->GetParamAsUInts("new_axis_mask");
    std::vector<uint8_t> new_axis_mask(tmp.begin(), tmp.end());
    tmp = stridedSliceLayer->GetParamAsUInts("shrink_axis_mask");
    std::vector<uint8_t> shrink_axis_mask(tmp.begin(), tmp.end());

    std::string stridedSliceLayerName = layer_type_name_ID(layer);
    auto stridedSlicePrim = cldnn::strided_slice(
            stridedSliceLayerName,
            inputPrimitives[0], inputPrimitives[1], inputPrimitives[2], inputPrimitives[3],
            begin_mask, end_mask, new_axis_mask, shrink_axis_mask);

    m_env.primitiveIDs[stridedSliceLayerName] = stridedSliceLayerName;
    m_topology->add(stridedSlicePrim);
    m_env.profilingIDs.push_back(stridedSliceLayerName);
}

void CLDNNGraph::CreateReverseSequencePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto reverseSequence = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    const int32_t numberOfDims = reverseSequence->input()->getDims().size();

    const auto input = reverseSequence->insData[0].lock()->getDims();
    const auto sequence_lengths = reverseSequence->insData[1].lock()->getDims();

    int32_t batch_axis = reverseSequence->GetParamAsInt("batch_axis", 0);
    int32_t seq_axis = reverseSequence->GetParamAsInt("seq_axis", 1);

    if (batch_axis < 0)
        batch_axis += input.size();

    if (seq_axis < 0)
        seq_axis += input.size();

    if (batch_axis == seq_axis)
        THROW_CLDNN_EXCEPTION("Batch axis and sequence axis should not be equal\n");

    if (seq_axis < 0 || seq_axis >= input.size())
        THROW_CLDNN_EXCEPTION("Incorrect Sequence axis value! Actual axis is " + std::to_string(seq_axis));

    if (batch_axis < 0 || batch_axis >= input.size())
        THROW_CLDNN_EXCEPTION("Incorrect Sequence axis value! Actual axis is " + std::to_string(batch_axis));

    if (sequence_lengths[0] != input[batch_axis])
        THROW_CLDNN_EXCEPTION("Sequence lengths must be a vector of length " + std::to_string(input[batch_axis])
                            + "! Actual axis is " + std::to_string(sequence_lengths[0]));

    std::string reverseSequenceLayerName = layer_type_name_ID(layer);
    auto reverseSequencePrim = cldnn::reverse_sequence(
            reverseSequenceLayerName,
            inputPrimitives[0],
            inputPrimitives[1],
            seq_axis,
            batch_axis);

    m_env.primitiveIDs[reverseSequenceLayerName] = reverseSequenceLayerName;
    m_topology->add(reverseSequencePrim);
    m_env.profilingIDs.push_back(reverseSequence->name);
}

bool CLDNNGraph::IsValidSplitConvMerge(const InferenceEngine::SplitLayer *splitLayer) const {
    if (splitLayer->outData.size() != 2) return false;  // split into 2

    for (auto out : splitLayer->outData) {
        if (out->getInputTo().size() != 1) {
            return false;
        }
    }

    auto convLayer1 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[0]).get());
    auto convLayer2 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[1]).get());
    if (!convLayer1 || !convLayer2) {   // outputs aren't convolutions
        return false;
    }
    auto allPad1 = getPaddings(*convLayer1);
    auto allPad2 = getPaddings(*convLayer2);
    if (convLayer1->precision != convLayer2->precision                       // wrong precision
        || convLayer1->_fusedWith || convLayer2->_fusedWith                     // convolutions are fused
        || convLayer1->outData.size() != 1 || convLayer2->outData.size() != 1   // more than 1 output for convolutions
        || allPad1.begin[X_AXIS] != allPad2.begin[X_AXIS]                     // different padding
        || allPad1.begin[Y_AXIS] != allPad2.begin[Y_AXIS]                     // different padding
        || convLayer1->_stride[X_AXIS] != convLayer2->_stride[X_AXIS]                       // different strides
        || convLayer1->_stride[Y_AXIS] != convLayer2->_stride[Y_AXIS]                       // different strides
        || convLayer1->_dilation[X_AXIS] != convLayer2->_dilation[X_AXIS]                   // different dilation
        || convLayer1->_dilation[Y_AXIS] != convLayer2->_dilation[Y_AXIS]                   // different dilation
        || (GetNextSingleLayer(GetNextSingleLayer(splitLayer->outData[0]))      // no merge after convolutions
            != GetNextSingleLayer(GetNextSingleLayer(splitLayer->outData[1])))
        || (p_currentOutputs->find(convLayer1->name) != p_currentOutputs->end())
        || (p_currentOutputs->find(convLayer2->name) != p_currentOutputs->end())) {
        return false;
    }
    auto concatLayer =
        dynamic_cast<InferenceEngine::ConcatLayer *> (
                GetNextSingleLayer(GetNextSingleLayer(splitLayer->outData[0])).get());
    if (!concatLayer ||                         // not a merge layer
        concatLayer->_axis != 1 ||              // merge on unsupported axis
        concatLayer->outData.size() != 1) {     // too many outputs
        return false;
    }
    if (m_config.customLayers.find(convLayer1->type) != m_config.customLayers.end() ||
        m_config.customLayers.find(concatLayer->type) != m_config.customLayers.end()) {
        return false;  // convolution or concat were overwritten by a custom layer
    }

    return true;
}

void CLDNNGraph::AddInputPrimitive(InferenceEngine::InputInfo::Ptr inputInfo, Precision inputPrecision) {
    // first create and add the input layout
    auto inputDims = inputInfo->getDims();
    InferenceEngine::Layout l = inputInfo->getTensorDesc().getLayout();
    auto consumers = inputInfo->getInputData()->getInputTo();
    bool single_consumer = consumers.size() == 1;
    CLDNNGraph::LayerType consumerType = LayerTypeFromStr(consumers.begin()->second->type);

    cldnn::tensor dataTensor;
    cldnn::tensor::value_type batch = (m_env.m_max_batch <= 1)
                                        ? (inputDims.size() == 4 ? TensorValue(inputDims[3]) : 1)
                                        : TensorValue(m_curBatch);
    switch (inputDims.size()) {
        case 4:
            if (InferenceEngine::Layout::NCHW == l || InferenceEngine::Layout::CHW == l) {
                dataTensor = cldnn::tensor(batch,
                    TensorValue(inputDims[2]), TensorValue(inputDims[0]),
                    TensorValue(inputDims[1]));
            } else if (InferenceEngine::Layout::NHWC == l) {
                dataTensor = cldnn::tensor(batch,
                    TensorValue(inputDims[2]), TensorValue(inputDims[0]),
                    TensorValue(inputDims[1]));
            } else {
                THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(l) << ") in 4D input " + inputInfo->name());
            }
            break;
        case 3:
            if (InferenceEngine::Layout::CHW == l) {
                dataTensor = cldnn::tensor(TensorValue(inputDims[2]), TensorValue(inputDims[1]), 1, TensorValue(inputDims[0]));
            } else {
                THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(l) << ") in 3D input " + inputInfo->name());
            }
            break;
        case 2:
            if (InferenceEngine::Layout::NCHW == l) {
                dataTensor = cldnn::tensor(1, 1, TensorValue(inputDims[1]), TensorValue(inputDims[0]));
            } else if (InferenceEngine::NC == l) {
                dataTensor = cldnn::tensor(TensorValue(inputDims[1]), TensorValue(inputDims[0]), 1, 1);
            } else {
                THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(l) << ") in 2D input " + inputInfo->name());
            }
            break;
        case 1:
            dataTensor = cldnn::tensor(TensorValue(inputDims[0]), 1, 1, 1);
            break;
        default: THROW_CLDNN_EXCEPTION("Invalid data dimensions");
    }

    cldnn::layout inputLayout(DataTypeFromPrecision(inputInfo->getInputPrecision()),
        FormatFromLayout(l),
        dataTensor);

    // save the input dims
    m_env.inputLayouts.insert({ inputInfo->name(), inputLayout });

    auto inputName = "Input:" + inputInfo->name();
    m_topology->add(cldnn::input_layout(inputName, inputLayout));

    // create preprocess primitive for this input
    auto preProcess = inputInfo->getPreProcess();

    size_t meanChannels = preProcess.getNumberOfChannels();
    inputLayout.format = m_defaultFormat;
    inputLayout.size = inputLayout.size.transform(m_defaultFormat, 1);
    inputLayout.data_type = DataTypeFromPrecision(inputPrecision);
    auto preprocessPrimID = inputName + m_preProcessTag;

    if ((meanChannels > 0) &&
        (meanChannels != inputLayout.size.feature[0])) {
        THROW_CLDNN_EXCEPTION("Mismatched mean values channels in input " + inputName);
    }

    switch (preProcess.getMeanVariant()) {
    case NONE:
    case MEAN_VALUE: {
        std::vector<float> meanValues;
        if (meanChannels > 0) {
            for (size_t c = 0; c < meanChannels; c++) {
                if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                    THROW_CLDNN_EXCEPTION("not supporting stdScale yet in input " + inputName);
                meanValues.push_back(preProcess[c]->meanValue);
            }
        }
        m_topology->add(cldnn::reorder(preprocessPrimID, inputName, inputLayout, meanValues));
        m_env.profilingIDs.push_back(preprocessPrimID);
        InitProfileInfo(preprocessPrimID, "Reorder");
    }
    break;

    case MEAN_IMAGE: {
        IE_ASSERT(meanChannels);
        // first merge all mean values to a single blob
        // todo make sure mean blob precision is the same as the input precision
        auto meanDims = inputInfo->getDims();
        // overwrite batches with 1
        switch (meanDims.size()) {
        case 4: meanDims[3] = 1;
            break;
        default:
            THROW_CLDNN_EXCEPTION("Missing batch dimensions in input image");
        }
        InferenceEngine::TBlob<float> meanBlob(Precision(Precision::FP32), TensorDesc::getLayoutByDims(meanDims), meanDims);
        meanBlob.allocate();
        auto meanBlobData = meanBlob.data();
        for (size_t c = 0; c < meanChannels; c++) {
            if (fabs(preProcess[c]->stdScale - 1.0f) > 1e-10)
                THROW_CLDNN_EXCEPTION("not supporting stdScale yet in input " + inputName);
            auto channelMeanBlob = std::dynamic_pointer_cast<TBlob<float>>(preProcess[c]->meanData);
            auto channelSize = channelMeanBlob->size();
            auto channelBlobData = channelMeanBlob->data();
            for (size_t i = 0; i < channelSize; i++) {
                meanBlobData[(c * channelSize) + i] = channelBlobData[i];
            }
        }
        // then create a data primitive for the mean values
        auto meanBlobPtr = std::make_shared<InferenceEngine::TBlob<float>>(meanBlob);

        // mean values will use external format (sub in the input format before convert to new format)
        cldnn::tensor meanBlobTensor(inputLayout.size);
        meanBlobTensor.batch[0] = 1;  // mean values have no batches
        cldnn::layout meanBlobLayout(cldnn::data_types::f32, m_defaultFormat, meanBlobTensor);
        CreatePrimitiveFromBlob(
            inputName + m_meanValuesTag,
            meanBlobPtr,
            meanBlobLayout);
        m_topology->add(cldnn::reorder(preprocessPrimID,
            inputName,
            inputLayout,
            inputName + m_meanValuesTag));
        m_env.profilingIDs.push_back(preprocessPrimID);
        InitProfileInfo(preprocessPrimID, "Reorder");
    }
    break;

    default: THROW_CLDNN_EXCEPTION("Invalid mean variant in input " + inputName);
        break;
    }
    m_env.primitiveIDs[inputName] = preprocessPrimID;
    m_env.primitiveIDs[preprocessPrimID] = preprocessPrimID;
}

std::vector<cldnn::primitive_id> CLDNNGraph::GetPrevLayersPrimitives(const InferenceEngine::CNNLayerPtr layer) const {
    if (layer == nullptr) {
        return {};
    }
    std::vector<cldnn::primitive_id> inputPrimitives;
    for (auto inputData : layer->insData) {
        auto prevData = inputData.lock();
        if (prevData == nullptr) {
            THROW_CLDNN_EXCEPTION("Nonexistent input for layer: " << layer->name);
        }
        auto prevCreator = prevData->creatorLayer.lock();
        std::string prevName;

        if (prevCreator) {
            prevName = prevCreator->type + ":";
            if (prevCreator->outData.size() > 1)
                prevName += prevData->name;
            else
                prevName += prevCreator->name;
        } else {
            prevName = prevData->name;
        }
        inputPrimitives.push_back(m_env.primitiveIDs.at(prevName));
    }
    return inputPrimitives;
}

void CLDNNGraph::AddOutputPrimitive(std::string outputName, const InferenceEngine::DataPtr outputData, Precision outputPrecision) {
    // TODO: add precision check once there's an outputInfo object
    if (outputData->layout != InferenceEngine::NCHW &&
        outputData->layout != InferenceEngine::NHWC &&
        outputData->layout != InferenceEngine::CHW &&
        outputData->layout != InferenceEngine::NC) {
        THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(outputData->layout) << ") in output: " << outputName);
    }

    auto outputCreator = outputData->getCreatorLayer().lock();
    std::string outLayerName = outputCreator->type + ":";

    if (outputCreator->outData.size() > 1)
        outLayerName += outputName;
    else
        outLayerName += outputCreator->name;

    auto outputReorderID = outputName + m_postProcessTag;
    Precision precision = outputPrecision == Precision::UNSPECIFIED ? outputData->getPrecision() : outputPrecision;

    // Find correct output ID. Start with name stored in IR.
    std::string outputID = outLayerName;
    std::string finalID = m_env.primitiveIDs.at(outLayerName);

    while (outputID != finalID) {
        auto prim = m_env.primitiveIDs.find(finalID);

        if (prim == m_env.primitiveIDs.end()) {
            THROW_IE_EXCEPTION << "Unknown output primitive id " << outputID;
        }
        outputID = finalID;
        finalID = prim->second;
    }

    m_topology->add(cldnn::reorder(outputReorderID, outputID,
        FormatFromLayout(outputData->getLayout()),
        DataTypeFromPrecision(precision)));
    m_env.primitiveIDs[outputName] = outputReorderID;
    m_env.profilingIDs.push_back(outputReorderID);
    InitProfileInfo(outputReorderID, "Reorder");
    m_env.outputDims[outputName] = outputData->dims;
    m_env.prevPrimitiveIDs[outputReorderID] = {outputName};
}

void CLDNNGraph::AddSingleValuePrimitive(cldnn::primitive_id valPrimID, cldnn::data_types dataType, float value) {
    cldnn::layout primLayout(dataType, m_defaultFormat, { 1, 1, 1, 1 });
    auto primMem = cldnn::memory::allocate(*(m_env.engine), primLayout);
    switch (dataType) {
    case cldnn::data_types::f32:
    {
        auto tmpPointer = primMem.pointer<float>();  // implicitly maps buffer - unmap in destructor
        tmpPointer[0] = value;
    }
        break;
    case cldnn::data_types::f16:
    {
        auto tmpPointer = primMem.pointer<uint16_t>();  // implicitly maps buffer - unmap in destructor
        cldnn_status status = CLDNN_SUCCESS;
        tmpPointer[0] = cldnn_float_to_half(value, &status);
        if (status != CLDNN_SUCCESS) {
            THROW_CLDNN_EXCEPTION("Error converting value to fp16.");
        }
    }
        break;
    default:
        THROW_CLDNN_EXCEPTION("Unhandled data type (precision)");
    }

    m_topology->add(cldnn::data(valPrimID, primMem));
}

cldnn::data_types CLDNNGraph::DataTypeFromPrecision(InferenceEngine::Precision p) {
    switch (p) {
    case Precision::I16:
    case Precision::FP32:
        return cldnn::data_types::f32;
    case Precision::FP16:
        return cldnn::data_types::f16;
    case Precision::U8:
        return cldnn::data_types::u8;
    case Precision::I32:
        return cldnn::data_types::i32;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << p.name() << " precision";
        break;
    }
}

cldnn::format CLDNNGraph::FormatFromLayout(InferenceEngine::Layout l) {
    switch (l) {
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
        return cldnn::format::bfyx;
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::byxf;
    default:
        THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "The plugin does not support " << l << " layout";
        break;
    }
}

cldnn::upsampling_sample_type CLDNNGraph::UpsamplingTypeFromString(const std::string& str) {
    static const caseless_map<std::string, cldnn::upsampling_sample_type> UpsamplingTypeNameToType = {
        { "Bilinear" , cldnn::upsampling_sample_type::bilinear },
        { "Nearest" , cldnn::upsampling_sample_type::nearest },
    };
    auto it = UpsamplingTypeNameToType.find(str);
    if (it != UpsamplingTypeNameToType.end())
        return it->second;
    else
        THROW_CLDNN_EXCEPTION("Unknown Upsampling type: " << str);
}

cldnn::softmax::dimension_t CLDNNGraph::SoftmaxDimensionFromIEAxis(const InferenceEngine::SoftMaxLayer* softmaxLayer, bool isPrevFC) {
    // WA for default softmax dimension in cldnn for fyx
    // todo: remove this once clDNN changes FC output to BF instead of BX
    auto dims = softmaxLayer->outData[0]->dims;
    unsigned non1Dims = 0;
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] > 1) {
            non1Dims++;
        }
    }
    if (non1Dims == 1 || isPrevFC) {
        return cldnn::softmax::normalize_fyx;
    }
    // end of WA

    switch (softmaxLayer->axis) {
    case 1: return cldnn::softmax::normalize_f;
    case 2: return cldnn::softmax::normalize_y;
    case 3: return cldnn::softmax::normalize_x;
    default: THROW_CLDNN_EXCEPTION("Invalid softmax axis " << softmaxLayer->axis);
    }
    return cldnn::softmax::normalize_fyx;
}

cldnn::prior_box_code_type CLDNNGraph::PriorBoxCodeFromString(const std::string& str) {
    static const std::map<std::string, cldnn::prior_box_code_type> CodeNameToType = {
        { "caffe.PriorBoxParameter.CORNER" , cldnn::prior_box_code_type::corner },
        { "caffe.PriorBoxParameter.CENTER_SIZE" , cldnn::prior_box_code_type::center_size },
        { "caffe.PriorBoxParameter.CORNER_SIZE" , cldnn::prior_box_code_type::corner_size },
    };
    auto it = CodeNameToType.find(str);
    if (it != CodeNameToType.end()) {
        return it->second;
    } else {
        THROW_CLDNN_EXCEPTION("Unknown Prior-Box code type: " + str);
        return cldnn::prior_box_code_type::corner;
    }
}

void CLDNNGraph::CreateGenericLayerBlobPrimitives(const InferenceEngine::GenericLayer* layer) {
    IE_ASSERT(layer);
    for (auto& blob : layer->blobs) {
        if (blob.second->dims().size() != 1) {
            THROW_CLDNN_EXCEPTION("Unhandled blob dim in layer " + layer->name);
        }
        CreatePrimitiveFromBlob(
            layer->type + ":" + layer->name + "_" + blob.first + m_weightsTag,
            blob.second,
            cldnn::layout(
                DataTypeFromPrecision(blob.second->precision()),
                m_defaultFormat, cldnn::spatial(TensorValue(blob.second->dims()[0]))));
    }
}

void CLDNNGraph::ValidateGenericLayerBlobs(const InferenceEngine::GenericLayer* layer, const std::vector<std::string>& blobNames) {
    IE_ASSERT(layer);
    for (auto& name : blobNames) {
        if (layer->blobs.find(name) == layer->blobs.end()) {
            THROW_CLDNN_EXCEPTION("Missing blob " + name + " in layer " + layer->name);
        }
    }
}

cldnn::tensor CLDNNGraph::CldnnTensorFromIEDims(const InferenceEngine::SizeVector& dims) {
    auto numDims = dims.size();
    std::vector<cldnn::tensor::value_type> outputTensor({ 1, 1, 1, 1 });
    for (size_t i = 0; i < numDims; i++) {
        outputTensor[i] = TensorValue(dims[numDims - i - 1]);
    }
    // swap x,y for cldnn tensor taking bfxy instead of bfyx
    auto tmp = outputTensor[2];
    outputTensor[2] = outputTensor[3];
    outputTensor[3] = tmp;

    return outputTensor;
}

InferRequestInternal::Ptr
CLDNNGraph::CreateInferRequestImpl(InputsDataMap networkInputs, OutputsDataMap networkOutputs) {
    if (m_env.network == nullptr) {
        THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
    }
    return std::make_shared<CLDNNInferRequest>(m_env, m_config.useProfiling, networkInputs, networkOutputs);
}

void CLDNNGraph::InitProfileInfo(const std::string& layerName,
                                 const std::string& layerType,
                                 bool isCPU,
                                 InferenceEngine::InferenceEngineProfileInfo::LayerStatus status) {
    m_env.perfMap[layerType + ":" + layerName].first = layerName;
    auto& perfEntry = m_env.perfMap[layerType + ":" + layerName].second;
    perfEntry.layerType = layerType;
    perfEntry.status = status;
    perfEntry.cpu_uSec = perfEntry.realTime_uSec = 0;
    perfEntry.isCPU = isCPU;
    perfEntry.status = status;
}

};  // namespace CLDNNPlugin
