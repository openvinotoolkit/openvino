// Copyright (C) 2018 Intel Corporation
//
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
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cldnn_graph.h"
#include "simple_math.h"
#include <description_buffer.hpp>
#include <cldnn/cldnn_config.hpp>
#include <graph_tools.hpp>
#include "cldnn_infer_request.h"
#include <cpp_interfaces/ie_executor_manager.hpp>
#include <caseless.hpp>
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
    if (layer->insData.size() < 2) {
        THROW_CLDNN_EXCEPTION("Invalid number of inputs for layer: " << layer->name << ". Eltwise layer should take at least 2 inputs");
    }
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
    m_networkPrecision(cldnn::data_types::f32),
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
            m_env.constBlobs.clear();
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
    m_networkPrecision = DataTypeFromPrecision(network.getPrecision());
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

    // add input data from all constant blobs
    for (auto& cblob : m_env.constBlobs) {
        m_env.network->set_input_data(cblob.first, cblob.second);
    }
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

    std::list<InferenceEngine::CNNLayerPtr> layersToHandle;
    for (auto input : networkInputs) {
        IE_ASSERT(input.first.compare(input.second->name()) == 0);
        AddInputPrimitive(input.second);

        // collect next layers to process
        for (auto l : input.second->getInputData()->getInputTo()) {
            layersToHandle.push_back(l.second);
        }
    }

    auto allInputs = CNNNetGetAllInputLayers(network);
    for (auto input : allInputs) {
        if (LayerTypeFromStr(input->type) == ConstantBlob) {
            AddConstantBlobInput(input);

            // collect next layers to process
            for (auto nl : GetNextLayers(input)) {
                layersToHandle.push_back(nl);
            }
        }
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
        auto layerName = currLayer->name;

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
        IE_ASSERT(_networkPrecision == currLayer->precision);
        CreateSingleLayerPrimitive(currLayer);  // currLayer will be advanced if layer was skipped or merged
        m_env.prevPrimitiveIDs[currLayer->name] = GetPrevLayersPrimitives(currLayer);

        for (auto nl : GetNextLayers(currLayer)) {
            layersToHandle.push_back(nl);
        }
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
        default: THROW_CLDNN_EXCEPTION("Unsupported eltwise operation: " << op);
            break;
    }

    return cldnn::eltwise_mode::max;  // shouldn't get here
}

cldnn::concatenation::concatenation_axis CLDNNGraph::ConcatAxisFromIEAxis(unsigned axis) {
    switch (axis) {
    case 0:
        THROW_CLDNN_EXCEPTION("Unsupported concatenation axis: " << axis);  // Currently unsupported (although existing in the API)
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
        groupSize = convLayer->_group;
        if ((inFeatures % groupSize) || (convLayer->_out_depth % groupSize)) {
            THROW_CLDNN_EXCEPTION("Invalid group size in layer " << convLayer->name);
        }
        weightDimsVec = {
            TensorValue(convLayer->_out_depth / groupSize),
            TensorValue(inFeatures / groupSize),
            TensorValue(convLayer->_kernel_x),
            TensorValue(convLayer->_kernel_y)
        };
        outFeatures = convLayer->_out_depth;
        pWeightsBlob = convLayer->_weights;
        pBiasBlob = convLayer->_biases;
    }
        break;
    case Deconvolution: {
        auto deconvLayer = dynamic_cast<InferenceEngine::DeconvolutionLayer *> (layer.get());
        groupSize = deconvLayer->_group;
        if ((inFeatures % groupSize) || (deconvLayer->_out_depth % groupSize)) {
            THROW_CLDNN_EXCEPTION("Invalid group size in layer " << deconvLayer->name);
        }
        weightDimsVec = {
            TensorValue(deconvLayer->_out_depth / groupSize),
            TensorValue(inFeatures / groupSize),
            TensorValue(deconvLayer->_kernel_x),
            TensorValue(deconvLayer->_kernel_y)
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
        m_networkPrecision,
        m_defaultFormat,
        cldnn::tensor(weightDimsVec));
    size_t bytesPerGroup = weightsLayout.bytes_count();

    for (unsigned g = 0; g < groupSize; g++) {
        cldnn::primitive_id weightID = layer->name + m_weightsTag + std::to_string(g);
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
            m_networkPrecision,
            m_defaultFormat,
            cldnn::spatial(TensorValue(outFeatures / groupSize)));
        size_t bytesPerGroup = biasesLayout.bytes_count();
        for (unsigned g = 0; g < groupSize; g++) {
            cldnn::primitive_id biasID = layer->name + m_biasesTag + std::to_string(g);
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
        m_networkPrecision,
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
    InitProfileInfo(layer->name, layer->type, "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);

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
        case RegionYolo: CreateYOLO2RegionPrimitive(layer);
            break;
        case ReorgYolo: CreateYOLO2ReorgPrimitive(layer);
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

    cldnn::layout blobLayout(m_networkPrecision, m_defaultFormat, weightTensor);
    CreatePrimitiveFromBlob(scalePrimID, scaleShiftLayer->_weights, blobLayout);
    if (scaleShiftLayer->_biases != nullptr) {
        if (scaleShiftLayer->_biases->dims() != dims) {
            THROW_CLDNN_EXCEPTION("Invalid bias blob dimensions in layer " << layer->name);
        }
        CreatePrimitiveFromBlob(biasPrimID, scaleShiftLayer->_biases, blobLayout);
    } else {
        biasPrimID = "";  // 0-bias
    }

    auto scaleShiftPrim = cldnn::scale(
        scaleShiftLayer->name,
        inputPrimitives[0],
        scalePrimID,
        biasPrimID);

    m_env.primitiveIDs[scaleShiftLayer->name] = scaleShiftLayer->name;
    m_topology->add(scaleShiftPrim);
    m_env.profilingIDs.insert(scaleShiftLayer->name);
}

void CLDNNGraph::CreateProposalPrimitive(InferenceEngine::CNNLayerPtr & layer) {
    ValidateLayer(layer, 3);
    IE_ASSERT(layer->insData[0].lock()->dims[3] == 1);  // only handling input batch size 1
    IE_ASSERT(layer->insData[1].lock()->dims[3] == 1);  // only handling input batch size 1
    auto proposalLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    float nms_thresh = proposalLayer->GetParamAsFloat("nms_thresh", 0.7f);
    int min_size = proposalLayer->GetParamAsInt("min_size", 16);
    int feature_stride = proposalLayer->GetParamAsInt("feat_stride", 16);
    int pre_nms_topn = proposalLayer->GetParamAsInt("pre_nms_topn", 6000);
    int post_nms_topn = proposalLayer->GetParamAsInt("post_nms_topn", 300);
    std::vector<float> ratio = proposalLayer->GetParamAsFloats("ratio");
    std::vector<float> scale = proposalLayer->GetParamAsFloats("scale");

    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto proposalPrim = cldnn::proposal(
        proposalLayer->name,
        inputPrimitives[0],  // cls_score
        inputPrimitives[1],  // bbox_pred
        inputPrimitives[2],  // im_info
        0,                   // max_num_proposals is unused
        nms_thresh,
        min_size,
        feature_stride,
        pre_nms_topn,
        post_nms_topn,
        ratio,
        scale);

    m_env.primitiveIDs[proposalLayer->name] = proposalLayer->name;
    m_topology->add(proposalPrim);
    m_env.profilingIDs.insert(proposalLayer->name);
}

void CLDNNGraph::CreatePReLUPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto preluLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    auto inDataPtr = preluLayer->insData[0].lock();
    if (!inDataPtr) {
        THROW_CLDNN_EXCEPTION("Data inserted into PreLu " << preluLayer->name << " is nullptr");
    }
    auto inputDims = inDataPtr->dims;
    if (inputDims.size() == 2) {
        // WA for FC output as BF instead of BX
        // todo: remove this once FC output is changed in clDNN
        cldnn::primitive_id reshapeID = preluLayer->name + m_workaroundTag;
        m_topology->add(cldnn::reshape(
            reshapeID,
            inputPrimitives[0],
            cldnn::tensor(TensorValue(inputDims[1]), TensorValue(inputDims[0]), 1, 1)));
        m_env.primitiveIDs[inputPrimitives[0]] = reshapeID;
        inputPrimitives[0] = reshapeID;
        m_env.primitiveIDs[reshapeID] = reshapeID;
        m_env.profilingIDs.insert(reshapeID);
    }

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
        m_topology->add(cldnn::activation(preluLayer->name, inputPrimitives[0], activation_relu_negative_slope, { slope, 0.f }));
    } else {
        CreateGenericLayerBlobPrimitives(preluLayer);
        cldnn::primitive_id slopePrimID(preluLayer->name + "_" + blobName + m_weightsTag);
        m_topology->add(cldnn::activation(preluLayer->name, inputPrimitives[0], slopePrimID, activation_relu_negative_slope));
    }

    m_env.primitiveIDs[preluLayer->name] = preluLayer->name;
    m_env.profilingIDs.insert(preluLayer->name);
}

void CLDNNGraph::CreateBatchNormalizationPrimitive(InferenceEngine::CNNLayerPtr & layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto bnLayer = dynamic_cast<InferenceEngine::BatchNormalizationLayer *> (layer.get());
    cldnn::primitive_id weightID = bnLayer->name + "_" + m_scalesTag;
    cldnn::primitive_id biasID = bnLayer->name + "_" + m_biasesTag;

#define _SCALE_BN_OPT
#ifdef _SCALE_BN_OPT
    // Using scale as an optimization (1 mad instead of mad+rsq)
    // create new blobs for scale shift
    CreateScaleWeightsAndBiasesFromBN(bnLayer, weightID, biasID);
    auto scalePrim = cldnn::scale(bnLayer->name, inputPrimitives[0], weightID, biasID);

    m_env.primitiveIDs[bnLayer->name] = bnLayer->name;
    m_topology->add(scalePrim);
    m_env.profilingIDs.insert(bnLayer->name);
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
        m_networkPrecision,
        m_defaultFormat,
        blobTensor);

    // Create variance primitive
    cldnn::primitive_id varianceID = bnLayer->name + "_" + m_weightsTag;
    CreatePrimitiveFromBlob(varianceID, bnLayer->_weights, blobLayout);

    // Create mean primitive
    cldnn::primitive_id meanID = bnLayer->name + "_" + m_biasesTag;
    CreatePrimitiveFromBlob(meanID, bnLayer->_biases, blobLayout);

    auto bnPrim = cldnn::batch_norm(
        bnLayer->name,
        inputPrimitives[0],
        meanID,
        varianceID,
        bnLayer->epsilon);

    m_env.primitiveIDs[bnLayer->name] = bnLayer->name;
    m_topology->add(bnPrim);
    m_env.profilingIDs.insert(bnLayer->name);
}

void CLDNNGraph::CreateFlattenPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto flattenLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    auto flattenPrim = cldnn::reshape(
        flattenLayer->name,
        inputPrimitives[0],
        CldnnTensorFromIEDims(flattenLayer->outData[0]->dims));

    m_env.primitiveIDs[flattenLayer->name] = flattenLayer->name;
    m_topology->add(flattenPrim);
    m_env.profilingIDs.insert(flattenLayer->name);
}

void CLDNNGraph::CreatePermutePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto permuteLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    std::vector<uint16_t> order;
    for (auto& a : permuteLayer->GetParamAsInts("order"))
        order.push_back(static_cast<uint16_t>(a));
    auto outputDims = permuteLayer->outData[0]->dims;

    auto permutePrim = cldnn::permute(
        permuteLayer->name,
        inputPrimitives[0],
        order);

    m_env.primitiveIDs[permuteLayer->name] = permuteLayer->name;
    m_topology->add(permutePrim);
    m_env.profilingIDs.insert(permuteLayer->name);
}

void CLDNNGraph::CreateReshapePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto reshapeLayer = dynamic_cast<InferenceEngine::ReshapeLayer*> (layer.get());
    IE_ASSERT(reshapeLayer->outData.size());

    auto reshapePrim = cldnn::reshape(
        reshapeLayer->name,
        inputPrimitives[0],
        CldnnTensorFromIEDims(reshapeLayer->outData[0]->dims));

    m_env.primitiveIDs[reshapeLayer->name] = reshapeLayer->name;
    m_topology->add(reshapePrim);
    m_env.profilingIDs.insert(reshapeLayer->name);
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

    auto normPrim = cldnn::normalize(
        normLayer->name,
        inputPrimitives[0],
        normLayer->name + "_weights" + m_weightsTag,
        across_spatial,
        eps);

    m_env.primitiveIDs[normLayer->name] = normLayer->name;
    m_topology->add(normPrim);
    m_env.profilingIDs.insert(normLayer->name);
}

void CLDNNGraph::CreateDetectionOutputPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 3);
    auto detectionLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    uint32_t num_classes = detectionLayer->GetParamAsUInt("num_classes", 1);
    bool share_location = detectionLayer->GetParamsAsBool("share_location", true);
    int background_label_id = detectionLayer->GetParamAsInt("background_label_id", 0);
    float nms_threshold = detectionLayer->GetParamAsFloat("nms_threshold", 0.3f);
    int top_k = detectionLayer->GetParamAsInt("top_k", -1);
    float confidence_threshold = detectionLayer->GetParamAsFloat("confidence_threshold", -FLT_MAX);
    float eta = detectionLayer->GetParamAsFloat("eta", 1.0f);
    int keep_top_k = detectionLayer->GetParamAsInt("keep_top_k", -1);
    bool variance_encoded_in_target = detectionLayer->GetParamsAsBool("variance_encoded_in_target", false);
    int input_width = detectionLayer->GetParamAsInt("input_width", -1);
    int input_height = detectionLayer->GetParamAsInt("input_height", -1);
    bool normalized = detectionLayer->GetParamsAsBool("normalized", true);
    std::string code_type = detectionLayer->GetParamAsString("code_type", "caffe.PriorBoxParameter.CORNER");
    bool clip = detectionLayer->GetParamsAsBool("clip", false);
    bool decrease_label_id = detectionLayer->GetParamsAsBool("decrease_label_id", false);
    cldnn::prior_box_code_type cldnnCodeType = PriorBoxCodeFromString(code_type);

    int32_t prior_info_size = normalized != 0 ? 4 : 5;
    int32_t prior_coordinates_offset = normalized != 0 ? 0 : 1;

    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto detectionPrim = cldnn::detection_output(
        detectionLayer->name,
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
        clip);

    m_env.primitiveIDs[detectionLayer->name] = detectionLayer->name;
    m_topology->add(detectionPrim);
    m_env.profilingIDs.insert(detectionLayer->name);
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

    auto priorBoxPrim = cldnn::prior_box(
        priorBoxLayer->name,
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

    m_env.primitiveIDs[priorBoxLayer->name] = priorBoxLayer->name;
    m_topology->add(priorBoxPrim);
    m_env.profilingIDs.insert(priorBoxLayer->name);
}

void CLDNNGraph::CreateDeconvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto deconvLayer = dynamic_cast<InferenceEngine::DeconvolutionLayer *> (layer.get());

    if (deconvLayer->_dilation_x != 1 || deconvLayer->_dilation_y != 1) {
        THROW_CLDNN_EXCEPTION("Unsupported dilation in deconvolution " << layer->name);
    }

    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(layer, weightPrimID, biasPrimID);
    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                         cldnn::spatial(deconvLayer->_stride_x, deconvLayer->_stride_y));
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
                                         cldnn::spatial(-deconvLayer->_padding_x, -deconvLayer->_padding_y));

    auto deconvPrim = cldnn::deconvolution(deconvLayer->name,
        inputPrimitives[0],
        weightPrimID,
        biasPrimID,
        stride,
        padding,
        false,
        0.0f,
        CldnnTensorFromIEDims(deconvLayer->outData[0]->dims));
    m_env.primitiveIDs[deconvLayer->name] = deconvLayer->name;
    m_topology->add(deconvPrim);
    m_env.profilingIDs.insert(deconvLayer->name);
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
    IE_ASSERT(cropLayer->outData[0] && cropLayer->outData[0]->dims.size() == 4);

    std::vector<cldnn::tensor::value_type> offset{ 0, 0, 0, 0 };
    for (size_t i = 0; i < cropLayer->axis.size(); i++) {
        if (cropLayer->axis[i] < 0 || cropLayer->axis[i] > 3) {
            THROW_CLDNN_EXCEPTION("Invalid crop axis: " + std::to_string(cropLayer->axis[i]) + " in layer " + cropLayer->name);
        }
        offset[cropLayer->axis[i]] = cropLayer->offset[i];
    }
    auto outputDims = cropLayer->outData[0]->dims;
    cldnn::tensor refSize(
        TensorValue(outputDims[3]),
        TensorValue(outputDims[2]),
        TensorValue(outputDims[0]),
        TensorValue(outputDims[1]));

    auto cropPrim = cldnn::crop(
        cropLayer->name,
        inputPrimitives[0],
        refSize,
        cldnn::tensor(offset));
    m_env.primitiveIDs[cropLayer->name] = cropLayer->name;
    m_topology->add(cropPrim);
    m_env.profilingIDs.insert(cropLayer->name);
}

void CLDNNGraph::CreateROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);
    auto roiPoolingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // params
    int pooled_width = roiPoolingLayer->GetParamAsInt("pooled_w", 0);
    int pooled_height = roiPoolingLayer->GetParamAsInt("pooled_h", 0);
    float spatial_scale = roiPoolingLayer->GetParamAsFloat("spatial_scale", 1.0f);

    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto roiPoolingPrim = cldnn::roi_pooling(
        roiPoolingLayer->name,
        inputPrimitives[0],  // input data
        inputPrimitives[1],  // input rois
        cldnn::pooling_mode::max,
        pooled_width,
        pooled_height,
        spatial_scale);
    m_env.primitiveIDs[roiPoolingLayer->name] = roiPoolingLayer->name;
    m_topology->add(roiPoolingPrim);
    m_env.profilingIDs.insert(roiPoolingLayer->name);
}

void CLDNNGraph::CreatePSROIPoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 2);
    auto psROIPoolingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // params
    int group_size = psROIPoolingLayer->GetParamAsInt("group_size");
    // todo: assert outputdim*group_size*group_size == input features
    float spatial_scale = psROIPoolingLayer->GetParamAsFloat("spatial_scale");
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto psROIPoolingPrim = cldnn::roi_pooling(
        psROIPoolingLayer->name,
        inputPrimitives[0],  // input data
        inputPrimitives[1],  // input rois
        cldnn::pooling_mode::average,
        group_size,
        group_size,
        spatial_scale,
        group_size);
    m_env.primitiveIDs[psROIPoolingLayer->name] = psROIPoolingLayer->name;
    m_topology->add(psROIPoolingPrim);
    m_env.profilingIDs.insert(psROIPoolingLayer->name);
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
            m_networkPrecision,
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
                    m_env.profilingIDs.insert(reorderPrimName);
                    InitProfileInfo(reorderPrimName, "Reorder", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
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
        genericLayer->name,
        reorderedInputs,
        { layerTitle, defineTitle, layerDefines, customLayer->KernelSource() },
        customLayer->KernelEntry(),
        kernelParameters,
        customLayer->CompilerOptions(),
        outputLayout,
        gws,
        lws);

    if (outputLayout.format != cldnn::format::any &&
        p_currentOutputs->find(genericLayer->name) == p_currentOutputs->end()) {
        // Handle output reorder
        auto reorderPrimName = genericLayer->name + m_postCustomLayerTag;
        m_topology->add(
            cldnn::reorder(
                reorderPrimName,
                genericLayer->name,
                m_defaultFormat,
                m_networkPrecision));
        m_env.primitiveIDs[genericLayer->name] = reorderPrimName;
        m_env.profilingIDs.insert(reorderPrimName);
        InitProfileInfo(reorderPrimName, "Reorder", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
    } else {
        m_env.primitiveIDs[genericLayer->name] = genericLayer->name;
    }
    m_topology->add(customPrim);
    m_env.profilingIDs.insert(genericLayer->name);
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

    auto simpleNMSPrim = cldnn::proposal(
        simpleNMSLayer->name,
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

    m_env.primitiveIDs[simpleNMSLayer->name] = simpleNMSLayer->name;
    m_topology->add(simpleNMSPrim);
    m_env.profilingIDs.insert(simpleNMSLayer->name);
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

    auto eltwisePrim = cldnn::eltwise(
        eltwiseLayer->name,
        inputPrimitives,
        EltwiseModeFromIEEltwise(eltwiseLayer->_operation),
        coefficients);
    m_env.primitiveIDs[eltwiseLayer->name] = eltwiseLayer->name;
    m_topology->add(eltwisePrim);
    m_env.profilingIDs.insert(eltwiseLayer->name);
}

void CLDNNGraph::CreateConcatenatePrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 0);
    auto concatLayer = dynamic_cast<InferenceEngine::ConcatLayer *> (layer.get());
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto concatPrim = cldnn::concatenation(
        concatLayer->name,
        inputPrimitives,
        ConcatAxisFromIEAxis(concatLayer->_axis));
    m_env.primitiveIDs[concatLayer->name] = concatLayer->name;
    m_topology->add(concatPrim);
    m_env.profilingIDs.insert(concatLayer->name);
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
            m_env.profilingIDs.insert(outLayer->name);
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
            case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(dims[1], def));
            case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[2], def));
            case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
            default: THROW_CLDNN_EXCEPTION("Invalid dimensions size(" << dims.size() << ") in split layer");
            }
        };

        for (auto& outLayer : splitLayer->outData) {
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

            auto cropPrim = cldnn::crop(outLayer->name, inputPrimitives[0], outTensor, offsetTensor);
            m_env.primitiveIDs[outLayer->name] = outLayer->name;
            m_topology->add(cropPrim);
            m_env.profilingIDs.insert(outLayer->name);
            InitProfileInfo(outLayer->name, "Crop", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);

            for (size_t i = 0; i < inputDims.size(); i++) {
                if (outLayer->dims[i] != inputDims[i]) {
                    startOffset[i] += outLayer->dims[i];
                }
            }
        }

        // set split as not_run
        InitProfileInfo(layer->name, layer->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);  // Mark this layer as optimized out
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
    InitProfileInfo(convLayer1->name, convLayer1->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);
    InitProfileInfo(convLayer2->name, convLayer2->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);
    InitProfileInfo(concatLayer->name, concatLayer->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);

    // build the split conv primitive
    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(GetNextSingleLayer(splitLayer->outData[0]), weightPrimID, biasPrimID);
    CreateWeightAndBiasPrimitives(GetNextSingleLayer(splitLayer->outData[1]), weightPrimID, biasPrimID);

    auto concatLayerPtr = std::make_shared<InferenceEngine::CNNLayer>(*concatLayer);

    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                         cldnn::spatial(convLayer1->_stride_x, convLayer1->_stride_y));
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
                                          cldnn::spatial(-convLayer1->_padding_x, -convLayer1->_padding_y));
    cldnn::tensor dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                           cldnn::spatial(convLayer1->_dilation_x, convLayer1->_dilation_y));

    auto splitPrim = cldnn::convolution(splitLayer->name,
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

    m_env.primitiveIDs[splitLayer->name]  = splitLayer->name;
    m_env.primitiveIDs[convLayer1->name]  = splitLayer->name;
    m_env.primitiveIDs[convLayer2->name]  = splitLayer->name;
    m_env.primitiveIDs[concatLayer->name] = splitLayer->name;  // pair the last merged layer (concat or relu) with
                                                               // this primitive name to be used as
                                                              // input prim for subsequent layers
    m_topology->add(splitPrim);
    m_env.profilingIDs.insert(splitLayer->name);
}

void CLDNNGraph::CreatePowerPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer *> (layer.get());
    if (powerLayer->power != 1.0f && powerLayer->power != 0.5f) {
        THROW_CLDNN_EXCEPTION("Power Layer " << layer->name << "uses unsupported power value");
    }

    if ((powerLayer->scale == 1.0f) && (powerLayer->offset == 0.0f)) {
        if (powerLayer->power == 0.5f) {
            auto activationPrim = cldnn::activation(powerLayer->name, inputPrimitives[0], activation_sqrt);
            m_topology->add(activationPrim);
            m_env.profilingIDs.insert(powerLayer->name);
            m_env.primitiveIDs[powerLayer->name] = powerLayer->name;
        } else {
            // skip this layer
            m_env.primitiveIDs[powerLayer->name] = inputPrimitives[0];  // register the previous primID for this layer too
            InitProfileInfo(layer->name, layer->type, "None", InferenceEngine::InferenceEngineProfileInfo::NOT_RUN);  // Mark this layer as not run
        }
    } else {
        // create scale primitive
        auto scaleValuePrimName = powerLayer->name + m_scalesTag;
        AddSingleValuePrimitive(scaleValuePrimName,
            DataTypeFromPrecision(powerLayer->precision),
            powerLayer->scale);

        cldnn::primitive_id biasValuePrimName = "";
        if (powerLayer->offset != 0.0f) {
            biasValuePrimName = powerLayer->name + m_biasesTag;
            AddSingleValuePrimitive(biasValuePrimName,
                DataTypeFromPrecision(powerLayer->precision),
                powerLayer->offset);
        }
        auto scalePrim = cldnn::scale(
            powerLayer->name,
            inputPrimitives[0],
            scaleValuePrimName,
            biasValuePrimName);

        m_env.primitiveIDs[powerLayer->name] = powerLayer->name;
        m_topology->add(scalePrim);
        m_env.profilingIDs.insert(powerLayer->name);

        if (powerLayer->power == 0.5f) {
            auto activationPrim = cldnn::activation(powerLayer->name+"_sqrt", powerLayer->name, activation_sqrt);
            m_topology->add(activationPrim);
            m_env.profilingIDs.insert(powerLayer->name+"_sqrt");
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

    auto softmaxPrim = cldnn::softmax(softmaxLayer->name, inputPrimitives[0], SoftmaxDimensionFromIEAxis(softmaxLayer, isPrevFC));
    m_env.primitiveIDs[softmaxLayer->name] = softmaxLayer->name;
    m_topology->add(softmaxPrim);
    m_env.profilingIDs.insert(softmaxLayer->name);
}

void CLDNNGraph::CreateFullyConnectedPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto fcLayer = dynamic_cast<InferenceEngine::FullyConnectedLayer *> (layer.get());

    // create bias primitive
    cldnn::primitive_id biasesPrimID = "";
    if (fcLayer->_biases != nullptr) {
        biasesPrimID = fcLayer->name + m_biasesTag;
        CreatePrimitiveFromBlob(biasesPrimID,
            fcLayer->_biases,
            cldnn::layout(m_networkPrecision, m_defaultFormat,
                cldnn::spatial(TensorValue(fcLayer->_out_num))));
    }

    // create weights primitive
    // gcc bug to resolve auto, at least for 5.4 version
    std::shared_ptr<Data> insData0 = fcLayer->insData[0].lock();
    IE_ASSERT(insData0 != nullptr);
    cldnn::primitive_id weightsPrimID = fcLayer->name + m_weightsTag;
    cldnn::tensor weightsDims;
    switch (insData0->dims.size()) {
    case 4:
        weightsDims = { TensorValue(fcLayer->outData[0]->dims[0]),
                        TensorValue(insData0->dims[2]),
                        TensorValue(insData0->dims[0]),
                        TensorValue(insData0->dims[1]) };
        break;
    case 2:
        weightsDims = { TensorValue(fcLayer->outData[0]->dims[0]), 1, TensorValue(insData0->dims[0]), 1 };
        break;
    default: THROW_CLDNN_EXCEPTION("Invalid data dimensions");
    }
    CreatePrimitiveFromBlob(weightsPrimID,
                            fcLayer->_weights,
                            cldnn::layout(m_networkPrecision, m_defaultFormat, weightsDims));

    auto fcPrim = cldnn::fully_connected(fcLayer->name,
                                         inputPrimitives[0],
                                         weightsPrimID,
                                         biasesPrimID,
                                         false,
                                         0.0f);

    m_env.primitiveIDs[fcLayer->name] = fcLayer->name;
    m_topology->add(fcPrim);
    m_env.profilingIDs.insert(fcLayer->name);
}

void CLDNNGraph::CreatePoolingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto poolLayer = dynamic_cast<InferenceEngine::PoolingLayer *> (layer.get());

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
        case 2: mutableTensor = cldnn::tensor(TensorValue(argmaxDims[1]), 1, TensorValue(argmaxDims[0]), 1);
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
        auto poolPrim = cldnn::pooling(poolLayer->name,
            inputPrimitives[0],
            argmaxPrimID,
            cldnn::pooling_mode::max_with_argmax,
            cldnn::spatial(TensorValue(poolLayer->_kernel_x), TensorValue(poolLayer->_kernel_y)),  // size
            cldnn::spatial(TensorValue(poolLayer->_stride_x), TensorValue(poolLayer->_stride_y)),  // stride
                                                                                                   // input offset (padding) - explicit tensor for 0 bf
            { 0, 0, -TensorValue(poolLayer->_padding_x), -TensorValue(poolLayer->_padding_y) },
            CldnnTensorFromIEDims(poolLayer->outData[0]->dims));
        m_topology->add(poolPrim);
        m_env.primitiveIDs[realOutputID] = poolLayer->name;
    } else {
        // regular pooling
        auto poolPrim = cldnn::pooling(poolLayer->name,
            inputPrimitives[0],
            PoolingModeFromIEPooling(poolLayer->_type, poolLayer->_exclude_pad),
            cldnn::spatial(TensorValue(poolLayer->_kernel_x), TensorValue(poolLayer->_kernel_y)),  // size
            cldnn::spatial(TensorValue(poolLayer->_stride_x), TensorValue(poolLayer->_stride_y)),  // stride
                                                                                                   // input offset (padding) - explicit tensor for 0 bf
            { 0, 0, -TensorValue(poolLayer->_padding_x), -TensorValue(poolLayer->_padding_y) },
            CldnnTensorFromIEDims(poolLayer->outData[0]->dims));
    m_topology->add(poolPrim);
        m_env.primitiveIDs[poolLayer->name] = poolLayer->name;
    }

    m_env.profilingIDs.insert(poolLayer->name);
}

void CLDNNGraph::CreateLRNPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto lrnLayer = dynamic_cast<InferenceEngine::NormLayer *> (layer.get());
    auto lrnPrim = cldnn::lrn(
        lrnLayer->name,
        inputPrimitives[0],
        lrnLayer->_size,
        static_cast<float>(lrnLayer->_k),
        lrnLayer->_alpha,
        lrnLayer->_beta,
        lrnLayer->_isAcrossMaps ? cldnn_lrn_norm_region_across_channel : cldnn_lrn_norm_region_within_channel);

    m_env.primitiveIDs[lrnLayer->name] = lrnLayer->name;
    m_topology->add(lrnPrim);
    m_env.profilingIDs.insert(lrnLayer->name);
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
    default:
        THROW_CLDNN_EXCEPTION("Unsupported activation type (" + layer->type +
                              ") in layer " + layer->name);
    }

    auto activationPrimitive = cldnn::activation(layer->name, inputPrimitives[0], func, params);
    m_env.primitiveIDs[layer->name] = layer->name;
    m_topology->add(activationPrimitive);
    m_env.profilingIDs.insert(layer->name);
}

void CLDNNGraph::CreateCopyPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto copyLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    // Optimize out and just update references
    m_env.primitiveIDs[copyLayer->name] = inputPrimitives[0];
    InitProfileInfo(layer->name, layer->type, "None", InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT);  // Mark this layer as optimized out
}

void CLDNNGraph::CreateUpsamplingPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    // Assuming multi-input will be handled by prev concat/eltwise layers
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto upsamplingLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    uint32_t scale = upsamplingLayer->GetParamAsUInt("scale");
    uint32_t numFilter = upsamplingLayer->GetParamAsUInt("num_filter");
    std::string sampleType = upsamplingLayer->GetParamAsString("sample_type");

    auto upsamplingPrim = cldnn::upsampling(
        upsamplingLayer->name,
        inputPrimitives[0],
        scale,
        numFilter,
        UpsamplingTypeFromString(sampleType));

    m_env.primitiveIDs[upsamplingLayer->name] = upsamplingLayer->name;
    m_topology->add(upsamplingPrim);
    m_env.profilingIDs.insert(upsamplingLayer->name);
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

    auto upsamplingPrim = cldnn::upsampling(
        resampleLayer->name,
        inputPrimitives[0],
        scale,
        inFeatures,
        cldnn::upsampling_sample_type::nearest);

    m_env.primitiveIDs[resampleLayer->name] = resampleLayer->name;
    m_topology->add(upsamplingPrim);
    m_env.profilingIDs.insert(resampleLayer->name);
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

    auto regionPrim = cldnn::region_yolo(
        YOLOregionLayer->name,
        inputPrimitives[0],
        coords,
        classes,
        num,
        mask_size,
        do_softmax);

    m_env.primitiveIDs[YOLOregionLayer->name] = YOLOregionLayer->name;
    m_topology->add(regionPrim);
    m_env.profilingIDs.insert(YOLOregionLayer->name);
}

void CLDNNGraph::CreateYOLO2ReorgPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto YOLOreorgLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());
    uint32_t stride = YOLOreorgLayer->GetParamAsUInt("stride");

    auto reorgPrim = cldnn::reorg_yolo(
        YOLOreorgLayer->name,
        inputPrimitives[0],
        stride);

    m_env.primitiveIDs[YOLOreorgLayer->name] = YOLOreorgLayer->name;
    m_topology->add(reorgPrim);
    m_env.profilingIDs.insert(YOLOreorgLayer->name);
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

    auto argmaxPrim = cldnn::arg_max_min(
        ArgMaxLayer->name,
        inputPrimitives[0],
        otype,
        top_k,
        chosen_axis);

    m_env.primitiveIDs[ArgMaxLayer->name] = ArgMaxLayer->name;
    m_topology->add(argmaxPrim);
    m_env.profilingIDs.insert(ArgMaxLayer->name);
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

    auto unpoolingPrim = cldnn::max_unpooling(
        UnpoolingLayer->name,
        real_input,
        argmax_mutable,
        cldnn::spatial(kernel_size, kernel_size),  // size
        cldnn::spatial(stride, stride) );          // stride

    m_env.primitiveIDs[UnpoolingLayer->name] = UnpoolingLayer->name;
    m_topology->add(unpoolingPrim);
    m_env.profilingIDs.insert(UnpoolingLayer->name);
}

void CLDNNGraph::CreateMVNPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto MvnLayer = dynamic_cast<InferenceEngine::GenericLayer*> (layer.get());

    bool across_channels = MvnLayer->GetParamsAsBool("across_channels", false);
    bool normalize_variance = MvnLayer->GetParamsAsBool("normalize_variance", true);
    float eps = MvnLayer->GetParamAsFloat("eps", 1e-10f);

    auto mvnPrim = cldnn::mvn(
        MvnLayer->name,
        inputPrimitives[0],
        across_channels,
        normalize_variance,
        eps);

    m_env.primitiveIDs[MvnLayer->name] = MvnLayer->name;
    m_topology->add(mvnPrim);
    m_env.profilingIDs.insert(MvnLayer->name);
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
    case 2: constTensor = cldnn::tensor(TensorValue(constDims[1]), 1, TensorValue(constDims[0]), 1);
            break;
        case 1:  // not implemented yet.
        default: THROW_CLDNN_EXCEPTION("Invalid constant blob dimensions");
    }

    cldnn::layout constLayout = cldnn::layout(
        DataTypeFromPrecision(layer->blobs.begin()->second->precision()),
        m_defaultFormat,
        constTensor);

    size_t bytes = constLayout.bytes_count();
    cldnn::primitive_id constPrimID = layer->name;

    auto mem = cldnn::memory::allocate(*(m_env.engine), constLayout);
    auto tmpPointer = mem.pointer<char>();  // implicitly maps buffer - unmap in destructor
    auto buf = tmpPointer.data();

    // fill cldnn::memory from blob
    auto bufSize = constLayout.bytes_count();
    auto data = static_cast<const char *>(constBlob->buffer());
    for (size_t i = 0; i < bufSize; i++) {
        buf[i] = data[i];
    }

    // add new input to topology
    // and put it in const blob map
    // (to set input memory after network compilation)
    m_topology->add(cldnn::input_layout(constPrimID, constLayout));
    m_env.primitiveIDs[layer->name] = constPrimID;
    m_env.constBlobs.insert({ layer->name, mem });
}

void CLDNNGraph::CreateConvolutionPrimitive(InferenceEngine::CNNLayerPtr &layer) {
    ValidateLayer(layer, 1);
    auto inputPrimitives = GetPrevLayersPrimitives(layer);
    auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer *> (layer.get());

    std::vector<cldnn::primitive_id> weightPrimID;
    std::vector<cldnn::primitive_id> biasPrimID;
    CreateWeightAndBiasPrimitives(layer, weightPrimID, biasPrimID);

    cldnn::tensor stride = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                         cldnn::spatial(convLayer->_stride_x, convLayer->_stride_y));
    cldnn::tensor padding = cldnn::tensor(cldnn::batch(0), cldnn::feature(0),
                                          cldnn::spatial(-convLayer->_padding_x, -convLayer->_padding_y));
    cldnn::tensor dilation = cldnn::tensor(cldnn::batch(1), cldnn::feature(1),
                                           cldnn::spatial(convLayer->_dilation_x, convLayer->_dilation_y));

    auto convPrim = cldnn::convolution(convLayer->name,
                                       inputPrimitives[0],
                                       weightPrimID,
                                       biasPrimID,
                                       stride,
                                       padding,
                                       dilation,
                                       false,
                                       0.0f,
                                       CldnnTensorFromIEDims(convLayer->outData[0]->dims));

    m_env.primitiveIDs[convLayer->name] = convLayer->name;
    m_topology->add(convPrim);
    m_env.profilingIDs.insert(convLayer->name);
}

bool CLDNNGraph::IsValidSplitConvMerge(const InferenceEngine::SplitLayer *splitLayer) const {
    if (splitLayer->outData.size() != 2) return false;  // split into 2
    auto convLayer1 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[0]).get());
    auto convLayer2 =
        dynamic_cast<InferenceEngine::ConvolutionLayer *> (GetNextSingleLayer(splitLayer->outData[1]).get());
    if (!convLayer1 || !convLayer2  // outputs aren't convolutions
        || convLayer1->precision != convLayer2->precision                       // wrong precision
        || convLayer1->_fusedWith || convLayer2->_fusedWith                     // convolutions are fused
        || convLayer1->outData.size() != 1 || convLayer2->outData.size() != 1   // more than 1 output for convolutions
        || convLayer1->_padding_x != convLayer2->_padding_x                     // different padding
        || convLayer1->_padding_y != convLayer2->_padding_y                     // different padding
        || convLayer1->_stride_x != convLayer2->_stride_x                       // different strides
        || convLayer1->_stride_y != convLayer2->_stride_y                       // different strides
        || convLayer1->_dilation_x != convLayer2->_dilation_x                   // different dilation
        || convLayer1->_dilation_y != convLayer2->_dilation_y                   // different dilation
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

void CLDNNGraph::AddInputPrimitive(InferenceEngine::InputInfo::Ptr inputInfo) {
    // first create and add the input layout
    auto inputDims = inputInfo->getDims();
    InferenceEngine::Layout l = inputInfo->getTensorDesc().getLayout();

    cldnn::tensor dataTensor;
    switch (inputDims.size()) {
        case 4:
        {
            cldnn::tensor::value_type batch = (m_env.m_max_batch <= 1) ? TensorValue(inputDims[3]) : TensorValue(m_curBatch);

            if (InferenceEngine::Layout::NCHW == l) {
                dataTensor = cldnn::tensor(batch, TensorValue(inputDims[2]),
                    TensorValue(inputDims[0]), TensorValue(inputDims[1]));
            } else if (InferenceEngine::Layout::NHWC == l) {
                dataTensor = cldnn::tensor(batch,
                    TensorValue(inputDims[2]), TensorValue(inputDims[0]),
                    TensorValue(inputDims[1]));
            } else {
                THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(l) << ") in input " + inputInfo->name());
            }
            break;
        }
        case 2:
            if (InferenceEngine::NC == l)
                dataTensor = cldnn::tensor(TensorValue(inputDims[1]), 1, TensorValue(inputDims[0]), 1);
            else
                THROW_CLDNN_EXCEPTION("Unsupported layout (" << DebugOptions::IELayoutToString(l) << ") in input " + inputInfo->name());
            break;
        case 3:  // not implemented yet.
        case 1:  // not implemented yet.
        default: THROW_CLDNN_EXCEPTION("Invalid data dimensions");
    }

    cldnn::layout inputLayout(DataTypeFromPrecision(inputInfo->getInputPrecision()),
        FormatFromLayout(l),
        dataTensor);
    auto inputName = inputInfo->name();
    m_topology->add(cldnn::input_layout(inputName, inputLayout));

    // save the input dims
    m_env.inputLayouts.insert({ inputName, inputLayout });

    // create preprocess primitive for this input
    auto preProcess = inputInfo->getPreProcess();

    size_t meanChannels = preProcess.getNumberOfChannels();
    auto internalInputLayout = m_env.inputLayouts.at(inputName);
    internalInputLayout.format = m_defaultFormat;
    internalInputLayout.size = internalInputLayout.size.transform(m_defaultFormat, 1);
    internalInputLayout.data_type = m_networkPrecision;
    auto preprocessPrimID = inputName + m_preProcessTag;

    if ((meanChannels > 0) &&
        (meanChannels != internalInputLayout.size.feature[0])) {
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
        m_topology->add(cldnn::reorder(preprocessPrimID, inputName, internalInputLayout, meanValues));
        m_env.profilingIDs.insert(preprocessPrimID);
        InitProfileInfo(preprocessPrimID, "Reorder", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
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
        cldnn::tensor meanBlobTensor(internalInputLayout.size);
        meanBlobTensor.batch[0] = 1;  // mean values have no batches
        cldnn::layout meanBlobLayout(cldnn::data_types::f32, m_defaultFormat, meanBlobTensor);
        CreatePrimitiveFromBlob(
            inputName + m_meanValuesTag,
            meanBlobPtr,
            meanBlobLayout);
        m_topology->add(cldnn::reorder(preprocessPrimID,
            inputName,
            internalInputLayout,
            inputName + m_meanValuesTag));
        m_env.profilingIDs.insert(preprocessPrimID);
        InitProfileInfo(preprocessPrimID, "Reorder", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
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
        auto prevName = prevCreator ? prevCreator->name : prevData->name;
        if (prevCreator && prevCreator->outData.size() > 1) {
            inputPrimitives.push_back(m_env.primitiveIDs.at(prevData->name));
        } else {
            inputPrimitives.push_back(m_env.primitiveIDs.at(prevName));
        }
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
    auto outputReorderID = outputName + m_postProcessTag;
    Precision precision = outputPrecision == Precision::UNSPECIFIED ? outputData->getPrecision() : outputPrecision;

    // Find correct output ID. Start with name stored in IR.
    std::string outputID = outputName;
    std::string finalID = m_env.primitiveIDs.at(outputName);

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
    m_env.profilingIDs.insert(outputReorderID);
    InitProfileInfo(outputReorderID, "Reorder", "GPU", InferenceEngine::InferenceEngineProfileInfo::EXECUTED);
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
            layer->name + "_" + blob.first + m_weightsTag,
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
                                 const std::string& execType,
                                 InferenceEngine::InferenceEngineProfileInfo::LayerStatus status) {
    m_env.perfMap[layerName].status = status;
    m_env.perfMap[layerName].cpu_uSec = m_env.perfMap[layerName].realTime_uSec = 0;
    layerType.copy(m_env.perfMap[layerName].layer_type, layerType.length());
    execType.copy(m_env.perfMap[layerName].exec_type, execType.length());
}

};  // namespace CLDNNPlugin
