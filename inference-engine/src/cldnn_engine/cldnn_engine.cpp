// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <algorithm>

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <cmath>
#include <tuple>
#include <cctype>

#include "ie_metric_helpers.hpp"
#include <ie_data.h>
#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>
#include <memory>
#include "ie_plugin_config.hpp"
#include "caseless.hpp"
#include <legacy/details/ie_cnn_network_tools.h>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <generic_ie.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformations/op_conversions/gru_cell_decomposition.hpp>
#include <transformations/op_conversions/lstm_cell_decomposition.hpp>
#include <transformations/op_conversions/rnn_cell_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>

#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_transformer.h>

#include "cldnn_engine.h"
#include "cldnn_executable_network.h"
#include "cldnn_custom_layer.h"

#include <low_precision/transformer.hpp>
#include <low_precision/mat_mul.hpp>

#ifdef __linux__
# include <dlfcn.h>
#endif

using InferenceEngine::DescriptionBuffer;
using InferenceEngine::TBlob;
using InferenceEngine::Blob;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

struct clDNNEngine::impl {
    CLDNNPlugin::Config m_config;
};

cldnn::device_info clDNNEngine::GetDeviceInfo(const std::map<std::string, std::string> &config) const {
    auto device_info = device_map.begin()->second.get_info();
    if (config.find(PluginConfigParams::KEY_DEVICE_ID) != config.end()) {
        auto val = config.at(PluginConfigParams::KEY_DEVICE_ID);
        if (device_map.find(val) == device_map.end()) {
            THROW_IE_EXCEPTION << "Invalid device ID: " << val;
        }
        device_info = device_map.at(val).get_info();
    }

    return device_info;
}

InferenceEngine::ICNNNetwork::Ptr clDNNEngine::CloneAndTransformNetwork(const InferenceEngine::ICNNNetwork& network, CLDNNPlugin::Config config) const {
    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);
    bool baselineIsFP16 = false;

    if (clonedNetwork->getFunction()) {
        const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
            // Reshape->Permute->Reshape pattern in theory can change output rank, so this check is added to be sure
            // that the following primitives will be handled correctly
            // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
            if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
                return dtsOp->input_value(0).get_shape().size() <= 5lu && dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
            }

            // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
            if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
                return stdOp->input_value(0).get_shape().size() <= 5lu && stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
            }

            // Reduce node implementation with reduce along features performs better with Reshape->Pooling->Reshape pattern
            // Reshape->Pooling->Reshape scenario is also more optimal in case when batch > 1 and network precission is FP16
            if (auto redOp = std::dynamic_pointer_cast<const ::ngraph::opset1::ReduceMean>(node)) {
                auto reduction_axes = redOp->get_reduction_axes().to_vector();
                bool reduce_along_f = redOp->get_reduction_axes().size() == 1 && std::count(reduction_axes.begin(), reduction_axes.end(), 1) != 0;
                bool fp16_batch_not_1 = redOp->get_element_type() == ngraph::element::f16 && redOp->input(0).get_shape()[0] != 1;
                bool can_use_reduce = !reduce_along_f && !fp16_batch_not_1;
                return can_use_reduce;
            }
            if (auto redOp = std::dynamic_pointer_cast<const ::ngraph::opset1::ReduceMax>(node)) {
                auto reduction_axes = redOp->get_reduction_axes().to_vector();
                bool reduce_along_f = redOp->get_reduction_axes().size() == 1 && std::count(reduction_axes.begin(), reduction_axes.end(), 1) != 0;
                bool fp16_batch_not_1 = redOp->get_element_type() == ngraph::element::f16 && redOp->input(0).get_shape()[0] != 1;
                bool can_use_reduce = !reduce_along_f && !fp16_batch_not_1;
                return can_use_reduce;
            }
            if (auto redOp = std::dynamic_pointer_cast<const ::ngraph::opset1::ReduceSum>(node)) {
                auto reduction_axes = redOp->get_reduction_axes().to_vector();
                bool reduce_along_f = redOp->get_reduction_axes().size() == 1 && std::count(reduction_axes.begin(), reduction_axes.end(), 1) != 0;
                bool fp16_batch_not_1 = redOp->get_element_type() == ngraph::element::f16 && redOp->input(0).get_shape()[0] != 1;
                bool can_use_reduce = !reduce_along_f && !fp16_batch_not_1;
                return can_use_reduce;
            }

            if (auto add_op = std::dynamic_pointer_cast<const ngraph::opset1::Add>(node)) {
                return ngraph::is_type<ngraph::opset1::Convolution>(add_op->get_input_node_shared_ptr(0)) ||
                       ngraph::is_type<ngraph::opset1::GroupConvolution>(add_op->get_input_node_shared_ptr(0)) ||
                       ngraph::is_type<ngraph::opset1::MatMul>(add_op->get_input_node_shared_ptr(0));
            }

            return std::dynamic_pointer_cast<const ::ngraph::opset2::Gelu>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset3::ShuffleChannels>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset2::BatchToSpace>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset2::SpaceToBatch>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset5::HSigmoid>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset4::HSwish>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset4::ReduceL1>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset4::ReduceL2>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset4::SoftPlus>(node) ||
                   std::dynamic_pointer_cast<const ::ngraph::opset5::LogSoftmax>(node);
        };
        auto nGraphFunc = clonedNetwork->getFunction();
        // Disable shape inference (WA for generic operations)
        ::ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

        bool enableInt8;
        {
            // Note: instead of running all Conversion Transformations you can make up your own transformation pipeline
            ngraph::pass::Manager manager;
            using const_node_ptr = const std::shared_ptr<const ngraph::Node>;
            const auto& pass_config = manager.get_pass_config();
            manager.register_pass<ngraph::pass::InitNodeInfo>();
            // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
            manager.register_pass<ngraph::pass::ConvertPriorBox>();
            manager.register_pass<ngraph::pass::CommonOptimizations>();
            manager.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
            manager.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
            manager.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
            manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
            manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
            manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
            manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
            manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
            manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
            manager.register_pass<ngraph::pass::GRUCellDecomposition>();
            manager.register_pass<ngraph::pass::RNNCellDecomposition>();

            manager.set_callback(transformations_callback);

            auto isCellPrimitiveSupported = [](const_node_ptr &node) -> bool {
                if (const auto &rnn_cell = std::dynamic_pointer_cast<const ngraph::opset4::RNNCell>(node)) {
                    return false;
                } else if (const auto &gru_cell = std::dynamic_pointer_cast<const ngraph::opset4::GRUCell>(
                        node)) {
                    return false;
                } else if (const auto &lstm_cell = std::dynamic_pointer_cast<const ngraph::opset4::LSTMCell>(
                        node)) {
                    return lstm_cell->get_clip() == 0.0f &&
                           lstm_cell->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
                } else if (const auto &lstm_cell_v1 = std::dynamic_pointer_cast<const ngraph::opset1::LSTMCell>(
                        node)) {
                    return lstm_cell_v1->get_clip() == 0.0f &&
                           lstm_cell_v1->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"};
                }
                return false;
            };

            pass_config->set_callback<ngraph::pass::RNNCellDecomposition, ngraph::pass::GRUCellDecomposition,
                    ngraph::pass::LSTMCellDecomposition>(
                    [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                        return isCellPrimitiveSupported(node);
                    });

            pass_config->set_callback<ngraph::pass::ConvertTensorIteratorToRNNSequence,
                    ngraph::pass::ConvertTensorIteratorToLSTMSequence,
                    ngraph::pass::ConvertTensorIteratorToGRUSequence>(
                    [isCellPrimitiveSupported](const_node_ptr &node) -> bool {
                        if (const auto& ti_op = std::dynamic_pointer_cast<const ngraph::op::TensorIterator>(node)) {
                            size_t count_rnn = 0;
                            for (const auto &op : ti_op->get_body()->get_ops())
                                count_rnn += isCellPrimitiveSupported(op);
                            return count_rnn != 1;
                        }
                        return true;
                    });
            manager.run_passes(nGraphFunc);

            enableInt8 = config.enableInt8 && ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(nGraphFunc);
            if (enableInt8) {
                const auto fp16_callback = [&baselineIsFP16](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
                    if (!baselineIsFP16 && node->get_output_element_type(0) == ngraph::element::f16) {
                        baselineIsFP16 = true;
                    }

                    return true;
                };

                ngraph::pass::Manager conversion_manager;
                // [WA part1] Convert quantized FP16 model to FP32 to avoid possible overflow and mixed precision errors
                conversion_manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
                conversion_manager.set_callback(fp16_callback);
                conversion_manager.run_passes(nGraphFunc);
            }
        }

        using namespace ngraph::pass::low_precision;
        if (enableInt8) {
            auto params = LayerTransformation::Params(
                true,  // updatePrecisions
                LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
                LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
                true);  // supportAsymmetricQuantization
            LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params)
                .add<MatMulTransformation, ngraph::opset1::MatMul>(LayerTransformation::Params(params).setSupportAsymmetricQuantization(false)));

            transformer.transform(nGraphFunc);
        }

        {
            ngraph::pass::Manager manager = ngraph::pass::Manager();
            manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
            manager.register_pass<ngraph::pass::UnrollTensorIterator>();
            manager.set_callback(transformations_callback);
            manager.run_passes(nGraphFunc);
        }

        clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);
    }

    auto implNetwork = std::dynamic_pointer_cast<InferenceEngine::details::CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
    }

    if (baselineIsFP16) {
        // [WA part1] Store 'lpt_back_to_fp16' flag to convert FP32 operations to original FP16 after LPT
        InputsDataMap inputsMap;
        clonedNetwork->getInputsInfo(inputsMap);

        if (!inputsMap.empty()) {
            auto input0 = getInputTo(inputsMap.begin()->second->getInputData());
            input0.begin()->second->params["lpt_back_to_fp16"];
        }
    }

    return clonedNetwork;
}

clDNNEngine::clDNNEngine() : m_defaultContext(nullptr) {
    _pluginName = "GPU";
    _impl = std::make_shared<impl>();

    // try loading clDNN engine and get info from it
    {
        cldnn::device_query device_query;
        device_map = device_query.get_available_devices();
    }
    // locate global custom kernel config
    // and auto-load kernels from it
#ifdef _WIN32
    CHAR mpath[MAX_PATH + 1];
    HMODULE nModule;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)CLDNNCustomLayer::LoadFromFile,
        &nModule);
    GetModuleFileName(nModule, mpath, sizeof(mpath));
#elif __linux__
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(CLDNNCustomLayer::LoadFromFile), &dl_info);
    const char* mpath = dl_info.dli_fname;
#endif
    std::string configFile(mpath);
    std::size_t dir_split_pos = configFile.find_last_of("/\\");
    std::string config_path;

    if (dir_split_pos != std::string::npos) {
        // path contains directory
        config_path = configFile.substr(0, dir_split_pos);
    }
    config_path += "/cldnn_global_custom_kernels/cldnn_global_custom_kernels.xml";
    CLDNNCustomLayer::LoadFromFile(config_path, _impl->m_config.customLayers, true);
}

auto check_inputs = [](InferenceEngine::InputsDataMap _networkInputs) {
    for (auto ii : _networkInputs) {
        auto input_precision = ii.second->getTensorDesc().getPrecision();
        if (input_precision != InferenceEngine::Precision::FP16 &&
            input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::I64 &&
            input_precision != InferenceEngine::Precision::BOOL) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                << "Input image format " << input_precision << " is not supported yet...";
        }
    }
};

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               const std::map<std::string, std::string> &config) {
    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    check_inputs(_networkInputs);

    CLDNNPlugin::Config conf = _impl->m_config;
    auto device_info = GetDeviceInfo(config);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(config);

    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }

    CLDNNRemoteCLContext::Ptr context;

    auto canReuseDefaultContext = [&]() -> bool {
        if (m_defaultContext == nullptr)
            return false;

        const Config& context_config = m_defaultContext->GetConfig();
        const Config& current_config = conf;

        return context_config.throughput_streams == current_config.throughput_streams &&
               context_config.useProfiling == current_config.useProfiling &&
               context_config.dumpCustomKernels == current_config.dumpCustomKernels &&
               context_config.memory_pool_on == current_config.memory_pool_on &&
               context_config.queueThrottle == current_config.queueThrottle &&
               context_config.queuePriority == current_config.queuePriority &&
               context_config.sources_dumps_dir == current_config.sources_dumps_dir &&
               context_config.tuningConfig.mode == current_config.tuningConfig.mode &&
               context_config.tuningConfig.cache_file_path == current_config.tuningConfig.cache_file_path &&
               context_config.kernels_cache_dir == current_config.kernels_cache_dir &&
               context_config.device_id == current_config.device_id;
    };

    {
        std::lock_guard<std::mutex> lock(engine_mutex);
        if (!canReuseDefaultContext()) {
            m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), ParamMap(), conf));
        }
    }

    context = m_defaultContext;

    return std::make_shared<CLDNNExecNetwork>(*CloneAndTransformNetwork(network, conf), context, conf);
}

ExecutableNetworkInternal::Ptr clDNNEngine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network,
                                                               RemoteContext::Ptr context,
                                                               const std::map<std::string, std::string> &config) {
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    check_inputs(_networkInputs);

    auto casted = std::dynamic_pointer_cast<ClContext>(context);
    if (nullptr == casted) {
        THROW_IE_EXCEPTION << "Invalid context";
    }

    CLDNNPlugin::Config conf = getContextImpl(casted)->GetConfig();
    auto device_info = GetDeviceInfo(config);
    conf.enableInt8 = device_info.supports_imad || device_info.supports_immad;
    conf.UpdateFromMap(config);

    if (conf.enableDynamicBatch) {
        conf.max_dynamic_batch = static_cast<int>(network.getBatchSize());
    }

    return std::make_shared<CLDNNExecNetwork>(*CloneAndTransformNetwork(network, conf), casted, conf);
}

RemoteContext::Ptr clDNNEngine::CreateContext(const ParamMap& params) {
    // parameter map is non-empty
    std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

    if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
        auto context = std::make_shared<CLDNNRemoteCLContext>(shared_from_this(), params, _impl->m_config);
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
        #ifdef WIN32
        auto context = std::make_shared<CLDNNRemoteD3DContext>(shared_from_this(), params, _impl->m_config);
        #else
        auto context = std::make_shared<CLDNNRemoteVAContext>(shared_from_this(), params, _impl->m_config);
        #endif
        return std::dynamic_pointer_cast<RemoteContext>(context);
    } else {
        THROW_IE_EXCEPTION << "Invalid remote context type" << contextTypeStr;
    }
}

RemoteContext::Ptr clDNNEngine::GetDefaultContext(const ParamMap& params) {
    if (nullptr == m_defaultContext) {
        m_defaultContext.reset(new CLDNNRemoteCLContext(shared_from_this(), params, _impl->m_config));
    }
    return std::dynamic_pointer_cast<RemoteContext>(m_defaultContext);
}

void clDNNEngine::SetConfig(const std::map<std::string, std::string> &config) {
    _impl->m_config.UpdateFromMap(config);
}

QueryNetworkResult clDNNEngine::QueryNetwork(const ICNNNetwork& network,
                                             const std::map<std::string, std::string>& config) const {
    QueryNetworkResult res;
    GetDeviceInfo(config);      // Verify device id
    auto function = network.getFunction();
    if (function != nullptr) {
        std::unordered_set<std::string> originalOps;
        for (auto&& node : function->get_ops()) {
            originalOps.emplace(node->get_friendly_name());
        }
        auto clonedNetwork = CloneAndTransformNetwork(network, _impl->m_config);
        std::unordered_set<std::string> supported;
        std::unordered_set<std::string> unsupported;

        std::unordered_set<std::string> splitNames;
        std::unordered_set<std::string> concatNames;
        std::unordered_set<std::string> depLayerNames;

        std::vector<std::shared_ptr<ngraph::Node>> splits;
        std::vector<std::shared_ptr<ngraph::Node>> concats;
        std::vector<std::shared_ptr<ngraph::Node>> nextLayerDependent;

        for (CNNNetworkIterator itLayer{clonedNetwork.get()};
             itLayer != CNNNetworkIterator();
             itLayer++) {
            auto layerIsSupported = [&] {
                auto node = (*itLayer)->getNode();
                if (std::dynamic_pointer_cast<const ::ngraph::opset3::DetectionOutput>(node) != nullptr ||
                    std::dynamic_pointer_cast<const ::ngraph::opset3::PriorBox>(node) != nullptr ||
                    std::dynamic_pointer_cast<const ::ngraph::opset3::PriorBoxClustered>(node) != nullptr ||
                    std::dynamic_pointer_cast<const ::ngraph::opset3::Proposal>(node) != nullptr) {
                    return false;
                } else if (std::dynamic_pointer_cast<const ::ngraph::opset3::Split>(node) != nullptr) {
                    splitNames.emplace(node->get_friendly_name());
                    splits.push_back(node);
                    return false;
                } else if (std::dynamic_pointer_cast<const ::ngraph::opset3::Concat>(node) != nullptr) {
                    concatNames.emplace(node->get_friendly_name());
                    concats.push_back(node);
                    return false;
                } else if (std::dynamic_pointer_cast<const ::ngraph::opset3::Reshape>(node) != nullptr ||
                           std::dynamic_pointer_cast<const ::ngraph::opset3::Squeeze>(node) != nullptr ||
                           std::dynamic_pointer_cast<const ::ngraph::opset3::Unsqueeze>(node) != nullptr ||
                           std::dynamic_pointer_cast<const ::ngraph::opset3::Transpose>(node) != nullptr ||
                           ngraph::op::is_constant(node)) {
                    depLayerNames.emplace(node->get_friendly_name());
                    nextLayerDependent.push_back(node);
                    return false;
                } else if (CLDNNGraph::IsLayerSupported((*itLayer)->type)) {
                    return true;
                } else {
                    return false;
                }
            }();
            const auto fusedNode = (*itLayer)->getNode();
            if (fusedNode == nullptr) {
                // skip layers completely generated by IR transformation
                continue;
            }
            for (auto&& fusedLayerName : ngraph::getFusedNamesVector(fusedNode)) {
                if (contains(originalOps, fusedLayerName)) {
                    if (layerIsSupported) {
                        supported.emplace(fusedLayerName);
                    } else {
                        unsupported.emplace(fusedLayerName);
                    }
                }
            }
        }

        for (auto&& layerName : supported) {
            if (contains(unsupported, layerName)) {
                supported.erase(layerName);
            }
        }
        unsupported.clear();

        for (const auto & split : splits) {
            bool is_supported = true;
            const auto outputs = split->outputs();
            for (const auto& output : outputs) {
                const auto& name = output.get_node()->get_friendly_name();
                if (!contains(supported, name) &&
                    !contains(depLayerNames, name) &&
                    !contains(concatNames, name) &&
                    !contains(splitNames, name)) {
                    is_supported = false;
                    break;
                }
            }
            if (is_supported) {
                supported.emplace(split->get_friendly_name());
            }
        }

        for (const auto& concat : concats) {
            bool is_supported = true;
            const auto inputs = concat->inputs();
            for (const auto& input : inputs) {
                const auto& name = input.get_node()->get_friendly_name();
                if (!contains(supported, name) &&
                    !contains(depLayerNames, name) &&
                    !contains(concatNames, name)) {
                    is_supported = false;
                    break;
                }
            }
            if (is_supported) {
                supported.emplace(concat->get_friendly_name());
            }
        }

        for (const auto& cnl : nextLayerDependent) {
            bool is_supported = true;
            // both inputs and output should be GPU to remain on GPU
            const auto inputs = cnl->inputs();
            for (const auto& input : inputs) {
                const auto& name = input.get_node()->get_friendly_name();
                if (!contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
            const auto outputs = cnl->outputs();
            for (const auto& output : outputs) {
                const auto& name = output.get_node()->get_friendly_name();
                if (!contains(supported, name)) {
                    is_supported = false;
                    break;
                }
            }
            if (is_supported) {
                supported.emplace(cnl->get_friendly_name());
            }
        }

        for (auto&& node : function->get_ops()) {
            if (contains(supported, node->get_friendly_name())) {
                for (auto&& inputNodeOutput : node->input_values()) {
                    if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
                        supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                    }
                }
                for (auto&& outputs : node->outputs()) {
                    for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                        if (ngraph::op::is_output(outputNodeInput.get_node())) {
                            supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                        }
                    }
                }
            }
        }

        for (auto&& layerName : supported) {
            res.supportedLayersMap.emplace(layerName, GetName());
        }
    } else {
        std::vector<CNNLayer::Ptr> concats;
        std::vector<CNNLayer::Ptr> nextLayerDependent;
        std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
        for (auto layer : sortedLayers) {
            if (CaselessEq<std::string>()(layer->type, "DetectionOutput")) {
            } else if (CaselessEq<std::string>()(layer->type, "PriorBox")) {
            } else if (CaselessEq<std::string>()(layer->type, "Proposal")) {
            } else if (CaselessEq<std::string>()(layer->type, "SimplerNMS")) {
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
                concats.push_back(layer);
            } else if (CaselessEq<std::string>()(layer->type, "reshape")) {
                nextLayerDependent.push_back(layer);
            } else if (CaselessEq<std::string>()(layer->type, "permute")) {
                nextLayerDependent.push_back(layer);
            } else if (CaselessEq<std::string>()(layer->type, "Const")) {
                nextLayerDependent.push_back(layer);
            } else if (CLDNNGraph::IsLayerSupported(layer->type)) {
                res.supportedLayersMap.insert({ layer->name, GetName() });
            }
        }
        // evaluation of concats - if all parent layers are supported, only in this case we
        // will mark concat as a supported for GPU
        for (const auto& concat : concats) {
            // take all parrents.
            bool supported = true;
            for (DataWeakPtr insData : concat->insData) {
                CNNLayerPtr prev = getCreatorLayer(insData.lock()).lock();
                // verify if previous layer is not supported or if it in the list of not defined layers yet
                // not defined layers are treated as layers which will be assigned to GPU if next layer is assigned to GPU
                if (res.supportedLayersMap.find(prev->name) == res.supportedLayersMap.end()
                    && std::find(nextLayerDependent.begin(), nextLayerDependent.end(), prev) == nextLayerDependent.end()) {
                    supported = false;
                }
            }
            if (supported) {
                res.supportedLayersMap.insert({ concat->name, GetName() });
            }
        }

        // evaluation of constant blobs - if all consumers are on GPU,
        // then leave it on GPU, else - move to other device
        for (auto cnl = nextLayerDependent.rbegin();
            cnl != nextLayerDependent.rend();
            cnl++) {
            bool supported = true;
            for (DataPtr out : (*cnl)->outData) {
                for (auto ol : getInputTo(out)) {
                    if (res.supportedLayersMap.find(ol.second->name) == res.supportedLayersMap.end()) {
                        supported = false;
                    }
                }
            }

            if (supported) {
                res.supportedLayersMap.insert({ (*cnl)->name, GetName() });
            }
        }
    }

    return res;
}

Parameter clDNNEngine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    Parameter result;
    auto option = _impl->m_config.key_config_map.find(name);
    if (option != _impl->m_config.key_config_map.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key : " << name;
    }
    return result;
}

auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
    auto ret_str = string;
    if (!case_sensitive) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        std::transform(substring.begin(), substring.end(), substring.begin(), ::tolower);
    }
    auto erase_position = string.rfind(substring);
    if (erase_position != std::string::npos) {
        // if space exists before substring remove it also
        if (std::isspace(string.at(erase_position - 1))) {
            erase_position--;
        }
        return ret_str.substr(0, erase_position);
    }
    return ret_str;
};

Parameter clDNNEngine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    auto device_id = GetConfig(CONFIG_KEY(DEVICE_ID), {});
    if (options.find(CONFIG_KEY(DEVICE_ID)) != options.end())
        device_id = options.at(CONFIG_KEY(DEVICE_ID)).as<std::string>();

    auto iter = device_map.find(device_id);
    auto device_info = iter != device_map.end() ?
        iter->second.get_info() :
        device_map.begin()->second.get_info();

    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { };
        for (auto const& dev : device_map)
            availableDevices.push_back(dev.first);
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, deviceName);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto opt : _impl->m_config.key_config_map)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;

        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(BIN));
        if (device_info.supports_fp16)
            capabilities.push_back(METRIC_VALUE(FP16));
        if (device_info.supports_imad || device_info.supports_immad)
            capabilities.push_back(METRIC_VALUE(INT8));

        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, 2);
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

};  // namespace CLDNNPlugin

static const Version version = { {2, 1}, CI_BUILD_NUMBER, "clDNNPlugin" };
IE_DEFINE_PLUGIN_CREATE_FUNCTION(CLDNNPlugin::clDNNEngine, version)
