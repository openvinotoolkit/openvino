// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "dynamic_graph.hpp"
#include "graph.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "weightless_graph.hpp"
#include "weightless_utils.hpp"

namespace intel_npu {

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.info("initialize PluginCompilerAdapter start");

    _logger.info("Loading PLUGIN compiler");
    try {
        auto vclCompilerPtr = VCLCompilerImpl::getInstance();
        OPENVINO_ASSERT(vclCompilerPtr != nullptr, "VCL compiler is nullptr");
        auto vclLib = vclCompilerPtr->getLinkedLibrary();
        _logger.info("PLUGIN VCL compiler is loading");
        OPENVINO_ASSERT(vclLib != nullptr, "VCL library is nullptr");
        _compiler = ov::SoPtr<VCLCompilerImpl>(vclCompilerPtr, vclLib);
    } catch (const std::exception& vcl_exception) {
        OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vcl_exception.what());
    }

    if (_zeroInitStruct == nullptr) {
        return;
    }

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize PluginCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto tensor = _compiler->compile(model, config);
    _logger.debug("compile end");

    if (config.get<COMPILATION_MODE>() == "HostCompile") {
        // metadata will be obtained in initialze() of DynamicGraph
        _logger.debug("Use dynamicGraph to hold blob for HostCompile mode!");
        return std::make_shared<DynamicGraph>(_zeroInitStruct, std::move(tensor), true, config);
    }

    GraphDescriptor graphDesc;
    NetworkMetadata networkMeta;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled
        // network
        try {
            graphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
            networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
            networkMeta.name = model->get_friendly_name();
        } catch (const std::exception& ex) {
            _logger.info("Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                         "allowed. Only exports are available",
                         ex.what());
        }
    } else {
        _logger.warning("No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
    }

    return std::make_shared<Graph>(
        _zeGraphExt,
        _zeroInitStruct,
        graphDesc,
        std::move(networkMeta),
        std::move(tensor),
        config,
        /* persistentBlob = */ true);  // exporting the blob shall be available in such a scenario
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compileWS(std::shared_ptr<ov::Model>&& model,
                                                         const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");
    _logger.debug("compile start");

    FilteredConfig localConfig = config;
    if (!localConfig.has<SEPARATE_WEIGHTS_VERSION>()) {
        localConfig.update({{ov::intel_npu::separate_weights_version.name(), "ONE_SHOT"}});
    }

    _logger.info("SEPARATE_WEIGHTS_VERSION: %s",
                 SEPARATE_WEIGHTS_VERSION::toString(localConfig.get<SEPARATE_WEIGHTS_VERSION>()).c_str());

    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }

    std::vector<ov::Tensor> tensorsInits;
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initNetworkMetadata;

    ov::Tensor tensorMain;
    GraphDescriptor mainGraphDesc;
    NetworkMetadata mainNetworkMetadata;

    switch (localConfig.get<SEPARATE_WEIGHTS_VERSION>()) {
    case ov::intel_npu::WSVersion::ONE_SHOT: {
        std::vector<ov::Tensor> initMainTensors = _compiler->compileWsOneShot(model, localConfig);

        tensorMain = initMainTensors.back();
        initMainTensors.pop_back();
        if (initMainTensors.empty()) {
            _logger.warning("NPU compiler did not produce any init schedules. "
                            "This likely means that the compiled model blob has weights inside even "
                            "though weightless compilation was requested.");
        }

        tensorsInits = std::move(initMainTensors);

        if (_zeGraphExt) {
            // Depending on the config, we may get an error when trying to
            // get the graph handle from the compiled network
            try {
                mainGraphDesc = _zeGraphExt->getGraphDescriptor(tensorMain.data(), tensorMain.get_byte_size());
                mainNetworkMetadata = _zeGraphExt->getNetworkMeta(mainGraphDesc);
            } catch (const std::exception& ex) {
                _logger.info("Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                             "allowed. Only exports are available",
                             ex.what());
            }
        } else {
            _logger.warning(
                "No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
        }

        initGraphDescriptors.reserve(tensorsInits.size());
        initNetworkMetadata.reserve(tensorsInits.size());
        for (const auto& tensor : tensorsInits) {
            GraphDescriptor initGraphDesc;
            NetworkMetadata initNetworkMeta;
            if (_zeGraphExt) {
                try {
                    initGraphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
                    initNetworkMeta = _zeGraphExt->getNetworkMeta(initGraphDesc);
                } catch (const std::exception& ex) {
                    _logger.info(
                        "Failed to use the level zero graph handle: %s. Inference requests for this model are not "
                        "allowed. Only exports are available",
                        ex.what());
                }
            } else {
                _logger.warning(
                    "No driver is found, zeGraphExt is nullptr, so metadata is empty. Only exports are available");
            }

            initGraphDescriptors.push_back(initGraphDesc);
            initNetworkMetadata.push_back(std::move(initNetworkMeta));
        }
    } break;
    case ov::intel_npu::WSVersion::ITERATIVE: {
        OPENVINO_ASSERT(_zeGraphExt,
                        "The \"iterative\" implementation of the weights separation feature requires a Level Zero "
                        "graph handle to compile a model.");

        // The state of the model needs to be reset every iteration
        const std::shared_ptr<ov::Model> originalModel = model->clone();
        std::shared_ptr<ov::Model> targetModel = model;
        size_t i = 0;

        while (auto tensor = _compiler->compileWsIterative(targetModel, localConfig, i++)) {
            GraphDescriptor graphDesc = _zeGraphExt->getGraphDescriptor(tensor.data(), tensor.get_byte_size());
            NetworkMetadata networkMetadata = _zeGraphExt->getNetworkMeta(graphDesc);

            if (isInitMetadata(networkMetadata)) {
                networkMetadata.name = model->get_friendly_name() + "_init";
                targetModel = originalModel->clone();
                initGraphDescriptors.push_back(graphDesc);
                tensorsInits.push_back(std::move(tensor));
                initNetworkMetadata.push_back(std::move(networkMetadata));
                continue;
            }

            networkMetadata.name = model->get_friendly_name() + "_main";
            tensorMain = std::move(tensor);
            mainGraphDesc = graphDesc;
            mainNetworkMetadata = std::move(networkMetadata);
            break;
        }
    } break;
    default:
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call: ",
                       localConfig.get<SEPARATE_WEIGHTS_VERSION>());
        break;
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        auto compile_model_mem_end = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compile_model_mem_start);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compile_model_mem_end);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compile_model_mem_end - compile_model_mem_start);
    }

    _logger.debug("compile end");

    return std::make_shared<WeightlessGraph>(
        _zeGraphExt,
        _zeroInitStruct,
        mainGraphDesc,
        std::move(mainNetworkMetadata),
        std::move(tensorMain),
        initGraphDescriptors,
        std::move(initNetworkMetadata),
        tensorsInits,
        std::move(model),
        localConfig,
        /* persistentBlob = */ true);  // exporting the blob shall be available in such a scenario
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const FilteredConfig& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

uint32_t PluginCompilerAdapter::get_version() const {
    // returning max val as PluginCompiler supports all features and options the plugin is aware of
    return _compiler->get_version();
}

std::optional<std::vector<std::string>> PluginCompilerAdapter::get_supported_options() const {
    std::vector<char> options;
    if (!_compiler->get_supported_options(options)) {
        _logger.warning("VCLCompilerImpl get_supported_options failed. Returning empty supported options.");
        return std::nullopt;
    }

    if (options.empty()) {
        _logger.warning("get_supported_options returned no options; returning an empty supported options vector.");
        return std::vector<std::string>{};
    }

    std::string compilerOptionsStr(options.data(), options.size());
    _logger.debug("VCLCompilerImpl return supported_options: %s", compilerOptionsStr.c_str());
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts = {};
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool PluginCompilerAdapter::is_option_supported(std::string optname, std::optional<std::string> optValue) const {
    const char* optvalue_ch = optValue.has_value() ? optValue.value().c_str() : nullptr;
    if (_compiler->is_option_supported(optname, std::move(optValue))) {
        _logger.debug("Option %s is supported `%s` by VCLCompilerImpl",
                      optname.c_str(),
                      optvalue_ch ? optvalue_ch : "null");
        return true;
    } else {
        _logger.debug("Option %s is not supported `%s` by VCLCompilerImpl",
                      optname.c_str(),
                      optvalue_ch ? optvalue_ch : "null");
        return false;
    }
}

}  // namespace intel_npu
