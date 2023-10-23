// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network_reader.hpp"

#include <fstream>
#include <istream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_cnn_network.h"
#include "dev/converter_utils.hpp"
#include "file_utils.h"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_input_info.hpp"
#include "openvino/frontend/manager.hpp"
#ifdef ENABLE_IR_V7_READER
#    include "legacy/ie_ir_version.hpp"
#endif
#include "itt.hpp"
#include "legacy/ie_reader.hpp"
#include "legacy_op_extension.hpp"
#include "ngraph/function.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/util/shared_object.hpp"
#include "so_ptr.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

#ifdef ENABLE_IR_V7_READER

/**
 * @brief This class is a wrapper for reader interfaces
 */
class Reader : public IReader {
#    ifdef OPENVINO_STATIC_LIBRARY
    using ReaderPtr = std::shared_ptr<IReader>;
#    else
    using ReaderPtr = ov::SoPtr<IReader>;
#    endif
    ReaderPtr ptr;

public:
    using Ptr = std::shared_ptr<Reader>;

    explicit Reader(const std::string& location) {
#    ifdef OPENVINO_STATIC_LIBRARY
        // call library creator directly, since we are in the same application
        InferenceEngine::CreateReader(ptr);
        OPENVINO_ASSERT(ptr != nullptr, "Failed to create static version of IR v7 reader");
#    else
        ov::util::FilePath libraryPath = ov::util::to_file_path(FileUtils::makePluginLibraryName({}, location));
        ov::util::FilePath readersLibraryPath = FileUtils::makePath(getInferenceEngineLibraryPath(), libraryPath);

        if (FileUtils::fileExist(readersLibraryPath)) {
            libraryPath = readersLibraryPath;
        }

        auto so = ov::util::load_shared_object(libraryPath.c_str());
        std::shared_ptr<IReader> plugin_impl;
        using createFunc = void(std::shared_ptr<IReader>&);
        reinterpret_cast<createFunc*>(ov::util::get_symbol(so, "CreateReader"))(plugin_impl);
        ptr = {plugin_impl, so};
#    endif  // OPENVINO_STATIC_LIBRARY
    }

    bool supportModel(std::istream& model) const override {
        OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Reader::supportModel");
        return ptr->supportModel(model);
    }

    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override {
        return ptr->read(model, exts);
    }

    CNNNetwork read(std::istream& model,
                    const Blob::CPtr& weights,
                    const std::vector<IExtensionPtr>& exts) const override {
        return ptr->read(model, weights, exts);
    }

    std::vector<std::string> getDataFileExtensions() const override {
        return ptr->getDataFileExtensions();
    }
};

namespace {

Reader::Ptr reader_irv7 = nullptr;

void registerReaders() {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "registerReaders");
    static bool initialized = false;
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    if (initialized)
        return;

    initialized = true;

    // try to load IR reader v7 if library exists
    try {
        reader_irv7 =
            std::make_shared<Reader>(std::string("inference_engine_ir_v7_reader") + std::string(OV_BUILD_POSTFIX));
    } catch (const std::runtime_error&) {
        // runtime error is thrown in case of library cannot be loaded
    }
}

void assertIfIRv7LikeModel(std::istream& modelStream) {
    auto irVersion = details::get_ir_version(modelStream);
    bool isIRv7 = irVersion > 1 && irVersion <= 7;

    if (!isIRv7 || reader_irv7)
        return;

    IE_THROW() << "The support of IR v" << irVersion
               << " has been removed from the product. "
                  "Please, convert the original model using the Model Optimizer which comes with this "
                  "version of the OpenVINO to generate supported IR version.";
}

CNNNetwork load_ir_v7_network(const std::string& modelPath,
                              const std::string& binPath,
                              const std::vector<IExtensionPtr>& exts) {
    // Fix unicode name
#    if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = ov::util::string_to_wstring(modelPath.c_str());
#    else
    std::string model_path = modelPath;
#    endif

    if (ov::util::directory_exists(modelPath)) {
        return {};
    }

    // Try to open model file
    std::ifstream modelStream(model_path.c_str(), std::ios::binary);
    if (!modelStream.is_open())
        IE_THROW() << "Model file " << modelPath << " cannot be opened!";

    assertIfIRv7LikeModel(modelStream);

    // Check that reader supports the model
    if (reader_irv7 && reader_irv7->supportModel(modelStream)) {
        // Find weights
        std::string bPath = binPath;
        if (bPath.empty()) {
            auto pathWoExt = modelPath;
            auto pos = modelPath.rfind('.');
            if (pos != std::string::npos)
                pathWoExt = modelPath.substr(0, pos);
            for (const auto& ext : reader_irv7->getDataFileExtensions()) {
                bPath = pathWoExt + "." + ext;
                if (!FileUtils::fileExist(bPath)) {
                    bPath.clear();
                } else {
                    break;
                }
            }
        }
        if (!bPath.empty()) {
            // Open weights file
#    if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            std::wstring weights_path = ov::util::string_to_wstring(bPath.c_str());
#    else
            std::string weights_path = bPath;
#    endif
            std::ifstream binStream;
            binStream.open(weights_path.c_str(), std::ios::binary);
            if (!binStream.is_open())
                IE_THROW() << "Weights file " << bPath << " cannot be opened!";

            binStream.seekg(0, std::ios::end);
            size_t fileSize = binStream.tellg();
            binStream.seekg(0, std::ios::beg);

            Blob::Ptr weights = make_shared_blob<uint8_t>({Precision::U8, {fileSize}, C});

            {
                OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ReadTime, "ReadNetworkWeights");
                weights->allocate();
                binStream.read(weights->buffer(), fileSize);
                binStream.close();
            }

            // read model with weights
            auto network = reader_irv7->read(modelStream, weights, exts);
            modelStream.close();
            return network;
        }
        // read model without weights
        return reader_irv7->read(modelStream, exts);
    }

    return {};
}

}  // namespace

#endif  // ENABLE_IR_V7_READER

namespace {

CNNNetwork convert_to_cnnnetwork(std::shared_ptr<ngraph::Function>& function,
                                 const std::vector<IExtensionPtr>& exts,
                                 bool is_new_api,
                                 bool frontendMode = false) {
    // only for IR cases we need preprocessing or postprocessing steps
    if (function->has_rt_info("version") && function->get_rt_info<int64_t>("version") == 11 && !is_new_api) {
        IR_READER_SCOPE(ir11_old_api);
        ov::preprocess::PrePostProcessor prepost(function);

        const std::string& old_api_map_key_order = ov::OldApiMapOrder::get_type_info_static();
        const std::string& old_api_map_key_type = ov::OldApiMapElementType::get_type_info_static();

        bool need_validate_nodes_and_infer_types = false;
        auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            const auto& parameter = parameters[i];
            ov::RTMap& rtInfo = parameter->get_rt_info();
            const auto it_type = rtInfo.find(old_api_map_key_type);
            auto& pre_input = prepost.input(i);
            if (it_type != rtInfo.end()) {
                const auto old_api_map_type = it_type->second.as<ov::OldApiMapElementType>().value;
                const auto param_type = parameter->get_element_type();

                // In the following code we add Convert node from old_api_map_type to Parameter type
                // using PrePostProcessor. As some plugins do not support uint8 type, Convert to uint8 leads
                // to error, so for such case type is set directly to Parameter node instead of inserting Convert.
                if ((param_type == ngraph::element::u8 && old_api_map_type.is_real())) {
                    parameter->set_element_type(old_api_map_type);
                    need_validate_nodes_and_infer_types = true;
                } else {
                    pre_input.tensor().set_element_type(old_api_map_type);
                }

                OPENVINO_ASSERT(!old_api_map_type.is_dynamic(), "Old API map does not support dynamic type");
                rtInfo.erase(it_type);
            }
            const auto it_order = rtInfo.find(old_api_map_key_order);
            if (it_order != rtInfo.end()) {
                const auto order = it_order->second.as<ov::OldApiMapOrder>().value;
                pre_input.preprocess().convert_layout(order);
                rtInfo.erase(it_order);
            }
        }

        auto& results = function->get_results();
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            ov::RTMap& rtInfo = result->get_rt_info();
            const auto it = rtInfo.find(old_api_map_key_order);
            if (it == rtInfo.end())
                continue;

            const auto order = it->second.as<ov::OldApiMapOrder>().value;
            auto& post_output = prepost.output(i);
            post_output.postprocess().convert_layout(order);

            // remove old api once we applied it
            rtInfo.erase(it);
        }

        if (need_validate_nodes_and_infer_types)
            function->validate_nodes_and_infer_types();

        // Set version to 10
        function->set_rt_info<int64_t>(10, "version");

        function = prepost.build();
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(function, exts, is_new_api));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath,
                                const std::string& binPath,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool is_new_api,
                                bool enable_mmap) {
    auto exts = ov::legacy_convert::convert_extension(ov_exts);
#ifdef ENABLE_IR_V7_READER
    // IR v7 obsolete code
    {
        // Register readers if it is needed
        registerReaders();
        auto cnnnetwork = load_ir_v7_network(modelPath, binPath, exts);

        OPENVINO_SUPPRESS_DEPRECATED_START
        if (static_cast<ICNNNetwork::Ptr>(cnnnetwork) != nullptr) {
            return cnnnetwork;
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
#endif  // ENABLE_IR_V7_READER

    // Fix unicode name
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = ov::util::string_to_wstring(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif

    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{model_path};

    if (!binPath.empty()) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::wstring& weights_path = ov::util::string_to_wstring(binPath.c_str());
#else
        const std::string& weights_path = binPath;
#endif
        params.emplace_back(weights_path);
    }
    params.emplace_back(enable_mmap);

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }

    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, is_new_api);
    }

    const auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    std::string FEs;
    for (const auto& fe_name : manager.get_available_front_ends())
        FEs += fe_name + " ";
    IE_THROW(NetworkNotRead) << "Unable to read the model: " << modelPath
                             << " Please check that model format: " << fileExt
                             << " is supported and the model is correct."
                             << " Available frontends: " << FEs;
}

CNNNetwork details::ReadNetwork(const std::string& model,
                                const Blob::CPtr& weights,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool is_new_api,
                                bool frontendMode) {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;
    auto exts = ov::legacy_convert::convert_extension(ov_exts);

#ifdef ENABLE_IR_V7_READER
    // IR v7 obsolete code
    {
        // Register readers if it is needed
        registerReaders();
        assertIfIRv7LikeModel(modelStream);

        if (reader_irv7 && reader_irv7->supportModel(modelStream)) {
            if (weights)
                return reader_irv7->read(modelStream, weights, exts);
            return reader_irv7->read(modelStream, exts);
        }
    }
#endif  // ENABLE_IR_V7_READER

    // Try to load with FrontEndManager
    ov::frontend::FrontEndManager manager;
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream};
    if (weights) {
        char* data = weights->cbuffer().as<char*>();
        std::shared_ptr<ngraph::runtime::AlignedBuffer> weights_buffer =
            std::make_shared<ngraph::runtime::SharedBuffer<Blob::CPtr>>(data, weights->byteSize(), weights);
        params.emplace_back(weights_buffer);
    }

    FE = manager.load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        inputModel = FE->load(params);
    }
    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, is_new_api, frontendMode);
    }

    IE_THROW(NetworkNotRead)
        << "Unable to read the model. Please check if the model format is supported and model is correct.";
}

}  // namespace InferenceEngine
