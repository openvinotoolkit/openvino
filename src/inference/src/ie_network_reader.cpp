// Copyright (C) 2018-2022 Intel Corporation
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
#include "file_utils.h"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_input_info.hpp"
#include "openvino/frontend/manager.hpp"
#ifdef ENABLE_IR_V7_READER
#    include "legacy/ie_ir_version.hpp"
#endif
#include "ie_itt.hpp"
#include "legacy/ie_reader.hpp"
#include "ngraph/function.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/util/shared_object.hpp"
#include "so_ptr.hpp"
#include "transformations/rt_info/old_api_map_order_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {

/*
 * @brief Wrapper for old IE extensions to new API
 */
class ExtensionWrapper : public ov::BaseOpExtension {
public:
    ExtensionWrapper(const InferenceEngine::IExtensionPtr& ext, const std::string& opset, const std::string& name)
        : m_ext(ext),
          m_opset_name(opset),
          m_type(name),
          m_ext_type(m_type.c_str(), 0, m_opset_name.c_str()) {}

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return m_ext_type;
    }

    ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) const override {
        std::shared_ptr<ngraph::Node> node(m_ext->getOpSets().at(m_opset_name).create_insensitive(m_ext_type.name));

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        return {};
    }

private:
    InferenceEngine::IExtensionPtr m_ext;
    std::string m_opset_name;
    std::string m_type;
    ov::DiscreteTypeInfo m_ext_type;
};

}  // namespace ov

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
    std::once_flag readFlag;
    std::string name;
    std::string location;

    ReaderPtr getReaderPtr() {
        std::call_once(readFlag, [&]() {
#    ifdef OPENVINO_STATIC_LIBRARY
            // call library creator directly, since we are in the same application
            InferenceEngine::CreateReader(ptr);
            OPENVINO_ASSERT(ptr != nullptr, "Failed to create static version of IR v7 reader");
#    else
            ov::util::FilePath libraryName = ov::util::to_file_path(location);
            ov::util::FilePath readersLibraryPath =
                FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                IE_THROW() << "Please, make sure that Inference Engine reader library exists "
                           << ov::util::from_file_path(::FileUtils::makePluginLibraryName({}, libraryName)) << " is in "
                           << getIELibraryPath();
            }

            auto so = ov::util::load_shared_object(readersLibraryPath.c_str());
            std::shared_ptr<IReader> plugin_impl;
            using createFunc = void(std::shared_ptr<IReader>&);
            reinterpret_cast<createFunc*>(ov::util::get_symbol(so, "CreateReader"))(plugin_impl);
            ptr = {plugin_impl, so};
#    endif  // OPENVINO_STATIC_LIBRARY
        });

        return ptr;
    }

    ReaderPtr getReaderPtr() const {
        return const_cast<Reader*>(this)->getReaderPtr();
    }

public:
    using Ptr = std::shared_ptr<Reader>;

    Reader(const std::string& name, const std::string location) : name(name), location(location) {}
    bool supportModel(std::istream& model) const override {
        OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Reader::supportModel");
        auto reader = getReaderPtr();
        return reader->supportModel(model);
    }
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, exts);
    }
    CNNNetwork read(std::istream& model,
                    const Blob::CPtr& weights,
                    const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, weights, exts);
    }
    std::vector<std::string> getDataFileExtensions() const override {
        auto reader = getReaderPtr();
        return reader->getDataFileExtensions();
    }
    std::string getName() const {
        return name;
    }
};

namespace {

// Extension to plugins creator
std::multimap<std::string, Reader::Ptr> readers;

void registerReaders() {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "registerReaders");
    static bool initialized = false;
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    if (initialized)
        return;

    auto create_if_exists = [](const std::string name, const std::string library_name) {
#    ifndef OPENVINO_STATIC_LIBRARY
        ov::util::FilePath libraryName = ov::util::to_file_path(library_name);
        ov::util::FilePath readersLibraryPath =
            FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

        if (!FileUtils::fileExist(readersLibraryPath))
            return std::shared_ptr<Reader>();
#    endif  // !OPENVINO_STATIC_LIBRARY
        return std::make_shared<Reader>(name, library_name);
    };

    // try to load IR reader v7 if library exists
    auto irReaderv7 =
        create_if_exists("IRv7", std::string("inference_engine_ir_v7_reader") + std::string(IE_BUILD_POSTFIX));
    if (irReaderv7)
        readers.emplace("xml", irReaderv7);

    initialized = true;
}

void assertIfIRv7LikeModel(std::istream& modelStream) {
    auto irVersion = details::GetIRVersion(modelStream);
    bool isIRv7 = irVersion > 1 && irVersion <= 7;

    if (!isIRv7)
        return;

    for (auto&& kvp : readers) {
        Reader::Ptr reader = kvp.second;
        // if we have reader for IR v7
        if (reader->getName() == "IRv7") {
            return;
        }
    }

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

    // Try to open model file
    std::ifstream modelStream(model_path, std::ios::binary);
    if (!modelStream.is_open())
        IE_THROW() << "Model file " << modelPath << " cannot be opened!";

    assertIfIRv7LikeModel(modelStream);

    auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    for (auto it = readers.lower_bound(fileExt); it != readers.upper_bound(fileExt); it++) {
        auto reader = it->second;
        // Check that reader supports the model
        if (reader->supportModel(modelStream)) {
            // Find weights
            std::string bPath = binPath;
            if (bPath.empty()) {
                auto pathWoExt = modelPath;
                auto pos = modelPath.rfind('.');
                if (pos != std::string::npos)
                    pathWoExt = modelPath.substr(0, pos);
                for (const auto& ext : reader->getDataFileExtensions()) {
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
                binStream.open(weights_path, std::ios::binary);
                if (!binStream.is_open())
                    IE_THROW() << "Weights file " << bPath << " cannot be opened!";

                binStream.seekg(0, std::ios::end);
                size_t fileSize = binStream.tellg();
                binStream.seekg(0, std::ios::beg);

                Blob::Ptr weights = make_shared_blob<uint8_t>({Precision::U8, {fileSize}, C});

                {
                    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "ReadNetworkWeights");
                    weights->allocate();
                    binStream.read(weights->buffer(), fileSize);
                    binStream.close();
                }

                // read model with weights
                auto network = reader->read(modelStream, weights, exts);
                modelStream.close();
                return network;
            }
            // read model without weights
            return reader->read(modelStream, exts);
        }
    }

    return {};
}

}  // namespace

#endif  // ENABLE_IR_V7_READER

namespace {

CNNNetwork convert_to_cnnnetwork(std::shared_ptr<ngraph::Function>& function,
                                 const std::vector<IExtensionPtr>& exts,
                                 bool newAPI) {
    auto& rt_info = function->get_rt_info();
    const auto it = rt_info.find("version");
    const bool is_ir = it != rt_info.end();

    // only for IR cases we need preprocessing or postprocessing steps
    if (is_ir) {
        using namespace ov::preprocess;
        PrePostProcessor prepost(function);

        const int64_t ir_version = it->second.as<int64_t>();

        if (ir_version == 10 && newAPI) {
            std::unordered_map<std::string, std::shared_ptr<ov::descriptor::Tensor>> leaf_names;
            const auto inputs = function->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto ngraph_type = inputs[i].get_element_type();
                const auto legacy_type = details::toLegacyType(ngraph_type, true);
                prepost.input(i).tensor().set_element_type(legacy_type);
                for (const auto& name : inputs[i].get_names()) {
                    OPENVINO_ASSERT(leaf_names.find(name) == leaf_names.end(),
                                    "Model tensor names have collisions.",
                                    " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.emplace(name, inputs[i].get_tensor_ptr());
                }
            }

            const auto outputs = function->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto ngraph_type = outputs[i].get_element_type();
                const auto legacy_type = details::toLegacyType(ngraph_type, false);

                prepost.output(i).tensor().set_element_type(legacy_type);
                for (const auto& name : outputs[i].get_names()) {
                    auto tensor_it = leaf_names.find(name);
                    OPENVINO_ASSERT(tensor_it == leaf_names.end() || tensor_it->second == outputs[i].get_tensor_ptr(),
                                    "Model tensor names have collisions.",
                                    " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.emplace(name, outputs[i].get_tensor_ptr());
                }
            }

            // in order to support the following scenarios for IR v10 cases:
            // ov::Model f = ie.read_model(..);
            // f.input("input_operation_name");
            // f.output("output_operation_name");
            // f.add_output("operation_name[].port_index]");
            // f.reshape({ { "input_operation_name", ov::PartialShape{} } });
            // we need to add operation names as tensor names for inputs and outputs
            {
                for (const auto& result : function->get_results()) {
                    auto res_name = ngraph::op::util::create_ie_output_name(result->input_value(0));
                    OPENVINO_ASSERT(
                        leaf_names.find(res_name) == leaf_names.end() ||
                            result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                        "Model operation names have collisions with tensor names.",
                        " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.emplace(res_name, nullptr);
                    result->output(0).get_tensor().add_names({res_name});
                }
                for (const auto& param : function->get_parameters()) {
                    auto param_name = param->get_friendly_name();
                    OPENVINO_ASSERT(
                        leaf_names.find(param_name) == leaf_names.end() ||
                            param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                        "Model operation names have collisions with tensor names.",
                        " Please use MO to generate new IR version, it should allow to avoid the issue");
                    leaf_names.emplace(param_name, nullptr);
                    param->output(0).get_tensor().add_names({param_name});
                }
            }

            function = prepost.build();

            // Set version to 10
            rt_info["version"] = int64_t(10);
        } else if (ir_version == 11 && !newAPI) {
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
            rt_info["version"] = int64_t(10);

            function = prepost.build();
        }
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(function, exts, newAPI));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::vector<ov::Extension::Ptr> wrap_old_extensions(const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    std::vector<ov::Extension::Ptr> extensions;
    for (const auto& ext : exts) {
        for (const auto& item : ext->getOpSets()) {
            for (const auto& type_info : item.second.get_types_info()) {
                extensions.emplace_back(std::make_shared<ov::ExtensionWrapper>(ext, item.first, type_info.name));
            }
        }
    }
    return extensions;
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath,
                                const std::string& binPath,
                                const std::vector<IExtensionPtr>& exts,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool newAPI) {
#ifdef ENABLE_IR_V7_READER
    // IR v7 obsolete code
    {
        // Register readers if it is needed
        registerReaders();
        auto cnnnetwork = load_ir_v7_network(modelPath, binPath, exts);

        OPENVINO_SUPPRESS_DEPRECATED_START
        if (static_cast<ICNNNetwork::Ptr>(cnnnetwork) != nullptr) {
            OPENVINO_ASSERT(!newAPI, "Cannot read IR v7 from OpenVINO 2.0 API");
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
    auto frontEndManager = ov::frontend::get_frontend_manager();
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

    FE = frontEndManager->load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        if (!exts.empty())
            FE->add_extension(wrap_old_extensions(exts));
        inputModel = FE->load(params);
    }

    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, newAPI);
    }

    const auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    std::string FEs;
    for (const auto& fe_name : frontEndManager->get_available_front_ends())
        FEs += fe_name + " ";
    IE_THROW(NetworkNotRead) << "Unable to read the model: " << modelPath
                             << " Please check that model format: " << fileExt
                             << " is supported and the model is correct."
                             << " Available frontends: " << FEs;
}

CNNNetwork details::ReadNetwork(const std::string& model,
                                const Blob::CPtr& weights,
                                const std::vector<IExtensionPtr>& exts,
                                const std::vector<ov::Extension::Ptr>& ov_exts,
                                bool newAPI) {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;

#ifdef ENABLE_IR_V7_READER
    // IR v7 obsolete code
    {
        // Register readers if it is needed
        registerReaders();
        assertIfIRv7LikeModel(modelStream);

        for (auto it = readers.begin(); it != readers.end(); it++) {
            auto reader = it->second;
            if (reader->supportModel(modelStream)) {
                OPENVINO_ASSERT(!newAPI, "Cannot read IR v7 from OpenVINO 2.0 API");
                if (weights)
                    return reader->read(modelStream, weights, exts);
                return reader->read(modelStream, exts);
            }
        }
    }
#endif  // ENABLE_IR_V7_READER

    // Try to load with FrontEndManager
    auto frontEndManager = ov::frontend::get_frontend_manager();
    ov::frontend::FrontEnd::Ptr FE;
    ov::frontend::InputModel::Ptr inputModel;

    ov::AnyVector params{&modelStream};
    if (weights) {
        char* data = weights->cbuffer().as<char*>();
        std::shared_ptr<ngraph::runtime::AlignedBuffer> weights_buffer =
            std::make_shared<ngraph::runtime::SharedBuffer<Blob::CPtr>>(data, weights->byteSize(), weights);
        params.emplace_back(weights_buffer);
    }

    FE = frontEndManager->load_by_model(params);
    if (FE) {
        FE->add_extension(ov_exts);
        if (!exts.empty())
            FE->add_extension(wrap_old_extensions(exts));
        inputModel = FE->load(params);
    }
    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, newAPI);
    }

    IE_THROW(NetworkNotRead)
        << "Unable to read the model. Please check if the model format is supported and model is correct.";
}

}  // namespace InferenceEngine
