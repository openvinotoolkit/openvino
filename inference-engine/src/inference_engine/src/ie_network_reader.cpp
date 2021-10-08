// Copyright (C) 2018-2021 Intel Corporation
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
#include "details/ie_so_pointer.hpp"
#include "file_utils.h"
#include "frontend_manager/frontend_manager.hpp"
#include "ie_api.h"
#include "ie_common.h"
#include "ie_icnn_network.hpp"
#include "ie_input_info.hpp"
#include "ie_ir_version.hpp"
#include "ie_itt.hpp"
#include "ie_reader.hpp"
#include "ngraph/function.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/input_network_info.hpp"
#include "openvino/core/preprocess/input_tensor_info.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/rt_info/old_api_map_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

namespace details {

/**
 * @brief This class defines the name of the fabric for creating an IReader object in DLL
 */
template <>
class SOCreatorTrait<IReader> {
public:
    /**
     * @brief A name of the fabric for creating IReader object in DLL
     */
    static constexpr auto name = "CreateReader";
};

}  // namespace details

/**
 * @brief This class is a wrapper for reader interfaces
 */
class Reader : public IReader {
    InferenceEngine::details::SOPointer<IReader> ptr;
    std::once_flag readFlag;
    std::string name;
    std::string location;

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() {
        std::call_once(readFlag, [&]() {
            ov::util::FilePath libraryName = ov::util::to_file_path(location);
            ov::util::FilePath readersLibraryPath =
                FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                IE_THROW() << "Please, make sure that Inference Engine ONNX reader library "
                           << ov::util::from_file_path(::FileUtils::makePluginLibraryName({}, libraryName)) << " is in "
                           << getIELibraryPath();
            }
            ptr = {readersLibraryPath};
        });

        return ptr;
    }

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() const {
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

static ngraph::frontend::FrontEndManager& get_frontend_manager() {
    static ngraph::frontend::FrontEndManager manager;
    return manager;
}

void registerReaders() {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "registerReaders");
    static bool initialized = false;
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    if (initialized)
        return;

    auto create_if_exists = [](const std::string name, const std::string library_name) {
        ov::util::FilePath libraryName = ov::util::to_file_path(library_name);
        ov::util::FilePath readersLibraryPath =
            FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

        if (!FileUtils::fileExist(readersLibraryPath))
            return std::shared_ptr<Reader>();
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

ov::Extensions get_extensions_map(const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    ov::Extensions extensions;
    for (const auto& ext : exts) {
        for (const auto& item : ext->getOpSets()) {
            if (extensions.count(item.first)) {
                IE_THROW() << "Extension with " << item.first << " name already exists";
            }
            extensions[item.first] = item.second;
        }
    }
    return extensions;
}

CNNNetwork load_ir_v7_network(const std::string& modelPath,
                              const std::string& binPath,
                              const std::vector<IExtensionPtr>& exts) {
    // Fix unicode name
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = ov::util::string_to_wstring(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif

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
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                std::wstring weights_path = ov::util::string_to_wstring(bPath.c_str());
#else
                std::string weights_path = bPath;
#endif
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

CNNNetwork convert_to_cnnnetwork(std::shared_ptr<ngraph::Function>& function,
                                 const std::vector<IExtensionPtr>& exts,
                                 bool newAPI) {
    auto& rt_info = function->get_rt_info();
    const auto it = rt_info.find("version");
    const bool is_ir = it != rt_info.end();

    // only for IR cases we need preprocessing or postprocessing steps
    if (is_ir) {
        using namespace ov::preprocess;
        PrePostProcessor prepost;

        auto iv_version_impl = std::dynamic_pointer_cast<ngraph::VariantImpl<int64_t>>(it->second);
        OPENVINO_ASSERT(iv_version_impl != nullptr, "Failed to extract IR version from 'version' attribute");
        const int64_t ir_version = iv_version_impl->get();

        if (ir_version == 10 && newAPI) {
            const auto inputs = function->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto ngraph_type = inputs[i].get_element_type();
                const auto legacy_type = details::toLegacyType(ngraph_type, true);
                prepost.input(ov::preprocess::InputInfo(i)
                                  .tensor(InputTensorInfo().set_element_type(legacy_type))
                                  .preprocess(PreProcessSteps()
                                                  // TODO: remove explicit type
                                                  .convert_element_type(ngraph_type)));
            }

            std::vector<std::string> result_names;
            std::vector<ov::Output<ov::Node>> prevPorts;
            result_names.reserve(function->get_results().size());
            prevPorts.reserve(function->get_results().size());
            for (const auto& result : function->get_results()) {
                result_names.emplace_back(ngraph::op::util::create_ie_output_name(result->input_value(0)));
                prevPorts.emplace_back(result->input_value(0));
            }
            const auto outputs = function->outputs();
            for (size_t i = 0; i < outputs.size(); ++i) {
                const auto ngraph_type = outputs[i].get_element_type();
                const auto legacy_type = details::toLegacyType(ngraph_type, false);

                prepost.output(OutputInfo(i)
                                   .postprocess(PostProcessSteps().convert_element_type())
                                   .tensor(OutputTensorInfo().set_element_type(legacy_type)));
            }

            function = prepost.build(function);
            // Change tensor names for inputs/outputs for cases with old IR
            for (const auto& param : function->get_parameters()) {
                param->output(0).get_tensor().add_names({param->get_friendly_name()});
            }
            OPENVINO_ASSERT(function->get_results().size() == result_names.size());
            for (size_t i = 0; i < function->get_results().size(); i++) {
                const auto& result = function->get_results()[i];
                result->output(0).get_tensor().add_names({result_names[i]});
                // FIXME: WA to fix CNNNetwork output name
                if (prevPorts[i].get_node() != result->input_value(0).get_node()) {
                    result->input_value(0).get_node()->set_friendly_name(prevPorts[i].get_node()->get_friendly_name());
                    prevPorts[i].get_node()->set_friendly_name("op_original_" +
                                                               prevPorts[i].get_node()->get_friendly_name());
                }
            }
        } else if (ir_version == 11 && !newAPI) {
            const std::string& old_api_map_key = ov::OldApiMap::get_type_info_static();

            auto& parameters = function->get_parameters();
            for (size_t i = 0; i < parameters.size(); ++i) {
                const auto& parameter = parameters[i];
                ov::RTMap& rtInfo = parameter->get_rt_info();
                const auto it = rtInfo.find(old_api_map_key);
                if (it == rtInfo.end())
                    continue;

                const auto old_api_map_attr = std::dynamic_pointer_cast<ov::OldApiMap>(it->second);
                OPENVINO_ASSERT(old_api_map_attr != nullptr, "Failed to cast to ov::OldApiMap");
                const auto old_api_map_attr_val = old_api_map_attr->get();
                auto old_api_type = old_api_map_attr_val.get_type();
                const auto old_api_transpose_args = old_api_map_attr_val.get_order();

                OPENVINO_ASSERT(!old_api_type.is_dynamic(), "Old API map does not support dynamic type");
                // if no differences between IR v10 and IR v11, add identity convert which will be optimized out
                if (old_api_type == ov::element::undefined)
                    old_api_type = parameter->get_element_type();

                std::stringstream tensorLayout, networkLayout;
                for (size_t i = 0; i < old_api_transpose_args.size(); ++i) {
                    tensorLayout << i;
                    networkLayout << old_api_transpose_args[i];
                }

                PreProcessSteps steps;
                // TODO: remove explicit type
                steps.convert_element_type(parameter->get_element_type());
                // TODO: move steps directly to builder once we allow Layout() -> Layout transpose
                if (!old_api_transpose_args.empty())
                    steps.convert_layout();

                prepost.input(
                    ov::preprocess::InputInfo(i)
                        .tensor(
                            InputTensorInfo().set_element_type(old_api_type).set_layout(ov::Layout(tensorLayout.str())))
                        .preprocess(std::move(steps))
                        .network(InputNetworkInfo().set_layout(ov::Layout(networkLayout.str()))));

                // remove old api once we applied it
                rtInfo.erase(it);
            }

            auto& resuls = function->get_results();
            for (size_t i = 0; i < resuls.size(); ++i) {
                const auto& result = resuls[i];
                ov::RTMap& rtInfo = result->get_rt_info();
                const auto it = rtInfo.find(old_api_map_key);
                if (it == rtInfo.end())
                    continue;

                const auto old_api_map_attr = std::dynamic_pointer_cast<ov::OldApiMap>(it->second);
                OPENVINO_ASSERT(old_api_map_attr != nullptr, "Failed to cast to ov::OldApiMap");
                const auto old_api_map_attr_val = old_api_map_attr->get();
                auto old_api_type = old_api_map_attr_val.get_type();
                const auto old_api_transpose_args = old_api_map_attr_val.get_order();

                OPENVINO_ASSERT(!old_api_type.is_dynamic(), "Old API map does not support dynamic type");
                // if no differences between IR v10 and IR v11, add identity convert which will be optimized out
                if (old_api_type == ov::element::undefined)
                    old_api_type = result->get_element_type();

                std::stringstream tensorLayout, networkLayout;
                for (size_t i = 0; i < old_api_transpose_args.size(); ++i) {
                    networkLayout << i;
                    tensorLayout << old_api_transpose_args[i];
                }

                prepost.output(OutputInfo(i)
                                   .network(OutputNetworkInfo().set_layout(ov::Layout(networkLayout.str())))
                                   .postprocess(PostProcessSteps().convert_layout().convert_element_type())
                                   .tensor(OutputTensorInfo()
                                               .set_element_type(old_api_type)
                                               .set_layout(ov::Layout(tensorLayout.str()))));

                // remove old api once we applied it
                rtInfo.erase(it);
            }

            function = prepost.build(function);

            // TODO: keep information about layout once we have an ability to
            // apply permutation to layout

            // restore layout information
            for (const auto& parameter : function->get_parameters()) {
                parameter->set_layout({});
            }
            for (const auto& result : function->get_results()) {
                result->set_layout({});
            }
        }
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    return CNNNetwork(std::make_shared<details::CNNNetworkNGraphImpl>(function, exts, newAPI));
    OPENVINO_SUPPRESS_DEPRECATED_END
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath,
                                const std::string& binPath,
                                const std::vector<IExtensionPtr>& exts,
                                bool newAPI) {
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
        IE_SUPPRESS_DEPRECATED_END
    }

    // Fix unicode name
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = ov::util::string_to_wstring(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif

    // Try to load with FrontEndManager
    auto& manager = get_frontend_manager();
    ngraph::frontend::FrontEnd::Ptr FE;
    ngraph::frontend::InputModel::Ptr inputModel;

    ov::VariantVector params{ov::make_variant(model_path)};
    if (!exts.empty()) {
        params.emplace_back(ov::make_variant(get_extensions_map(exts)));
    }

    if (!binPath.empty()) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::wstring& weights_path = ov::util::string_to_wstring(binPath.c_str());
#else
        const std::string& weights_path = binPath;
#endif
        params.emplace_back(ov::make_variant(weights_path));
    }

    FE = manager.load_by_model(params);
    if (FE)
        inputModel = FE->load(params);

    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, newAPI);
    }

    const auto fileExt = modelPath.substr(modelPath.find_last_of(".") + 1);
    IE_THROW(NetworkNotRead) << "Unable to read the model: " << modelPath
                             << " Please check that model format: " << fileExt
                             << " is supported and the model is correct.";
}

CNNNetwork details::ReadNetwork(const std::string& model,
                                const Blob::CPtr& weights,
                                const std::vector<IExtensionPtr>& exts,
                                bool newAPI) {
    std::istringstream modelStringStream(model);
    std::istream& modelStream = modelStringStream;

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

    // Try to load with FrontEndManager
    auto& manager = get_frontend_manager();
    ngraph::frontend::FrontEnd::Ptr FE;
    ngraph::frontend::InputModel::Ptr inputModel;

    ov::VariantVector params{ov::make_variant(&modelStream)};
    if (weights) {
        char* data = weights->cbuffer().as<char*>();
        ov::Weights weights_buffer =
            std::make_shared<ngraph::runtime::SharedBuffer<Blob::CPtr>>(data, weights->byteSize(), weights);
        params.emplace_back(ov::make_variant(weights_buffer));
    }
    if (!exts.empty()) {
        params.emplace_back(ov::make_variant(get_extensions_map(exts)));
    }

    FE = manager.load_by_model(params);
    if (FE)
        inputModel = FE->load(params);
    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return convert_to_cnnnetwork(ngFunc, exts, newAPI);
    }

    IE_THROW(NetworkNotRead)
        << "Unable to read the model. Please check if the model format is supported and model is correct.";
}

}  // namespace InferenceEngine
