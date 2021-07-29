// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network_reader.hpp"
#include "ie_itt.hpp"

#include <details/ie_so_pointer.hpp>
#include <file_utils.h>
#include <ie_reader.hpp>
#include <ie_ir_version.hpp>
#include <frontend_manager/frontend_manager.hpp>

#include <fstream>
#include <istream>
#include <mutex>
#include <map>

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
class Reader: public IReader {
    InferenceEngine::details::SOPointer<IReader> ptr;
    std::once_flag readFlag;
    std::string name;
    std::string location;

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() {
        std::call_once(readFlag, [&] () {
            FileUtils::FilePath libraryName = FileUtils::toFilePath(location);
            FileUtils::FilePath readersLibraryPath = FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                IE_THROW() << "Please, make sure that Inference Engine ONNX reader library "
                           << FileUtils::fromFilePath(::FileUtils::makePluginLibraryName({}, libraryName)) << " is in "
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
    Reader(const std::string& name, const std::string location): name(name), location(location) {}
    bool supportModel(std::istream& model) const override {
        OV_ITT_SCOPED_TASK(itt::domains::IE, "Reader::supportModel");
        auto reader = getReaderPtr();
        return reader->supportModel(model);
    }
    CNNNetwork read(std::istream& model, const std::vector<IExtensionPtr>& exts) const override {
        auto reader = getReaderPtr();
        return reader->read(model, exts);
    }
    CNNNetwork read(std::istream& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) const override {
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
    OV_ITT_SCOPED_TASK(itt::domains::IE, "registerReaders");
    static bool initialized = false;
    static std::mutex readerMutex;
    std::lock_guard<std::mutex> lock(readerMutex);
    if (initialized) return;

    // TODO: Read readers info from XML
    auto create_if_exists = [] (const std::string name, const std::string library_name) {
        FileUtils::FilePath libraryName = FileUtils::toFilePath(library_name);
        FileUtils::FilePath readersLibraryPath = FileUtils::makePluginLibraryName(getInferenceEngineLibraryPath(), libraryName);

        if (!FileUtils::fileExist(readersLibraryPath))
            return std::shared_ptr<Reader>();
        return std::make_shared<Reader>(name, library_name);
    };

    // try to load ONNX reader if library exists
    auto onnxReader = create_if_exists("ONNX", std::string("inference_engine_onnx_reader") + std::string(IE_BUILD_POSTFIX));
    if (onnxReader) {
        readers.emplace("onnx", onnxReader);
        readers.emplace("prototxt", onnxReader);
    }

    // try to load IR reader v10 if library exists
    auto irReaderv10 = create_if_exists("IRv10", std::string("inference_engine_ir_reader") + std::string(IE_BUILD_POSTFIX));
    if (irReaderv10)
        readers.emplace("xml", irReaderv10);

    // try to load IR reader v7 if library exists
    auto irReaderv7 = create_if_exists("IRv7", std::string("inference_engine_ir_v7_reader") + std::string(IE_BUILD_POSTFIX));
    if (irReaderv7)
        readers.emplace("xml", irReaderv7);

    initialized = true;
}

void assertIfIRv7LikeModel(std::istream & modelStream) {
    auto irVersion = details::GetIRVersion(modelStream);
    bool isIRv7 = irVersion > 1 && irVersion <= 7;

    if (!isIRv7)
        return;

    for (auto && kvp : readers) {
        Reader::Ptr reader = kvp.second;
        if (reader->getName() == "IRv7") {
            return;
        }
    }

    IE_THROW() << "The support of IR v" << irVersion <<  " has been removed from the product. "
                                                         "Please, convert the original model using the Model Optimizer which comes with this "
                                                         "version of the OpenVINO to generate supported IR version.";
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath, const std::string& binPath, const std::vector<IExtensionPtr>& exts) {
    // Register readers if it is needed
    registerReaders();

    // Fix unicode name
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring model_path = FileUtils::multiByteCharToWString(modelPath.c_str());
#else
    std::string model_path = modelPath;
#endif
    // Try to open model file
    std::ifstream modelStream(model_path, std::ios::binary);
    // save path in extensible array of stream
    // notice: lifetime of path pointed by pword(0) is limited by current scope
    const std::string path_to_save_in_stream = modelPath;
    modelStream.pword(0) = const_cast<char*>(path_to_save_in_stream.c_str());
    if (!modelStream.is_open())
        IE_THROW() << "Model file " << modelPath << " cannot be opened!";

    assertIfIRv7LikeModel(modelStream);

    // Find reader for model extension
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
                if (pos != std::string::npos) pathWoExt = modelPath.substr(0, pos);
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
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                std::wstring weights_path = FileUtils::multiByteCharToWString(bPath.c_str());
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

                Blob::Ptr weights = make_shared_blob<uint8_t>({Precision::U8, { fileSize }, C });

                {
                    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::IE_RT, "ReadNetworkWeights");
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
    // Try to load with FrontEndManager
    static ngraph::frontend::FrontEndManager manager;
    ngraph::frontend::FrontEnd::Ptr FE;
    ngraph::frontend::InputModel::Ptr inputModel;
    if (!binPath.empty()) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        std::wstring weights_path = FileUtils::multiByteCharToWString(binPath.c_str());
#else
        std::string weights_path = binPath;
#endif
        FE = manager.load_by_model(model_path, weights_path);
        if (FE) inputModel = FE->load(model_path, weights_path);
    } else {
        FE = manager.load_by_model(model_path);
        if (FE) inputModel = FE->load(model_path);
    }
    if (inputModel) {
        auto ngFunc = FE->convert(inputModel);
        return CNNNetwork(ngFunc);
    }
    IE_THROW() << "Unknown model format! Cannot find reader for model format: " << fileExt << " and read the model: " << modelPath <<
               ". Please check that reader library exists in your PATH.";
}

CNNNetwork details::ReadNetwork(const std::string& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) {
    // Register readers if it is needed
    registerReaders();
    std::istringstream modelStream(model);

    assertIfIRv7LikeModel(modelStream);

    for (auto it = readers.begin(); it != readers.end(); it++) {
        auto reader = it->second;
        if (reader->supportModel(modelStream)) {
            if (weights)
                return reader->read(modelStream, weights, exts);
            return reader->read(modelStream, exts);
        }
    }
    IE_THROW() << "Unknown model format! Cannot find reader for the model and read it. Please check that reader library exists in your PATH.";
}

}  // namespace InferenceEngine
