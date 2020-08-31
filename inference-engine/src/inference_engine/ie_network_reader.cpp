// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network_reader.hpp"
#include "ie_itt.hpp"

#include <details/ie_so_pointer.hpp>
#include <file_utils.h>
#include <ie_blob_stream.hpp>
#include <ie_reader.hpp>
#include <ie_ir_version.hpp>

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
            FileUtils::FilePath readersLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

            if (!FileUtils::fileExist(readersLibraryPath)) {
                THROW_IE_EXCEPTION << "Please, make sure that Inference Engine ONNX reader library "
                    << FileUtils::fromFilePath(::FileUtils::makeSharedLibraryName({}, libraryName)) << " is in "
                    << getIELibraryPath();
            }
            ptr = InferenceEngine::details::SOPointer<IReader>(readersLibraryPath);
        });

        return ptr;
    }

    InferenceEngine::details::SOPointer<IReader> getReaderPtr() const {
        return const_cast<Reader*>(this)->getReaderPtr();
    }

    void Release() noexcept override {
        delete this;
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
    CNNNetwork read(std::istream& model, std::istream& weights, const std::vector<IExtensionPtr>& exts) const override {
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
        FileUtils::FilePath readersLibraryPath = FileUtils::makeSharedLibraryName(getInferenceEngineLibraryPath(), libraryName);

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

    THROW_IE_EXCEPTION << "The support of IR v" << irVersion <<  " has been removed from the product. "
        "Please, convert the original model using the Model Optimizer which comes with this "
        "version of the OpenVINO to generate supported IR version.";
}

}  // namespace

CNNNetwork details::ReadNetwork(const std::string& modelPath, const std::string& binPath, const std::vector<IExtensionPtr>& exts) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "details::ReadNetwork");
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
    if (!modelStream.is_open())
        THROW_IE_EXCEPTION << "Model file " << modelPath << " cannot be opened!";

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
                    THROW_IE_EXCEPTION << "Weights file " << bPath << " cannot be opened!";

                // read model with weights
                auto network = reader->read(modelStream, binStream, exts);
                modelStream.close();
                return network;
            }
            // read model without weights
            return reader->read(modelStream, exts);
        }
    }
    THROW_IE_EXCEPTION << "Unknown model format! Cannot find reader for model format: " << fileExt << " and read the model: " << modelPath <<
        ". Please check that reader library exists in your PATH.";
}

CNNNetwork details::ReadNetwork(const std::string& model, const Blob::CPtr& weights, const std::vector<IExtensionPtr>& exts) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "details::ReadNetwork");
    // Register readers if it is needed
    registerReaders();
    std::istringstream modelStream(model);
    details::BlobStream binStream(weights);

    assertIfIRv7LikeModel(modelStream);

    for (auto it = readers.begin(); it != readers.end(); it++) {
        auto reader = it->second;
        if (reader->supportModel(modelStream)) {
            if (weights)
                return reader->read(modelStream, binStream, exts);
            return reader->read(modelStream, exts);
        }
    }
    THROW_IE_EXCEPTION << "Unknown model format! Cannot find reader for the model and read it. Please check that reader library exists in your PATH.";
}

}  // namespace InferenceEngine
