// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <samples/common.hpp>

#include <inference_engine.hpp>
#include <ie_allocator.hpp>
#include <mutex>

#include <details/os/os_filesystem.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define WEIGHTS_EXT L".bin"
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define WEIGHTS_EXT ".bin"
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif

double roundOff(double n) {
    double d = n * 100.0f;
    int i = d + 0.5f;
    d = i / 100.0f;
    return d;
}

std::string convertByteSize(long long size) {
    static std::vector<std::string> SIZES = { "B", "KB", "MB", "GB", "TB", "PB" };

    int div = 0;
    size_t rem = 0;

    while (size >= 1024 && div < (SIZES.size() - 1)) {
        rem = (size % 1024);
        div++;
        size /= 1024;
    }

    double size_d = size + rem / 1024.0f;

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << roundOff(size_d);
    std::string result = stream.str() + " " + SIZES[div];

    return result;
}

class SystemAllocator : public IAllocator {
public:
    using Ptr = std::shared_ptr<SystemAllocator>;

    void Release() noexcept override {
        delete this;
    }

    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* a) noexcept override {}

    void* alloc(size_t size) noexcept override {
        auto _malloc = [](size_t size, int alignment) {
            void *ptr;
#ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
            int rc = ((ptr)? 0 : errno);
#else
            int rc = ::posix_memalign(&ptr, alignment, size);
#endif /* _WIN32 */
            return (rc == 0) ? reinterpret_cast<char*>(ptr) : nullptr;
        };

        void* ptr = _malloc(size, 4096);

        std::unique_lock<std::mutex> lock(_mutex);
        _size_map[ptr] = size;
        _allocated += size;

        return ptr;
    }

    void free(void* ptr) noexcept override {
        try {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            ::free(ptr);
#endif /* _WIN32 */
        } catch (...) {
        }
        std::unique_lock<std::mutex> lock(_mutex);
        _allocated -= _size_map[ptr];
        _size_map.erase(_size_map.find(ptr));

        return true;
    }

    size_t allocated() const {
        return _allocated;
    }

private:
    size_t _allocated = 0;
    std::map<void*, size_t> _size_map;
    std::mutex _mutex;
};

void printAllocatedMemory(const std::string& msg, const SystemAllocator::Ptr& allocator) {
    std::cout << msg << convertByteSize(allocator->allocated()) << std::endl << std::endl;
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(
        input_image_path.c_str(),
        std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(
                std::istream_iterator<char>(input_image_stream),
                std::istream_iterator<char>(),
                std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}
#endif

int inference(const std::string& model_path, const std::string& image_path, const std::string& device, SystemAllocator::Ptr& allocator) {
    try {
        const file_name_t input_model{model_path};
        const file_name_t input_image_path{image_path};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = InferenceEngine::details::wStringtoMBCSstringChar(device.c_str());
#else
        const std::string device_name{device};
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine instance -------------------------------------
        std::cout << "Initializing IE Core..." << std::endl;

        Core ie;
        SetSystemAllocator(allocator.get());

        printAllocatedMemory("Total memory allocated: ", allocator);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        std::cout << "Loading network..." << std::endl;

        CNNNetwork network = ie.ReadNetwork(input_model, input_model.substr(0, input_model.size() - 4) + WEIGHTS_EXT);
        network.setBatchSize(1);

        printAllocatedMemory("Total memory allocated: ", allocator);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an infer request.
         * Resize and layout conversions are executed automatically during inference */
        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_info->setLayout(Layout::NHWC);
        input_info->setPrecision(Precision::U8);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        std::cout << "Loading network to plugin..." << std::endl;

        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);

        printAllocatedMemory("Total memory allocated: ", allocator);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        std::cout << "Creating inference request..." << std::endl;

        InferRequest infer_request = executable_network.CreateInferRequest();

        printAllocatedMemory("Total memory allocated: ", allocator);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /* Read input image to a blob and set it to an infer request without resize and layout conversions. */
        cv::Mat image = imread_t(input_image_path);
        Blob::Ptr imgBlob = wrapMat2Blob(image);  // just wrap Mat data by Blob::Ptr without allocating of new memory
        infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        std::cout << "Executing inference request..." << std::endl;

        infer_request.Infer();

        printAllocatedMemory("Total memory allocated: ", allocator);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        // Print classification results
        ClassificationResult_t classificationResult(output, {input_image_path});
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool" << std::endl;
    return EXIT_SUCCESS;
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
int wmain(int argc, wchar_t *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (argc != 4) {
        tcout << "Usage : ./hello_classification <path_to_model> <path_to_image> <device_name>" << std::endl;
        return EXIT_FAILURE;
    }

    std::shared_ptr<SystemAllocator> allocator = shared_from_irelease(new SystemAllocator());

    printAllocatedMemory("Memory allocated  before running inference: ", allocator);

    int result = inference(argv[1], argv[2], argv[3], allocator);

    printAllocatedMemory("Memory allocated  after freeing up all resources: ", allocator);

    return result;
}
