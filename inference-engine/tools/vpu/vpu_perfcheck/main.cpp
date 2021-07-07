// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)
#define NOMINMAX
#endif
#if (defined(_WIN32) || defined(_WIN64))
#define WIN32_LEAN_AND_MEAN
#else
#include <pthread.h>
#endif

#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <map>
#include <algorithm>
#include <utility>
#include <iomanip>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <ios>
#include <sys/stat.h>

#include <samples/os/windows/w_dirent.h>

#include <inference_engine.hpp>
#include <precision_utils.h>
#include <samples/common.hpp>

#include <vpu/vpu_config.hpp>

static char* m_exename = nullptr;

#if defined(_WIN32) || defined(__APPLE__) || defined(ANDROID)
typedef std::chrono::time_point<std::chrono::steady_clock> time_point;
#else
typedef std::chrono::time_point<std::chrono::system_clock> time_point;
#endif
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

#define TIMEDIFF(start, end) ((std::chrono::duration_cast<ms>((end) - (start))).count())

class BitMap {
private:
    struct BmpHeader {
        unsigned short type   = 0u;               /* Magic identifier            */
        unsigned int size     = 0u;               /* File size in bytes          */
        unsigned int reserved = 0u;
        unsigned int offset   = 0u;               /* Offset to image data, bytes */
    };

    struct BmpInfoHeader {
        unsigned int size = 0u;                   /* Header size in bytes      */
        int width = 0, height = 0;                /* Width and height of image */
        unsigned short planes = 0u;               /* Number of colour planes   */
        unsigned short bits = 0u;                 /* Bits per pixel            */
        unsigned int compression = 0u;            /* Compression type          */
        unsigned int imagesize = 0u;              /* Image size in bytes       */
        int xresolution = 0, yresolution = 0;     /* Pixels per meter          */
        unsigned int ncolours = 0u;               /* Number of colours         */
        unsigned int importantcolours = 0u;       /* Important colours         */
    };

public:
    explicit BitMap(const std::string &filename) {
        BmpHeader header;
        BmpInfoHeader infoHeader;

        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            return;
        }

        input.read(reinterpret_cast<char *>(&header.type), 2);

        if (header.type != 'M'*256+'B') {
            std::cerr << "[BMP] file is not bmp type\n";
            return;
        }

        input.read(reinterpret_cast<char *>(&header.size), 4);
        input.read(reinterpret_cast<char *>(&header.reserved), 4);
        input.read(reinterpret_cast<char *>(&header.offset), 4);

        input.read(reinterpret_cast<char *>(&infoHeader), sizeof(BmpInfoHeader));

        bool rowsReversed = infoHeader.height < 0;
        _width = infoHeader.width;
        _height = abs(infoHeader.height);

        if (infoHeader.bits != 24) {
            std::cerr << "[BMP] 24bpp only supported. But input has:" << infoHeader.bits << "\n";
            return;
        }

        if (infoHeader.compression != 0) {
            std::cerr << "[BMP] compression not supported\n";
        }

        int padSize = _width & 3;
        char pad[3];
        size_t size = _width * _height * 3;

        _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

        input.seekg(header.offset, std::ios::beg);

        // reading by rows in invert vertically
        for (uint32_t i = 0; i < _height; i++) {
            uint32_t storeAt = rowsReversed ? i : (uint32_t)_height - 1 - i;
            input.read(reinterpret_cast<char *>(_data.get()) + _width * 3 * storeAt, _width * 3);
            input.read(pad, padSize);
        }
    }

    ~BitMap() = default;

    size_t _height = 0;
    size_t _width = 0;
    std::shared_ptr<unsigned char> _data;

public:
    size_t size() const { return _width * _height * 3; }
    size_t width() const { return _width; }
    size_t height() const { return _height; }

    std::shared_ptr<unsigned char> getData() {
        return _data;
    }
};


static bool loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob);
static bool loadVideo(const std::vector<std::string> &imagesFolder, InferenceEngine::Blob::Ptr &blob);
static bool loadBinaryTensor(const std::string &binaryFilename, InferenceEngine::Blob::Ptr &blob);


static void setConfig(std::map<std::string, std::string>& config,
                      const std::string& file_config_cl) {
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_WARNING);
    config[InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME] = CONFIG_VALUE(YES);
    config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = file_config_cl;
}

static void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) {
    std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfVec(perfMap.begin(),
                                                                                             perfMap.end());
    std::sort(perfVec.begin(), perfVec.end(),
              [=](const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair1,
                  const std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo> &pair2) -> bool {
                  return pair1.second.execution_index < pair2.second.execution_index;
              });

    size_t maxLayerName = 0u, maxExecType = 0u;
    for (auto & it : perfVec) {
        maxLayerName = std::max(maxLayerName, it.first.length());
        maxExecType = std::max(maxExecType, std::strlen(it.second.exec_type));
    }

    size_t indexWidth = 7, nameWidth = maxLayerName + 5, typeWidth = maxExecType + 5, timeWidth = 10;
    size_t totalWidth = indexWidth + nameWidth + typeWidth + timeWidth;

    std::cout << std::endl << "Detailed Per Stage Profile" << std::endl;
    for (size_t i = 0; i < totalWidth; i++)
        std::cout << "=";
    std::cout << std::endl;
    std::cout << std::setw(indexWidth) << std::left << "Index"
              << std::setw(nameWidth) << std::left << "Name"
              << std::setw(typeWidth) << std::left << "Type"
              << std::setw(timeWidth) << std::right << "Time (ms)"
              << std::endl;
    for (size_t i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;

    long long totalTime = 0;
    for (const auto& p : perfVec) {
        const auto& stageName = p.first;
        const auto& info = p.second;
        if (info.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            std::cout << std::setw(indexWidth) << std::left << info.execution_index
                      << std::setw(nameWidth) << std::left << stageName
                      << std::setw(typeWidth) << std::left << info.exec_type
                      << std::setw(timeWidth) << std::right << info.realTime_uSec / 1000.0
                      << std::endl;

            totalTime += info.realTime_uSec;
        }
    }

    for (int i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(totalWidth / 2) << std::right << "Total inference time:"
              << std::setw(totalWidth / 2 + 1) << std::right << totalTime / 1000.0
              << std::endl;
    for (int i = 0; i < totalWidth; i++)
        std::cout << "-";
    std::cout << std::endl;
}

static std::string getAppRealName(const char* name) {
    std::string filename(name);
    size_t splitpos = filename.find_last_of('\\');
    if (std::string::npos == splitpos) {
        splitpos = filename.find_last_of('/');
        if (std::string::npos == splitpos) {
            return filename;
        }
    }
    return filename.substr(splitpos + 1);
}

static void print_usage() {
    std::cout << "Usage:" << std::endl << getAppRealName(m_exename) << " <model_path> <img_dir_path> [number of iterations >= 1000]"
              << " [batch >= 1, default=1] [num_networks, default=1] [config_file_custom_layer, default='']" << std::endl;
}

static void getBMPFiles(std::vector<std::string> &out, const std::string &directory) {
    const std::string ext = ".bmp";
    DIR *dir;
    dirent *ent;
    dir = opendir(directory.c_str());
    if (!dir)
        return;
    while ((ent = readdir(dir)) != nullptr) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;
        if ((file_name.length() >= ext.length())
            && (0 == file_name.compare(file_name.length() - ext.length(), ext.length(), ext))) {
            // proceed
        } else {
            continue;
        }
        struct stat st;
        if (stat(full_file_name.c_str(), &st) == -1)
            continue;
        const bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory)
            continue;
        out.push_back(full_file_name);
    }
    closedir(dir);
}

static void getBINFiles(std::vector<std::string> &out, const std::string &directory) {
    const std::string ext = ".bin";
    DIR *dir;
    dirent *ent;
    dir = opendir(directory.c_str());
    if (!dir)
        return;
    while ((ent = readdir(dir)) != nullptr) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;
        if ((file_name.length() >= ext.length())
            && (0 == file_name.compare(file_name.length() - ext.length(), ext.length(), ext))) {
            // proceed
        } else {
            continue;
        }
        struct stat st;
        if (stat(full_file_name.c_str(), &st) == -1)
            continue;
        const bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory)
            continue;
        out.push_back(full_file_name);
    }
    closedir(dir);
}

int num_requests = 4;

#define MIN_ITER 1000

#define USE_CALLBACK

int niter;
std::atomic<int> iterations_to_run;
std::mutex done_mutex;
std::condition_variable alldone;
int reallydone = 0;

std::vector<time_point> iter_start;
std::vector<time_point> iter_end;
std::vector<double> iter_time;

const int profile = 0;
std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;

int process(const std::string& modelFileName, const std::string& inputsDir,
            std::string& file_config_cl, int nBatch, int num_networks) {
    InferenceEngine::Core ie;
    niter /= nBatch;
    num_requests = num_requests * num_networks;

    // add some more requests. they'll be excluded on performance measurement
    niter += 2 * 2 * num_requests;

#if !(defined(_WIN32) || defined(_WIN64))
    if (pthread_setname_np(
#ifndef __APPLE__
    pthread_self(),
#endif
    "MainThread") != 0) {
        perror("Setting name for main thread failed");
    }
#endif

#ifdef USE_KMB_PLUGIN
    std::string deivceName = "KMB";
#else
    std::string deviceName = "MYRIAD";
#endif
    const auto pluginVersion = ie.GetVersions(deviceName);
    std::cout << "InferenceEngine: " << std::endl;
    std::cout << pluginVersion << std::endl << std::endl;

    std::ifstream file(file_config_cl);
    if (!file.is_open()) {
        file_config_cl.clear();
    }

    std::vector<std::string> pictures;
    getBMPFiles(pictures, inputsDir);
    int numPictures = pictures.size();

    std::vector<std::string> binaries;
    getBINFiles(binaries, inputsDir);
    int numBinaries = binaries.size();

    if (pictures.empty() && binaries.empty()) {
        std::cout << inputsDir << " directory doesn't contain input files" << std::endl;
        return 1;
    }

    InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(modelFileName);

    if (nBatch != 1) {
        std::cout << "Setting batch to : "<< nBatch << "\n";
        cnnNetwork.setBatchSize(nBatch);
    }

    InferenceEngine::InputsDataMap networkInputs;
    networkInputs = cnnNetwork.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputs;
    networkOutputs = cnnNetwork.getOutputsInfo();

    for (auto &input : networkInputs) {
        const auto inputPrecision = input.second->getPrecision();
        if (inputPrecision == InferenceEngine::Precision::FP32 ||
            inputPrecision == InferenceEngine::Precision::U8) {
            input.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }

    for (auto &output : networkOutputs) {
        const auto outputPrecision = output.second->getPrecision();
        if (outputPrecision == InferenceEngine::Precision::FP32) {
            output.second->setPrecision(InferenceEngine::Precision::FP16);
        }
    }

    std::vector<InferenceEngine::ExecutableNetwork> exeNetwork(num_networks);
    std::map<std::string, std::string> networkConfig;
    setConfig(networkConfig, file_config_cl);

    for (int n = 0; n < num_networks; ++n) {
        if (num_networks > 1)
            printf("Load network %d...\n", n);
        else
            printf("Load network... \n");
        fflush(stdout);
        exeNetwork[n] = ie.LoadNetwork(cnnNetwork, deviceName, networkConfig);
    }

    std::vector<InferenceEngine::InferRequest> request(num_requests);
    iter_start.resize(niter);
    iter_end.resize(niter);
    iter_time.resize(niter);

    iterations_to_run = niter - num_requests;

    for (int r = 0, idxPic = 0; r < num_requests; ++r) {
        int n = r % num_networks;
        request[r] = exeNetwork[n].CreateInferRequest();

        for (auto &input : networkInputs) {
            auto inputBlob = request[r].GetBlob(input.first);
            const auto& dims = inputBlob->getTensorDesc().getDims();
            auto layout = inputBlob->getTensorDesc().getLayout();

            // number of channels is 3 for Image, dims order is always NCHW
            const bool isImage = ((layout == InferenceEngine::NHWC || layout == InferenceEngine::NCHW) && dims[1] == 3);
            const bool isVideo = (inputBlob->getTensorDesc().getDims().size() == 5);
            if (isImage && (numPictures > 0)) {
                if (!loadImage(pictures[(idxPic++) % numPictures], inputBlob))
                    return 1;
            } else if (isVideo && (numPictures > 0)) {
                if (!loadVideo(pictures, inputBlob))
                    return 1;
            } else if (numBinaries > 0) {
                if (!loadBinaryTensor(binaries[(idxPic++) % numBinaries], inputBlob))
                    return 1;
            } else {
                std::cout << inputsDir << " directory doesn't contain correct input files" << std::endl;
                return 1;
            }
        }

        request[r].SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(
                [](InferenceEngine::InferRequest request, InferenceEngine::StatusCode code) -> void {
                    if (code != InferenceEngine::OK) {
                        std::cout << "Infer failed: " << code << std::endl;
                        exit(1);
                    }

                    int iter = --iterations_to_run;
                    int reqIdx = (niter - iter - 1) - num_requests;

                    iter_end[reqIdx] = Time::now();

                    if (profile && (reqIdx == niter / 2)) {
                        perfMap = request.GetPerformanceCounts();
                    }

                    if (iter >= 0) {
                        iter_start[reqIdx + (num_requests)] = Time::now();
                        request.StartAsync();
                    }

                    iter_time[reqIdx] = TIMEDIFF(iter_start[reqIdx], iter_end[reqIdx]);
                    // printf("request#%d %fms\n", reqIdx, iter_time[reqIdx]);

                    if (iter == -num_requests) {
                        reallydone = 1;
                        alldone.notify_all();
                    }
                });
    }

    printf("Inference started. Running %d iterations...\n", niter - 2 * 2 * num_requests);
    fflush(stdout);
    for (int r = 0; r < num_requests; ++r) {
        iter_start[r] = Time::now();
        request[r].StartAsync();
    }

    {
        std::unique_lock<std::mutex> lock(done_mutex);
        alldone.wait(lock, [&](){return reallydone;});
    }

    // check 10 time intervals to get min/max fps values
    const int fps_checks = 10;
    // exclude (2 * num_requests) first and last iterations
    int num_exclude = 2 * num_requests;
    time_point cstart = iter_end[num_exclude - 1];
    time_point cend = iter_end[niter - num_exclude - 1];

    double totalTime = (std::chrono::duration_cast<ms>(cend - cstart)).count();
    std::cout << std::endl << "Total time: " << (totalTime) << " ms" << std::endl;

    std::cout << "Average fps on " << (niter - 2 * num_exclude) << " iterations"
              << (nBatch == 1 ? ": " : (" of " + std::to_string(nBatch) + " frames: "))
              << static_cast<double>(niter - 2 * num_exclude) * 1000.0 * nBatch / (totalTime) << " fps" << std::endl;

    double check_time = totalTime / fps_checks;

    double min_fps = 100000;
    double max_fps = -100000;
    int citer = num_exclude;
    for (int f = 0; f < fps_checks; ++f) {
        int fiter = 0;
        auto fend = (f < fps_checks - 1) ? cstart + std::chrono::microseconds((unsigned int)(check_time * 1000)) : cend;
        while ((citer + fiter < niter - num_exclude) && iter_end[citer + fiter] <= fend) {
            fiter++;
        }

        double ffps = 1000 * fiter * nBatch / (check_time);
        min_fps = std::min(min_fps, ffps);
        max_fps = std::max(max_fps, ffps);
        citer += fiter;
        cstart = fend;
    }

    std::cout << "Min fps: " << min_fps << std::endl;
    std::cout << "Max fps: " << max_fps << std::endl;

    if (profile) {
        printPerformanceCounts(perfMap);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    niter = MIN_ITER;
    int num_networks = 1;
    int nBatch = 1;
    std::string file_config_cl;

    m_exename = argv[0];

    if (argc < 3) {
        print_usage();
        return 0;
    }

    auto parse = [](const std::string& src) {
        try {
            return std::stol(src, nullptr, 0);
        } catch (const std::invalid_argument& exception) {
            std::cout << "Cannot perform conversion for " << src << ": " << exception.what() << std::endl;
            print_usage();
            std::abort();
        } catch (const std::out_of_range& exception) {
            std::cout << src << " is out of range: " << exception.what() << std::endl;
            print_usage();
            std::abort();
        } catch (...) {
            std::cout << "Unexpected exception" << std::endl;
            print_usage();
            std::abort();
        }
    };

    if (argc > 3) {
        niter = static_cast<int>(parse(argv[3]));
    }

    if (argc > 4) {
        nBatch = static_cast<int>(parse(argv[4]));
    }

    if (argc > 5) {
        num_networks = static_cast<int>(parse(argv[5]));
    }

    if (argc > 6) {
        file_config_cl = std::string(argv[6]);
    }

    if (niter < MIN_ITER) {
        print_usage();
        return 0;
    }

    if (num_networks < 1 || num_networks > 16) {
        print_usage();
        return 0;
    }

    if (nBatch < 1) {
        print_usage();
        return 0;
    }

    try {
        std::string modelFileName(argv[1]);
        std::string inputsDir(argv[2]);
        return process(modelFileName, inputsDir, file_config_cl, nBatch, num_networks);
    }
    catch (const std::exception& ex) {
        std::cout << ex.what();
    }

    return -1;
}

static bool loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    const InferenceEngine::Layout layout = tensDesc.getLayout();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        std::cout << "loadImage error: Input must have FP16 precision" << std::endl;
        return false;
    }

    if (layout != InferenceEngine::NHWC && layout != InferenceEngine::NCHW) {
        std::cout << "loadImage error: Input must have NCHW or NHWC layout" << std::endl;
        return false;
    }

    BitMap reader(imageFilename);

    const auto dims = tensDesc.getDims();

    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t H = dims[2];
    const size_t W = dims[3];

    const size_t img_w = reader.width();
    const size_t img_h = reader.height();

    const auto strides = tensDesc.getBlockingDesc().getStrides();
    const auto strideN = strides[0];
    const auto strideC = layout == InferenceEngine::NHWC ? strides[3] : strides[1];
    const auto strideH = layout == InferenceEngine::NHWC ? strides[1] : strides[2];
    const auto strideW = layout == InferenceEngine::NHWC ? strides[2] : strides[3];

    const size_t numImageChannels = reader.size() / (reader.width() * reader.height());
    if (C != numImageChannels && C != 1) {
        std::cout << "loadImage error: Input channels mismatch: image channels " << numImageChannels << ", "
                  << "network channels " << C << ", expecting count of image channels are equal "
                  << "to count if network channels or count of network channels are equal to 1" << std::endl;
        return false;
    }

    int16_t* blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int16_t>>(blob)->data();
    const unsigned char* RGB8 = reader.getData().get();
    const float xScale = 1.0f * img_w / W;
    const float yScale = 1.0f * img_h / H;

    for (int n = 0; n != N; n++) {
        for (int h = 0; h < H; ++h) {
            int y = static_cast<int>(std::floor((h + 0.5f) * yScale));
            for (int w = 0; w < W; ++w) {
                int x = static_cast<int>(std::floor((w + 0.5f) * xScale));
                for (int c = 0; c < C; c++) {
                    blobDataPtr[n * strideN + c * strideC + h * strideH + w * strideW] =
                            InferenceEngine::PrecisionUtils::f32tof16(1.0 * RGB8[(y * img_w + x) * numImageChannels + c]);
                }
            }
        }
    }

    return true;
}

static bool loadVideo(const std::vector<std::string> &imagesFolder, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    const InferenceEngine::Layout layout = tensDesc.getLayout();

    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        std::cout << "loadVideo error: Input must have FP16 precision" << std::endl;
        return false;
    }
    if (layout != InferenceEngine::NDHWC && layout != InferenceEngine::NCDHW) {
        std::cout << "loadVideo error: Input must have NCDHW or NDHWC layout" << std::endl;
        return false;
    }

    const auto dims = tensDesc.getDims();
    const size_t N = dims[0];
    const size_t C = dims[1];
    const size_t D = dims[2];
    const size_t H = dims[3];
    const size_t W = dims[4];

    const auto numUsedImages = std::min(D, imagesFolder.size());
    const auto strides = tensDesc.getBlockingDesc().getStrides();
    const auto strideN = strides[0];
    const auto strideC = layout == InferenceEngine::NDHWC ? strides[4] : strides[1];
    const auto strideD = layout == InferenceEngine::NDHWC ? strides[1] : strides[2];
    const auto strideH = layout == InferenceEngine::NDHWC ? strides[2] : strides[3];
    const auto strideW = layout == InferenceEngine::NDHWC ? strides[3] : strides[4];

    auto d = 0;
    int16_t* blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int16_t>>(blob)->data();
    for ( ; d < numUsedImages; d++) {
        BitMap reader(imagesFolder[d]);
        const size_t img_w = reader.width();
        const size_t img_h = reader.height();
        const size_t numImageChannels = reader.size() / (reader.width() * reader.height());

        if (C != numImageChannels && C != 1) {
            std::cout << "loadVideo error: Input channels mismatch: image channels " << numImageChannels << ", "
                      << "network channels " << C << ", expecting count of image channels are equal "
                      << "to count if network channels or count of network channels are equal to 1" << std::endl;
            return false;
        }

        const unsigned char* RGB8 = reader.getData().get();
        const float xScale = 1.0f * img_w / W;
        const float yScale = 1.0f * img_h / H;

        for (int n = 0; n != N; n++) {
            for (int h = 0; h < H; ++h) {
                int y = static_cast<int>(std::floor((h + 0.5f) * yScale));
                for (int w = 0; w < W; ++w) {
                    int x = static_cast<int>(std::floor((w + 0.5f) * xScale));
                    for (int c = 0; c < C; c++) {
                        blobDataPtr[n * strideN + c * strideC + d * strideD + h * strideH + w * strideW] =
                                InferenceEngine::PrecisionUtils::f32tof16(1.0 * RGB8[(y * img_w + x) * numImageChannels + c]);
                    }
                }
            }
        }
    }

    for (; d < D; d++)
        for (auto n = 0; n != N; n++)
            for (auto c = 0; c < C; c++)
                for (auto k = 0; k < strideD; k++) {
                    blobDataPtr[n * strideN + c * strideC + (d)     * strideD + k] =
                    blobDataPtr[n * strideN + c * strideC + (d - 1) * strideD + k];
                }

    return true;
}

bool loadBinaryTensor(const std::string &binaryFilename, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();
    if (tensDesc.getPrecision() != InferenceEngine::Precision::FP16) {
        std::cout << "loadBinaryTensor error: Input must have FP16 precision" << std::endl;
        return false;
    }

    std::ifstream binaryFile(binaryFilename, std::ios_base::binary | std::ios_base::ate);

    if (!binaryFile) {
        std::cout << "loadBinaryTensor error: While opening a file an error is encountered" << std::endl;
        return false;
    }

    int fileSize = binaryFile.tellg();
    binaryFile.seekg(0, std::ios_base::beg);
    size_t count = blob->size();
    if (fileSize != count * sizeof(float)) {
        std::cout << "loadBinaryTensor error: File contains insufficient items" << std::endl;
        return false;
    }

    if (binaryFile.good()) {
        int16_t *blobDataPtr = std::dynamic_pointer_cast<InferenceEngine::TBlob<int16_t>>(blob)->data();
        for (size_t i = 0; i < count; i++) {
            float tmp = 0.f;
            binaryFile.read(reinterpret_cast<char *>(&tmp), sizeof(float));
            blobDataPtr[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
        }
    } else {
        std::cout << "loadBinaryTensor error: While reading a file an error is encountered" << std::endl;
        return false;
    }
    return true;
}
