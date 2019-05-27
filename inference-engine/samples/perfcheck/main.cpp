// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)
#include <os/windows/w_dirent.h>
#else
#include <sys/stat.h>
#include <dirent.h>
#endif

#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <map>
#include <cmath>
#include <future>
#include <atomic>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <limits>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "inference_engine.hpp"
#include "ext_list.hpp"

#include "vpu/vpu_plugin_config.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"

#include "perfcheck.h"


static bool parseCommandLine(int *argc, char ***argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
    }

    if (FLAGS_num_iterations < MIN_ITERATIONS) {
        throw std::invalid_argument("Number of iterations must be not smaller than 1000. "
                                    "Got " + std::to_string(FLAGS_num_iterations));
    }

    if (MAX_NETWORKS < FLAGS_num_networks) {
        throw std::invalid_argument("Only number of networks not greater than " + std::to_string(MAX_NETWORKS) + " "
                                    "is supported. Got " + std::to_string(FLAGS_num_networks));
    }

    if (FLAGS_d.empty()) {
        throw std::invalid_argument("Plugin name is required");
    }

    if (1 < *argc) {
        std::stringstream message;
        message << "Unknown arguments: ";
        for (auto arg = 1; arg < *argc; arg++) {
            message << argv[arg];
            if (arg < *argc) {
                message << " ";
            }
        }
        throw std::invalid_argument(message.str());
    }

    return true;
}

static std::map<std::string, std::string> parseConfig(const std::string &configName, char comment = '#') {
    std::map<std::string, std::string> config = {};

    std::ifstream file(configName);
    if (!file.is_open()) {
        return config;
    }

    std::string key, value;
    while (file >> key >> value) {
        if (key.empty() || key[0] == comment) {
            continue;
        }
        config[key] = value;
    }

    return config;
}

static std::size_t getNumberRequests(const std::string &plugin) {
    static const std::unordered_map<std::string, std::size_t> supported_plugins = {
        { "MYRIAD", 4   },
        { "FPGA",   3   },
    };

    auto device = plugin;
    if (plugin.find("HETERO:") == 0) {
        auto separator   = plugin.find(",");
        auto deviceBegin = std::string("HETERO:").size();
        auto deviceEnd   = separator == std::string::npos ? plugin.size() : separator;
        device = plugin.substr(deviceBegin, deviceEnd - deviceBegin);
    }

    auto num_requests = supported_plugins.find(device);
    return num_requests == supported_plugins.end() ? 1 : num_requests->second;
}

#if defined(WIN32) || defined(__APPLE__)
typedef std::chrono::time_point<std::chrono::steady_clock> time_point;
#else
typedef std::chrono::time_point<std::chrono::system_clock> time_point;
#endif

static void printFPS(std::size_t num_requests, std::size_t num_intervals, const std::vector<time_point> &points) {
    std::size_t num_exclude = 2 * num_requests;
    /* evaluate from the end of previous */
    std::size_t first_point = num_exclude - 1;
    std::size_t last_point  = points.size() - num_exclude;
    auto begin = points[first_point];
    auto end   = points[last_point - 1];

    using ms = std::chrono::duration<double, std::ratio<1, 1000>>;

    auto num_iterations = last_point - first_point - 1;
    auto total = std::chrono::duration_cast<ms>(end - begin).count();
    auto avg_fps = static_cast<double>(num_iterations) * 1000.0 * FLAGS_batch / total;

    auto min_fps = std::numeric_limits<double>::max();
    auto max_fps = std::numeric_limits<double>::min();
    double step = total / num_intervals;
    std::size_t first_point_in_interval = first_point + 1;
    auto first_time_in_interval = std::chrono::time_point_cast<ms>(begin);
    for (std::size_t interval = 0; interval < num_intervals; interval++) {
        std::size_t num_points_in_interval = 0;
        auto last_time_in_interval = first_time_in_interval + ms(step);
        if (interval == num_intervals - 1) {
            last_time_in_interval = end;
        }

        while (first_point_in_interval + num_points_in_interval < last_point &&
               points[first_point_in_interval + num_points_in_interval] <= last_time_in_interval) {
            num_points_in_interval++;
        }

        double fps = num_points_in_interval * FLAGS_batch / step * 1000;
        min_fps = std::min(min_fps, fps);
        max_fps = std::max(max_fps, fps);

        first_point_in_interval += num_points_in_interval;
        first_time_in_interval = last_time_in_interval;
    }

    std::cout << std::endl;
    std::cout << "Total time:     " << total << " ms";
    std::cout << std::endl;

    std::cout << "Num iterations: " << num_iterations << std::endl;
    std::cout << "Batch:          " << FLAGS_batch << std::endl;

    std::cout << "Min FPS:        " << min_fps << std::endl;
    std::cout << "Avg FPS:        " << avg_fps << std::endl;
    std::cout << "Max FPS:        " << max_fps << std::endl;
}

template<typename T>
static bool isImage(const T &blob) {
    auto descriptor = blob->getTensorDesc();
    if (descriptor.getLayout() != InferenceEngine::NCHW) {
        return false;
    }

    auto channels = descriptor.getDims()[1];
    return channels == 3;
}

static std::vector<std::string> extractFilesByExtension(const std::string &directory, const std::string &extension) {
    std::vector<std::string> files;

    DIR *dir = opendir(directory.c_str());
    if (!dir) {
        throw std::invalid_argument("Can not open " + directory);
    }

    auto getExtension = [](const std::string &name) {
        auto extensionPosition = name.rfind('.', name.size());
        return extensionPosition == std::string::npos ? "" : name.substr(extensionPosition + 1, name.size() - 1);
    };

    dirent *ent = nullptr;
    while ((ent = readdir(dir))) {
        std::string file_name = ent->d_name;
        if (getExtension(file_name) != extension) {
            continue;
        }

        std::stringstream stream;
        stream << directory << "/" << file_name;

        auto full_file_name = stream.str();

        struct stat st = {};
        if (stat(full_file_name.c_str(), &st) != 0) {
            continue;
        }

        bool is_directory = (st.st_mode & S_IFDIR) != 0;
        if (is_directory) {
            continue;
        }

        files.push_back(full_file_name);
    }

    closedir(dir);

    return files;
}

static float asfloat(uint32_t v) {
    union {
        float f;
        std::uint32_t u;
    } converter = {0};
    converter.u = v;
    return converter.f;
}

static short f32tof16(float x) {
    static float min16 = asfloat((127 - 14) << 23);

    static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
    static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

    static constexpr std::uint32_t EXP_MASK_F32 = 0x7F800000U;

    union {
        float f;
        uint32_t u;
    } v = {0};
    v.f = x;

    uint32_t s = (v.u >> 16) & 0x8000;

    v.u &= 0x7FFFFFFF;

    if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
        if (v.u & 0x007FFFFF) {
            return static_cast<short>(s | (v.u >> (23 - 10)) | 0x0200);
        } else {
            return static_cast<short>(s | (v.u >> (23 - 10)));
        }
    }

    float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
    v.f += halfULP;

    if (v.f < min16 * 0.5f) {
        return static_cast<short>(s);
    }

    if (v.f < min16) {
        return static_cast<short>(s | (1 << 10));
    }

    if (v.f >= max16) {
        return static_cast<short>(max16f16 | s);
    }

    v.u -= ((127 - 15) << 23);

    v.u >>= (23 - 10);

    return static_cast<short>(v.u | s);
}

static void loadImage(const std::string &imageFilename, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();

    cv::Mat image = cv::imread(imageFilename);
    if (image.empty()) {
        throw std::invalid_argument("Can not read image from " + imageFilename);
    }

    std::size_t batch = blob->dims()[3];
    std::size_t w = blob->dims()[0];
    std::size_t h = blob->dims()[1];
    auto img_w = static_cast<std::size_t>(image.cols);
    auto img_h = static_cast<std::size_t>(image.rows);

    auto numBlobChannels = blob->dims()[2];
    auto numImageChannels = static_cast<std::size_t>(image.channels());
    if (numBlobChannels != numImageChannels && numBlobChannels != 1) {
        throw std::invalid_argument("Input channels mismatch: image channels " + std::to_string(numImageChannels) +
                                    ", network channels " + std::to_string(numBlobChannels) +
                                    ", expecting count of image channels are equal to count if network channels"
                                    "or count of network channels are equal to 1");
    }

    auto nPixels = w * h;
    unsigned char *RGB8 = image.data;
    float xscale = 1.0f * img_w / w;
    float yscale = 1.0f * img_h / h;

    for (std::size_t n = 0; n != batch; n++) {
        for (std::size_t i = 0; i < h; ++i) {
            auto y = static_cast<std::size_t>(std::floor((i + 0.5f) * yscale));
            for (std::size_t j = 0; j < w; ++j) {
                auto x = static_cast<std::size_t>(std::floor((j + 0.5f) * xscale));
                for (std::size_t k = 0; k < numBlobChannels; k++) {
                    float value = 1.0f * RGB8[(y * img_w + x) * numImageChannels + k];
                    if (InferenceEngine::Precision::FP16 == tensDesc.getPrecision()) {
                        if (tensDesc.getLayout() == InferenceEngine::NHWC) {
                            blob->buffer().as<std::int16_t *>()[n * h * w * numBlobChannels + (i * w + j) * numBlobChannels + k] = f32tof16(value);
                        } else {
                            blob->buffer().as<std::int16_t *>()[n * h * w * numBlobChannels + (i * w + j) + k * nPixels] = f32tof16(value);
                        }
                    } else {
                        if (tensDesc.getLayout() == InferenceEngine::NHWC) {
                            blob->buffer().as<float *>()[n * h * w * numBlobChannels + (i * w + j) * numBlobChannels + k] = value;
                        } else {
                            blob->buffer().as<float *>()[n * h * w * numBlobChannels + (i * w + j) + k * nPixels] = value;
                        }
                    }
                }
            }
        }
    }
}

static void loadBinaryTensor(const std::string &binaryFileName, InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::TensorDesc tensDesc = blob->getTensorDesc();

    std::ifstream binaryFile(binaryFileName, std::ios_base::binary | std::ios_base::ate);
    if (!binaryFile) {
        throw std::invalid_argument("Can not open \"" + binaryFileName + "\"");
    }

    auto fileSize = static_cast<std::size_t>(binaryFile.tellg());
    binaryFile.seekg(0, std::ios_base::beg);
    if (!binaryFile.good()) {
        throw std::invalid_argument("Can not read \"" + binaryFileName + "\"");
    }

    auto networkSize = blob->size() * sizeof(float);
    if (fileSize != networkSize) {
        throw std::invalid_argument("File \"" + binaryFileName + "\" contains " + std::to_string(fileSize) + " bytes "
                                    "but network expects " + std::to_string(networkSize));
    }

    for (std::size_t i = 0; i < blob->size(); i++) {
        float src = 0.f;
        binaryFile.read(reinterpret_cast<char *>(&src), sizeof(float));
        if (InferenceEngine::Precision::FP16 == tensDesc.getPrecision()) {
            blob->buffer().as<std::int16_t *>()[i] = f32tof16(src);
        } else {
            blob->buffer().as<float *>()[i] = src;
        }
    }
}

static void loadInputs(std::size_t requestIdx, const std::vector<std::string> &images,
                       const std::vector<std::string> &binaries, InferenceEngine::InferRequest &request,
                       InferenceEngine::CNNNetwork &network) {
    for (auto &&input : network.getInputsInfo()) {
        auto blob = request.GetBlob(input.first);

        if (isImage(blob)) {
            loadImage(images[requestIdx % images.size()], blob);
        } else {
            loadBinaryTensor(binaries[requestIdx % binaries.size()], blob);
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        slog::info << "Inference Engine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        if (!parseCommandLine(&argc, &argv)) {
            return EXIT_SUCCESS;
        }

        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        slog::info << "Loading network files:" <<
            slog::endl << "\t" << FLAGS_m <<
            slog::endl << "\t" << binFileName <<
        slog::endl;

        InferenceEngine::CNNNetReader networkReader;
        networkReader.ReadNetwork(FLAGS_m);
        networkReader.ReadWeights(binFileName);

        auto network = networkReader.getNetwork();
        network.setBatchSize(FLAGS_batch);

        if (FLAGS_d.find("MYRIAD") != std::string::npos || FLAGS_d.find("HDDL") != std::string::npos) {
            /**
             * on VPU devices FP16 precision allows avoid extra conversion operations and shows better performance
             **/
            for (auto &&input : network.getInputsInfo()) {
                input.second->setPrecision(InferenceEngine::Precision::FP16);
            }

            for (auto &&output : network.getOutputsInfo()) {
                output.second->setPrecision(InferenceEngine::Precision::FP16);
            }
        }

        auto plugin = InferenceEngine::PluginDispatcher({FLAGS_pp}).getPluginByDevice(FLAGS_d);

        /* If CPU device, load default library with extensions that comes with the product */
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferencing custom topologies.
             **/
            plugin.AddExtension(std::make_shared<InferenceEngine::Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            plugin.AddExtension(InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l));
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }

        if (!FLAGS_c.empty()) {
            /* clDNN Extensions are loaded from an .xml description and OpenCL kernel files */
            plugin.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        auto config = parseConfig(FLAGS_config);
        std::vector<InferenceEngine::ExecutableNetwork> networks(FLAGS_num_networks);
        for (std::size_t net = 0; net < networks.size(); ++net) {
            slog::info << "Loading network " << net;
            if (FLAGS_d.find("FPGA") != std::string::npos) {
                if (FLAGS_num_fpga_devices != 1) {
                    config[InferenceEngine::PluginConfigParams::KEY_DEVICE_ID] = std::to_string(net % FLAGS_num_fpga_devices);
                    slog::info << " to device " << (net % FLAGS_num_fpga_devices);
                }
            }
            slog::info << slog::endl;

            networks[net] = plugin.LoadNetwork(network, config);
        }
        slog::info << "All networks are loaded" << slog::endl;

        auto num_requests = FLAGS_num_requests == 0 ? getNumberRequests(FLAGS_d) : FLAGS_num_requests;

        auto images = extractFilesByExtension(FLAGS_inputs_dir, "bmp");
        auto hasImageInput = [](const InferenceEngine::CNNNetwork &net) {
            auto inputs = net.getInputsInfo();
            auto isImageInput = [](const InferenceEngine::InputsDataMap::value_type &input) {
                return isImage(input.second);
            };
            return std::any_of(inputs.begin(), inputs.end(), isImageInput);
        };

        if (hasImageInput(network) && images.empty()) {
            throw std::invalid_argument("The directory \"" + FLAGS_inputs_dir + "\" does not contain images for network");
        }

        auto binaries = extractFilesByExtension(FLAGS_inputs_dir, "bin");
        auto hasBinaryInput = [](const InferenceEngine::CNNNetwork &net) {
            auto inputs = net.getInputsInfo();
            auto isBinaryInput = [](const InferenceEngine::InputsDataMap::value_type &input) {
                return !isImage(input.second);
            };
            return std::any_of(inputs.begin(), inputs.end(), isBinaryInput);
        };

        if (hasBinaryInput(network) && binaries.empty()) {
            throw std::invalid_argument("The directory \"" + FLAGS_inputs_dir + "\" does not contain binaries for network");
        }

        std::size_t iteration{0};
        std::mutex dump_time;
        std::atomic<std::size_t> num_finished{0};

        std::promise<void> done;
        num_requests *= FLAGS_num_networks;
        std::size_t num_iterations = 2 * num_requests + FLAGS_num_iterations + 2 * num_requests;

        std::vector<InferenceEngine::InferRequest> requests(num_requests);
        std::vector<time_point> time_points(num_iterations);

        using callback_t = std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>;

        for (std::size_t request = 0; request < num_requests; ++request) {
            requests[request] = networks[request % networks.size()].CreateInferRequest();

            loadInputs(request, images, binaries, requests[request], network);

            callback_t callback =
                [num_requests, num_iterations, &iteration, &time_points, &dump_time, &num_finished, &done]
                (InferenceEngine::InferRequest inferRequest, InferenceEngine::StatusCode code) {
                if (code != InferenceEngine::StatusCode::OK) {
                    THROW_IE_EXCEPTION << "Infer request failed with code " << code;
                }

                std::size_t current_finished_iteration = 0;
                {
                    std::lock_guard<std::mutex> lock(dump_time);

                    current_finished_iteration = iteration++;
                    if (current_finished_iteration < num_iterations) {
                        time_points[current_finished_iteration] = std::chrono::high_resolution_clock::now();
                    }
                }

                if (current_finished_iteration < num_iterations - 1) {
                    inferRequest.StartAsync();
                } else {
                    if (++num_finished == num_requests) {
                        done.set_value();
                    }
                }
            };

            requests[request].SetCompletionCallback<callback_t>(callback);
        }

        auto doneFuture = done.get_future();

        for (auto &&request : requests) {
            request.StartAsync();
        }

        doneFuture.wait();

        printFPS(num_requests, 10, time_points);
    } catch (const std::exception &error) {
        slog::err << error.what() << slog::endl;
        return EXIT_FAILURE;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
