/*
 // Copyright (c) 2018 Intel Corporation
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //      http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 */

/**
 * @brief The entry point for Inference Engine validation application
 * @file validation_app/main.cpp
 */
#include <gflags/gflags.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <limits>
#include <iomanip>
#include <memory>

#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "user_exception.hpp"
#include "ClassificationProcessor.hpp"
#include "SSDObjectDetectionProcessor.hpp"
#include "YOLOObjectDetectionProcessor.hpp"

using namespace std;
using namespace InferenceEngine;

using InferenceEngine::details::InferenceEngineException;

#define DEFAULT_PATH_P "./lib"

/// @brief message for help argument
static const char help_message[] = "Print a usage message";
/// @brief message for images argument
static const char image_message[] = "Required. Folder with validation images, folders grouped by labels or a .txt file "
                                    "list for classification networks or a VOC-formatted dataset for object detection networks";
/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";
/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model";
/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, "
                                     "the sample will look for this plugin only";
/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. "
                                            "Sample will look for a suitable plugin for device specified (CPU by default)";
/// @brief message for label argument
static const char label_message[] = "Path to the file containing labels for the model";
/// @brief message for batch argumenttype
static const char batch_message[] = "Batch size value. If not specified, the batch size value is determined from IR";
/// @brief message for dump argument
static const char dump_message[] = "Dump filenames and inference results to a csv file";
/// @brief message for network type
static const char type_message[] = "Type of the network being scored (\"C\" by default)";
/// @brief message for pp-type
static const char preprocessing_type[] = "Preprocessing type. One of \"None\", \"Resize\", \"ResizeCrop\"";
/// @brief message for pp-crop-size
static const char preprocessing_size[] = "Preprocessing size (used with ppType=\"ResizeCrop\")";
static const char preprocessing_width[] = "Preprocessing width (overrides -ppSize, used with ppType=\"ResizeCrop\")";
static const char preprocessing_height[] = "Preprocessing height (overrides -ppSize, used with ppType=\"ResizeCrop\")";

static const char obj_detection_annotations_message[] = "Required for OD networks. Path to the folder containing .xml annotations for images";

static const char obj_detection_classes_message[] = "Required for OD networks. Path to the file containing classes list";

static const char obj_detection_subdir_message[] = "Folder between the image path (-i) and image name, specified in the .xml. Use JPEGImages for VOC2007";

static const char obj_detection_kind_message[] = "Kind of an object detection network: SSD";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for clDNN (GPU)-targeted custom kernels."
                                           "Absolute path to the xml file with the kernel descriptions";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for MKLDNN (CPU)-targeted custom layers."
                                                 "Absolute path to a shared library with the kernel implementations";

static const char zero_background_message[] = "\"Zero is a background\" flag. Some networks are trained with a modified dataset where the class IDs "
                                              "are enumerated from 1, but 0 is an undefined \"background\" class (which is never detected)";

/// @brief Network type options and their descriptions
static const char* types_descriptions[][2] = {
    { "C", "classification" },
//    { "SS", "semantic segmentation" },    // Not supported yet
    { "OD", "object detection" },
    { nullptr, nullptr }
};

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);
/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// @brief Define parameter for set plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// @brief Define parameter for labels file name <br>
/// Default is empty
DEFINE_string(OCl, "", label_message);
/// @brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, DEFAULT_PATH_P, plugin_path_message);
/// @brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);
/// @brief Define parameter for batch size <br>
/// Default is 0 (that means don't specify)
DEFINE_int32(b, 0, batch_message);
/// @brief Define flag to dump results to a file <br>
DEFINE_bool(dump, false, dump_message);
/// @brief Define a network type parameter
DEFINE_string(t, "C", type_message);

/// @brief Preprocessing type
DEFINE_string(ppType, "", preprocessing_type);

/// @brief Preprocessing size
DEFINE_int32(ppSize, 0, preprocessing_size);
DEFINE_int32(ppWidth, 0, preprocessing_width);
DEFINE_int32(ppHeight, 0, preprocessing_height);

DEFINE_bool(Czb, false, zero_background_message);

DEFINE_string(ODa, "", obj_detection_annotations_message);

DEFINE_string(ODc, "", obj_detection_classes_message);

DEFINE_string(ODsubdir, "", obj_detection_subdir_message);

/// @brief kind of an object detection network
DEFINE_string(ODkind, "SSD", obj_detection_kind_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/**
 * @brief This function show a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Usage: validation_app [OPTION]" << std::endl << std::endl;
    std::cout << "Available options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -t <type>                 " << type_message << std::endl;
    for (int i = 0; types_descriptions[i][0] != nullptr; i++) {
        std::cout << "      -t \"" << types_descriptions[i][0] << "\" for " << types_descriptions[i][1] << std::endl;
    }
    std::cout << "    -i <path>                 " << image_message << std::endl;
    std::cout << "    -m <path>                 " << model_message << std::endl;
    std::cout << "    -l <absolute_path>        " << custom_cpu_library_message << std::endl;
    std::cout << "    -c <absolute_path>        " << custom_cldnn_message << std::endl;
    std::cout << "    -d <device>               " << target_device_message << std::endl;
    std::cout << "    -b N                      " << batch_message << std::endl;
    std::cout << "    -ppType <type>            " << preprocessing_type << std::endl;
    std::cout << "    -ppSize N                 " << preprocessing_size << std::endl;
    std::cout << "    -ppWidth W                " << preprocessing_width << std::endl;
    std::cout << "    -ppHeight H               " << preprocessing_height << std::endl;
    std::cout << "    --dump                    " << dump_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Classification-specific options:" << std::endl;
    std::cout << "      -Czb true               " << zero_background_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Object detection-specific options:" << std::endl;
    std::cout << "      -ODkind <kind>          " << obj_detection_kind_message << std::endl;
    std::cout << "      -ODa <path>             " << obj_detection_annotations_message << std::endl;
    std::cout << "      -ODc <file>             " << obj_detection_classes_message << std::endl;
    std::cout << "      -ODsubdir <name>        " << obj_detection_subdir_message << std::endl << std::endl;
}

enum NetworkType {
    Undefined = -1,
    Classification,
    ObjDetection
};

std::string strtolower(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

/**
 * @brief The main function of inference engine sample application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ---------------------------Parsing and validation of input args--------------------------------------
        slog::info << "Parsing input parameters" << slog::endl;

        bool noOptions = argc == 1;

        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h || noOptions) {
            showUsage();
            return 1;
        }

        UserExceptions ee;

        NetworkType netType = Undefined;
        // Checking the network type
        if (std::string(FLAGS_t) == "C") {
            netType = Classification;
        } else if (std::string(FLAGS_t) == "OD") {
            netType = ObjDetection;
        } else {
            ee << UserException(5, "Unknown network type specified (invalid -t option)");
        }

        // Checking required options
        if (FLAGS_m.empty()) ee << UserException(3, "Model file not specified (missing -m option)");
        if (FLAGS_i.empty()) ee << UserException(4, "Images list not specified (missing -i option)");
        if (FLAGS_d.empty()) ee << UserException(5, "Target device not specified (missing -d option)");
        if (FLAGS_b < 0) ee << UserException(6, "Batch should be positive (invalid -b option value)");

        if (netType == ObjDetection) {
            // Checking required OD-specific options
            if (FLAGS_ODa.empty()) ee << UserException(11, "Annotations folder not specified for object detection (missing -a option)");
            if (FLAGS_ODc.empty()) ee << UserException(12, "Classes file not specified (missing -c option)");
            if (FLAGS_b > 0) ee << UserException(13, "Batch option other than 0 is not supported for object detection networks");
        }

        if (!ee.empty()) throw ee;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------Load plugin for inference engine------------------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        /** Here we are loading the library with extensions if provided**/
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64", "" }).getPluginByDevice(FLAGS_d);

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        printPluginVersion(plugin, std::cout);

        CsvDumper dumper(FLAGS_dump);

        std::shared_ptr<Processor> processor;

        PreprocessingOptions preprocessingOptions;
        if (strtolower(FLAGS_ppType.c_str()) == "none") {
            preprocessingOptions = PreprocessingOptions(false, ResizeCropPolicy::DoNothing);
        } else if (strtolower(FLAGS_ppType) == "resizecrop") {
            size_t ppWidth = FLAGS_ppSize;
            size_t ppHeight = FLAGS_ppSize;

            if (FLAGS_ppWidth > 0) ppWidth = FLAGS_ppSize;
            if (FLAGS_ppHeight > 0) ppHeight = FLAGS_ppSize;

            if (FLAGS_ppSize > 0 || (FLAGS_ppWidth > 0 && FLAGS_ppHeight > 0)) {
                preprocessingOptions = PreprocessingOptions(false, ResizeCropPolicy::ResizeThenCrop, ppWidth, ppHeight);
            } else {
                THROW_USER_EXCEPTION(2) << "Size should be specified for preprocessing type " << FLAGS_ppType;
            }
        } else if (strtolower(FLAGS_ppType) == "resize" || FLAGS_ppType.empty()) {
            preprocessingOptions = PreprocessingOptions(false, ResizeCropPolicy::Resize);
        } else {
            THROW_USER_EXCEPTION(2) << "Unknown preprocessing type: " << FLAGS_ppType;
        }

        if (netType == Classification) {
            processor = std::shared_ptr<Processor>(
                    new ClassificationProcessor(FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_b, plugin, dumper, FLAGS_l, preprocessingOptions, FLAGS_Czb));
        } else if (netType == ObjDetection) {
            if (FLAGS_ODkind == "SSD") {
                processor = std::shared_ptr<Processor>(
                        new SSDObjectDetectionProcessor(FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b, 0.5, plugin, dumper, FLAGS_ODa, FLAGS_ODc));
            } else if (FLAGS_ODkind == "YOLO") {
                processor = std::shared_ptr<Processor>(
                        new YOLOObjectDetectionProcessor(FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b, 0.5, plugin, dumper, FLAGS_ODa, FLAGS_ODc));
            }
        } else {
            THROW_USER_EXCEPTION(2) <<  "Unknown network type specified" << FLAGS_ppType;
        }
        if (!processor.get()) {
            THROW_USER_EXCEPTION(2) <<  "Processor pointer is invalid" << FLAGS_ppType;
        }
        slog::info << (FLAGS_d.empty() ? "Plugin: " + FLAGS_p : "Device: " + FLAGS_d) << slog::endl;
        shared_ptr<Processor::InferenceMetrics> pIM = processor->Process();
        processor->Report(*pIM.get());

        if (dumper.dumpEnabled()) {
            slog::info << "Dump file generated: " << dumper.getFilename() << slog::endl;
        }
    } catch (const InferenceEngineException& ex) {
        slog::err << "Inference problem: \n" << ex.what() << slog::endl;
        return 1;
    } catch (const UserException& ex) {
        slog::err << "Input problem: \n" << ex.what() << slog::endl;
        showUsage();
        return ex.exitCode();
    } catch (const UserExceptions& ex) {
        if (ex.list().size() == 1) {
            slog::err << "Input problem: " << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        } else {
            const char* s = ex.what();
            slog::err << "Input problems: \n" << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        }
    }
    return 0;
}
