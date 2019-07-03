// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include "calibrator_processors.h"
#include "SSDObjectDetectionProcessor.hpp"
#include "YOLOObjectDetectionProcessor.hpp"
#include "ie_icnn_network_stats.hpp"
#include "details/caseless.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

using InferenceEngine::details::InferenceEngineException;

/// @brief Message for help argument
static const char help_message[] = "Print a help message";
/// @brief Message for images argument
static const char image_message[] = "Required. Path to a directory with validation images. For Classification models, the directory must contain"
                                    " folders named as labels with images inside or a .txt file with"
                                    " a list of images. For Object Detection models, the dataset must be in"
                                    " VOC format.";
/// @brief Message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";
/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model, including model name and "
                                    "extension.";
/// @brief Message for plugin argument
static const char plugin_message[] = "Plugin name. For example, CPU. If this parameter is passed, "
                                     "the sample looks for a specified plugin only.";
/// @brief Message for assigning cnn calculation to device
static const char target_device_message[] = "Target device to infer on: CPU (default), GPU, FPGA, HDDL or MYRIAD."
                                            " The application looks for a suitable plugin for the specified device.";
/// @brief Message for label argument
static const char label_message[] = "Path to a file with labels for a model";
/// @brief M`essage for batch argumenttype
static const char batch_message[] = "Batch size value. If not specified, the batch size value is taken from IR";
/// @brief Message for dump argument
static const char dump_message[] = "Dump file names and inference results to a .csv file";
/// @brief Message for network type
static const char type_message[] = "Type of an inferred network (\"C\" by default)";
/// @brief Message for pp-type
static const char preprocessing_type[] = "Preprocessing type. Options: \"None\", \"Resize\", \"ResizeCrop\"";
/// @brief Message for pp-crop-size
static const char preprocessing_size[] = "Preprocessing size (used with ppType=\"ResizeCrop\")";
static const char preprocessing_width[] = "Preprocessing width (overrides -ppSize, used with ppType=\"ResizeCrop\")";
static const char preprocessing_height[] = "Preprocessing height (overrides -ppSize, used with ppType=\"ResizeCrop\")";

static const char obj_detection_annotations_message[] = "Required for Object Detection models. Path to a directory"
                                                        " containing an .xml file with annotations for images.";

static const char obj_detection_classes_message[] = "Required for Object Detection models. Path to a file with"
                                                    " a list of classes";

static const char obj_detection_subdir_message[] = "Directory between the path to images (specified with -i) and image name (specified in the"
                                                   " .xml file). For VOC2007 dataset, use JPEGImages.";

static const char obj_detection_kind_message[] = "Type of an Object Detection model. Options: SSD";

/// @brief Message for GPU custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
                                           "Absolute path to an .xml file with the kernel descriptions.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernel implementations.";
/// @brief Message for labels file
static const char labels_file_message[] = "Labels file path. The labels file contains names of the dataset classes";

static const char zero_background_message[] = "\"Zero is a background\" flag. Some networks are trained with a modified"
                                              " dataset where the class IDs "
                                              " are enumerated from 1, but 0 is an undefined \"background\" class"
                                              " (which is never detected)";

static const char stream_output_message[] = "Flag for printing progress as a plain text. When used, interactive progress"
                                            " bar is replaced with multiline output";

static const char convert_fc_message[] = "Convert FullyConnected layers to Int8 or not (false by default)";


/// @brief Network type options and their descriptions
static const char* types_descriptions[][2] = {
    { "C", "calibrate Classification network and write the calibrated network to IR" },
//    { "SS", "semantic segmentation" },    // Not supported yet
    { "OD", "calibrate Object Detection network and write the calibrated network to IR" },
    { "RawC", "collect only statistics for Classification network and write statistics to IR. With this option, a model is not calibrated. For calibration "
              "and statisctics collection, use \"-t C\" instead." },
    { "RawOD", "collect only statistics for Object Detection network and write statistics to IR. With this option, a model is not calibrated. For calibration "
               "and statisctics collection, use \"-t OD\" instead" },
    { nullptr, nullptr }
};

static const char accuracy_threshold_message[] = "Threshold for a maximum accuracy drop of quantized model."
                                                 " Must be an integer number (percents)"
                                                 " without a percent sign. Default value is 1, which stands for accepted"
                                                 " accuracy drop in 1%";
static const char number_of_pictures_message[] = "Number of pictures from the whole validation set to"
                                                 "create the calibration dataset. Default value is 0, which stands for"
                                                 "the whole provided dataset";
static const char output_model_name[] = "Output name for calibrated model. Default is <original_model_name>_i8.xml|bin";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// @brief Define parameter for a path to images <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);
/// @brief Define parameter for a path to model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// @brief Define parameter for a plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// @brief Define parameter for a path to a file with labels <br>
/// Default is empty
DEFINE_string(OCl, "", label_message);
/// @brief Define parameter for a path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);
/// @brief Define paraneter for a target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);
/// @brief Define parameter for batch size <br>
/// Default is 0 (which means that batch size is not specified)
DEFINE_int32(b, 0, batch_message);
/// @brief Define flag to dump results to a file <br>
DEFINE_bool(dump, false, dump_message);
/// @brief Define parameter for a network type
DEFINE_string(t, "C", type_message);

/// @brief Define parameter for preprocessing type
DEFINE_string(ppType, "", preprocessing_type);

/// @brief Define parameter for preprocessing size
DEFINE_int32(ppSize, 0, preprocessing_size);
DEFINE_int32(ppWidth, 0, preprocessing_width);
DEFINE_int32(ppHeight, 0, preprocessing_height);

DEFINE_bool(Czb, false, zero_background_message);

DEFINE_string(ODa, "", obj_detection_annotations_message);

DEFINE_string(ODc, "", obj_detection_classes_message);

DEFINE_string(ODsubdir, "", obj_detection_subdir_message);

/// @brief Define parameter for a type of Object Detection network
DEFINE_string(ODkind, "SSD", obj_detection_kind_message);

/// @brief Define parameter for GPU kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Define parameter for a path to CPU library with user layers <br>
/// It is an optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Define parameter for accuracy drop threshold
DEFINE_double(threshold, 1.0f, accuracy_threshold_message);

/// @brief Define path to output calibrated model
DEFINE_bool(stream_output, false, stream_output_message);

DEFINE_int32(subset, 0, number_of_pictures_message);

DEFINE_string(output, "", output_model_name);

DEFINE_string(lbl, "", labels_file_message);

DEFINE_bool(convert_fc, false, convert_fc_message);

/**
 * @brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Usage: calibration_tool [OPTION]" << std::endl << std::endl;
    std::cout << "Available options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -t <type>                 " << type_message << std::endl;
    for (int i = 0; types_descriptions[i][0] != nullptr; i++) {
        std::cout << "      -t \"" << types_descriptions[i][0] << "\" to " << types_descriptions[i][1] << std::endl;
    }
    std::cout << "    -i <path>                 " << image_message << std::endl;
    std::cout << "    -m <path>                 " << model_message << std::endl;
    std::cout << "    -lbl <path>               " << labels_file_message << std::endl;
    std::cout << "    -l <absolute_path>        " << custom_cpu_library_message << std::endl;
    std::cout << "    -c <absolute_path>        " << custom_cldnn_message << std::endl;
    std::cout << "    -d <device>               " << target_device_message << std::endl;
    std::cout << "    -b N                      " << batch_message << std::endl;
    std::cout << "    -ppType <type>            " << preprocessing_type << std::endl;
    std::cout << "    -ppSize N                 " << preprocessing_size << std::endl;
    std::cout << "    -ppWidth W                " << preprocessing_width << std::endl;
    std::cout << "    -ppHeight H               " << preprocessing_height << std::endl;
    std::cout << "    --dump                    " << dump_message << std::endl;
    std::cout << "    -subset                  " << number_of_pictures_message << std::endl;
    std::cout << "    -output <output_IR>      " << output_model_name << std::endl;
    std::cout << "    -threshold               " << accuracy_threshold_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Classification-specific options:" << std::endl;
    std::cout << "      -Czb true               " << zero_background_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Object detection-specific options:" << std::endl;
    std::cout << "      -ODkind <kind>          " << obj_detection_kind_message << std::endl;
    std::cout << "      -ODa <path>             " << obj_detection_annotations_message << std::endl;
    std::cout << "      -ODc <file>             " << obj_detection_classes_message << std::endl;
    std::cout << "      -ODsubdir <name>        " << obj_detection_subdir_message << std::endl << std::endl;

    std::cout << std::endl;
    std::cout << "    -stream_output                   " << stream_output_message << std::endl;
}

enum NetworkType {
    Undefined = -1,
    Classification,
    ObjDetection,
    RawC,
    RawOD
};

std::string strtolower(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

void SaveCalibratedIR(const std::string &originalName,
                      const std::string &outModelName,
                      const std::map<std::string, bool>& layersToInt8,
                      const InferenceEngine::NetworkStatsMap& statMap,
                      bool convertFullyConnected) {
    slog::info << "Layers profile for Int8 quantization\n";
    CNNNetReader networkReader;
    networkReader.ReadNetwork(originalName);
    if (!networkReader.isParseSuccess())THROW_IE_EXCEPTION << "cannot load a failed Model";

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(originalName)+ ".bin";
    networkReader.ReadWeights(binFileName.c_str());

    auto network = networkReader.getNetwork();
    for (auto &&layer : network) {
        if (CaselessEq<std::string>()(layer->type, "convolution")) {
            auto it = layersToInt8.find(layer->name);
            if (it != layersToInt8.end() && it->second == false) {
                layer->params["quantization_level"] = "FP32";
                std::cout << layer->name << ": " << "FP32" << std::endl;
            } else {
                layer->params["quantization_level"] = "I8";
                std::cout << layer->name << ": " << "I8" << std::endl;
            }
        } else if (CaselessEq<std::string>()(layer->type, "fullyconnected")) {
            if (!convertFullyConnected) {
                layer->params["quantization_level"] = "FP32";
                std::cout << layer->name << ": " << "FP32" << std::endl;
            } else {
                layer->params["quantization_level"] = "I8";
                std::cout << layer->name << ": " << "I8" << std::endl;
            }
        }
    }


    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)networkReader.getNetwork()).getStats(&pstats, nullptr);
    if (s == StatusCode::OK && pstats) {
        pstats->setNodesStats(statMap);
    }

    slog::info << "Write calibrated network to " << outModelName << ".(xml|bin) IR file\n";
    networkReader.getNetwork().serialize(outModelName + ".xml", outModelName + ".bin");
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

        // ---------------------------Parsing and validating input arguments--------------------------------------
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
        } else if (std::string(FLAGS_t) == "RawC") {
            netType = RawC;
        } else if (std::string(FLAGS_t) == "RawOD") {
            netType = RawOD;
        } else {
            ee << UserException(5, "Unknown network type specified (invalid -t option)");
        }

        // Checking required options
        if (FLAGS_m.empty()) ee << UserException(3, "Model file is not specified (missing -m option)");
        if (FLAGS_i.empty()) ee << UserException(4, "Images list is not specified (missing -i option)");
        if (FLAGS_d.empty()) ee << UserException(5, "Target device is not specified (missing -d option)");
        if (FLAGS_b < 0) ee << UserException(6, "Batch must be positive (invalid -b option value)");

        if (netType == ObjDetection) {
            // Checking required OD-specific options
            if (FLAGS_ODa.empty()) ee << UserException(11, "Annotations folder is not specified for object detection (missing -a option)");
            if (FLAGS_ODc.empty()) ee << UserException(12, "Classes file is not specified (missing -c option)");
        }

        if (!ee.empty()) throw ee;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------Loading plugin for Inference Engine------------------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        /** Loading the library with extensions if provided**/
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom CPU plugin layer implementations. These layers are not supported
             * by CPU, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
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
                THROW_USER_EXCEPTION(2) << "Size must be specified for preprocessing type " << FLAGS_ppType;
            }
        } else if (strtolower(FLAGS_ppType) == "resize" || FLAGS_ppType.empty()) {
            preprocessingOptions = PreprocessingOptions(false, ResizeCropPolicy::Resize);
        } else {
            THROW_USER_EXCEPTION(2) << "Unknown preprocessing type: " << FLAGS_ppType;
        }

        if (netType == Classification || netType == RawC) {
            processor = std::shared_ptr<Processor>(
                new ClassificationCalibrator(FLAGS_subset, FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_b,
                                                plugin, dumper, FLAGS_lbl, preprocessingOptions, FLAGS_Czb));
        } else if (netType == ObjDetection || netType == RawOD) {
            if (FLAGS_ODkind == "SSD") {
                processor = std::shared_ptr<Processor>(
                    new SSDObjectDetectionCalibrator(FLAGS_subset, FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b,
                                                        0.5, plugin, dumper, FLAGS_ODa, FLAGS_ODc));
/*            } else if (FLAGS_ODkind == "YOLO") {
                processor = std::shared_ptr<Processor>(
                        new YOLOObjectDetectionProcessor(FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b,
                                                         0.5, plugin, dumper, FLAGS_ODa, FLAGS_ODc));
*/
            }
        } else {
            THROW_USER_EXCEPTION(2) <<  "Unknown network type specified" << FLAGS_ppType;
        }
        if (!processor.get()) {
            THROW_USER_EXCEPTION(2) <<  "Processor pointer is invalid" << FLAGS_ppType;
        }

        auto calibrator = dynamic_cast<Int8Calibrator*>(processor.get());
        if (calibrator == nullptr) {
            THROW_USER_EXCEPTION(2) << "processor object is not instance of Int8Calibrator class";
        }

        if (netType != RawC && netType != RawOD) {
            slog::info << "Collecting accuracy metric in FP32 mode to get a baseline, collecting activation statistics" << slog::endl;
        } else {
            slog::info << "Collecting activation statistics" << slog::endl;
        }
        calibrator->collectFP32Statistic();
        shared_ptr<Processor::InferenceMetrics> pIMFP32 = processor->Process(FLAGS_stream_output);
        const auto mFP32 = dynamic_cast<const CalibrationMetrics*>(pIMFP32.get());
        if (mFP32 == nullptr) {
            THROW_USER_EXCEPTION(2) << "FP32 inference metrics object is not instance of CalibrationMetrics class";
        }
        std:: cout << "  FP32 Accuracy: " << OUTPUT_FLOATING(100.0 * mFP32->AccuracyResult) << "% " << std::endl;

        InferenceEngine::NetworkStatsMap statMap;
        std::map<std::string, bool> layersToInt8;
        bool bAccuracy = false;

        if (netType != RawC && netType != RawOD) {
            slog::info << "Verification of network accuracy if all possible layers converted to INT8" << slog::endl;
            float bestThreshold = 100.f;
            float maximalAccuracy = 0.f;
            for (float threshold = 100.0f; threshold > 95.0f; threshold -= 0.5) {
                std::cout << "Validate int8 accuracy, threshold for activation statistics = " << threshold << std::endl;
                InferenceEngine::NetworkStatsMap tmpStatMap = calibrator->getStatistic(threshold);
                calibrator->validateInt8Config(tmpStatMap, {}, FLAGS_convert_fc);
                shared_ptr<Processor::InferenceMetrics> pIM_I8 = processor->Process(FLAGS_stream_output);
                auto *mI8 = dynamic_cast<const CalibrationMetrics *>(pIM_I8.get());
                if (mI8 == nullptr) {
                    THROW_USER_EXCEPTION(2) << "INT8 inference metrics object is not instance of CalibrationMetrics class";
                }
                if (maximalAccuracy < mI8->AccuracyResult) {
                    maximalAccuracy = mI8->AccuracyResult;
                    bestThreshold = threshold;
                }
                std::cout << "   Accuracy is " << OUTPUT_FLOATING(100.0 * mI8->AccuracyResult) << "%" << std::endl;
            }

            statMap = calibrator->getStatistic(bestThreshold);

            if ((mFP32->AccuracyResult - maximalAccuracy) > (FLAGS_threshold / 100)) {
                slog::info << "Accuracy of all layers conversion does not correspond to the required threshold\n";
                cout << "FP32 Accuracy: " << OUTPUT_FLOATING(100.0 * mFP32->AccuracyResult) << "% vs " <<
                    "all Int8 layers Accuracy: " << OUTPUT_FLOATING(100.0 * maximalAccuracy) << "%, " <<
                    "threshold for activation statistics: " << bestThreshold << "%" << std::endl;
                slog::info << "Collecting intermediate per-layer accuracy drop" << slog::endl;
                // getting statistic on accuracy drop by layers
                calibrator->collectByLayerStatistic(statMap);
                processor->Process(FLAGS_stream_output);
                // starting to reduce number of layers being converted to Int8
                std::map<std::string, float>  layersAccuracyDrop = calibrator->layersAccuracyDrop();

                std::map<float, std::string> orderedLayersAccuracyDrop;
                for (auto d : layersAccuracyDrop) {
                    orderedLayersAccuracyDrop[d.second] = d.first;
                    layersToInt8[d.first] = true;
                }
                auto it = orderedLayersAccuracyDrop.crbegin();

                shared_ptr<Processor::InferenceMetrics> pIM_I8;
                const CalibrationMetrics *mI8;
                while (it != orderedLayersAccuracyDrop.crend() && bAccuracy == false) {
                    slog::info << "Returning of '" << it->second << "' to FP32 precision, start validation\n";
                    layersToInt8[it->second] = false;
                    calibrator->validateInt8Config(statMap, layersToInt8, FLAGS_convert_fc);
                    pIM_I8 = processor->Process(FLAGS_stream_output);
                    mI8 = dynamic_cast<const CalibrationMetrics *>(pIM_I8.get());
                    maximalAccuracy = mI8->AccuracyResult;
                    if ((mFP32->AccuracyResult - maximalAccuracy) > (FLAGS_threshold / 100)) {
                        cout << "FP32 Accuracy: " << OUTPUT_FLOATING(100.0 * mFP32->AccuracyResult) << "% vs " <<
                            "current Int8 configuration Accuracy: " << OUTPUT_FLOATING(100.0 * maximalAccuracy) << "%" << std::endl;
                    } else {
                        bAccuracy = true;
                    }
                    it++;
                }
            } else {
                bAccuracy = true;
            }

            if (bAccuracy) {
                slog::info << "Achieved required accuracy drop satisfying threshold\n";
                cout << "FP32 accuracy: " << OUTPUT_FLOATING(100.0 * mFP32->AccuracyResult) << "% vs " <<
                    "current Int8 configuration accuracy: " << OUTPUT_FLOATING(100.0 * maximalAccuracy) << "% " <<
                    "with threshold for activation statistic: " << bestThreshold << "%" << std::endl;
                std::string outModelName = FLAGS_output.empty() ? fileNameNoExt(FLAGS_m) + "_i8" : fileNameNoExt(FLAGS_output);
                SaveCalibratedIR(FLAGS_m, outModelName, layersToInt8, statMap, FLAGS_convert_fc);
            } else {
                slog::info << "Required threshold of accuracy drop cannot be achieved with any int8 quantization\n";
            }
        } else {
            std::cout << "Collected activation statistics, writing maximum values to IR" << std::endl;
            statMap = calibrator->getStatistic(100.0f);
            std::string outModelName = FLAGS_output.empty() ? fileNameNoExt(FLAGS_m) + "_i8" : fileNameNoExt(FLAGS_output);
            SaveCalibratedIR(FLAGS_m, outModelName, layersToInt8, statMap, FLAGS_convert_fc);
        }

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
            slog::err << "Input problems: \n" << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        }
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
