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
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov4/main.cpp
* \example object_detection_demo_yolov4/main.cpp
*/

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "object_detection_demo_yolov4.h"
#include "math.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    double xmin, ymin, xmax, ymax;
    int class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = (x - w / 2) * w_scale;
        this->ymin = (y - h / 2) * h_scale;
        this->xmax = this->xmin + w * w_scale;
        this->ymax = this->ymin + h * h_scale;

        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator<(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

typedef struct box {
    double x, y, w, h;
} box;

typedef struct boxabs {
    double left, right, top, bot;
} boxabs;

double overlap(double x1, double w1, double x2, double w2) {
    double l1 = x1 - w1 / 2;
    double l2 = x2 - w2 / 2;
    double left = l1 > l2 ? l1 : l2;
    double r1 = x1 + w1 / 2;
    double r2 = x2 + w2 / 2;
    double right = r1 < r2 ? r1 : r2;
    return right - left;
}

double box_intersection(box a, box b) {
    double w = overlap(a.x, a.w, b.x, b.w);
    double h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    double area = w * h;
    return area;
}

double box_union(box a, box b) {
    double i = box_intersection(a, b);
    double u = a.w * a.h + b.w * b.h - i;
    return u;
}

double box_iou(box a, box b) {
    //return box_intersection(a, b) / box_union(a, b);
    double I = box_intersection(a, b);
    double U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

double box_diou(const DetectionObject &box_1, const DetectionObject &box_2) {
     box a = {box_1.xmin * 1.0, box_1.ymin * 1.0, (box_1.xmax - box_1.xmin) * 1.0, (box_1.ymax - box_1.ymin) * 1.0};
     box b = {box_2.xmin * 1.0, box_2.ymin * 1.0, (box_2.xmax - box_2.xmin) * 1.0, (box_2.ymax - box_2.ymin) * 1.0};
     boxabs ba = { 0 };
     ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
     ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
     ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
     ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
     double w = ba.right - ba.left;
     double h = ba.bot - ba.top;
     double c = w * w + h * h;
     double iou = box_iou(a, b);
     if (c == 0) {
         return iou;
     }
     double d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
     double u = pow(d / c, 0.6);
     double diou_term = u;
 #ifdef DEBUG_PRINTS
     printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
 #endif
     return iou - diou_term;
}

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap + 1e-09;
    return area_of_overlap / area_of_union;
}

float sigmoid(float x) {
    return (1 / (1 + expf(-x)));
}

void ParseYOLOV4Output(const Blob::Ptr &blob,
                       const unsigned long resized_im_h,
                       const unsigned long resized_im_w,
                       const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold,
                       std::vector<DetectionObject> &objects) {
    // --------------------------- Extracting layer parameters -------------------------------------
    const int num = 3;
    const int coords = 4;
    const int classes = 80;

    std::vector<float> anchors = {
        12.0f, 16.0f, 19.0f, 36.0f, 40.0f, 28.0f, 36.0f, 75.0f, 76.0f, 55.0f, 72.0f, 146.0f, 142.0f, 110.0f, 192.0f, 243.0f, 459.0f, 401.0f
    };

    float scale_x_y;
    auto shape = blob->getTensorDesc().getDims();
    const int output_blob_w = shape[2];
    const int output_blob_h = shape[3];
    std::vector<int> mask;
    if (output_blob_w == 40 || output_blob_w == 52 || output_blob_w == 76) {
        mask = {0, 1, 2};
        scale_x_y = 1.2f;
    } else if (output_blob_w == 20 || output_blob_w == 26 || output_blob_w == 38) {
        mask = {3, 4, 5};
        scale_x_y = 1.1f;
    } else if (output_blob_w == 10 || output_blob_w == 13 || output_blob_w == 19) {
        mask = {6, 7, 8};
        scale_x_y = 1.05f;
    } else {
        throw std::logic_error("Unknow output_blob_w");
    }

    std::vector<int> masked_anchors(18);
    for (int i = 0; i < num; i++) {
        masked_anchors[i * 2] = static_cast<int>(anchors[mask[i] * 2]);
        masked_anchors[i * 2 + 1] = static_cast<int>(anchors[mask[i] * 2 + 1]);
    }

    auto side = output_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = sigmoid(output_blob[obj_index]);
            if (scale < threshold)
                continue;

            float x = static_cast<float>((col + sigmoid(output_blob[box_index + 0 * side_square]) * scale_x_y - 0.5 * (scale_x_y - 1)) / output_blob_w);
            float y = static_cast<float>((row + sigmoid(output_blob[box_index + 1 * side_square]) * scale_x_y - 0.5 * (scale_x_y - 1)) / output_blob_h);
            float height = std::exp(output_blob[box_index + 3 * side_square]) * masked_anchors[2 * n + 1] / resized_im_h;
            float width  = std::exp(output_blob[box_index + 2 * side_square]) * masked_anchors[2 * n] / resized_im_w;
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
                float prob = scale * sigmoid(output_blob[class_index]);
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(resized_im_h),
                        static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void calloc_error() {
    fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
    exit(EXIT_FAILURE);
}

void *xcalloc(size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (!ptr) {
        calloc_error();
    }
    memset(ptr, 0, nmemb * size);
    return ptr;
}

typedef struct Image {
    int w;
    int h;
    int c;
    float *data;
} Image;

Image copy_image(Image p) {
    Image copy = p;
    copy.data = reinterpret_cast<float*> (xcalloc(p.h * p.w * p.c, sizeof(float)));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

Image make_empty_image(int w, int h, int c) {
    Image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

Image make_image(int w, int h, int c) {
    Image out = make_empty_image(w, h, c);
    out.data = reinterpret_cast<float*> (xcalloc(h * w * c, sizeof(float)));
    return out;
}

static float get_pixel(Image m, int x, int y, int c) {
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}

static void set_pixel(Image m, int x, int y, int c, float val) {
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] = val;
}

static void add_pixel(Image m, int x, int y, int c, float val) {
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] += val;
}

void free_image(Image m) {
    if (m.data) {
        free(m.data);
    }
}

Image mat_to_image(cv::Mat mat) {
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    Image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
            }
        }
    }
    return im;
}

cv::Mat image_to_mat(Image img) {
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c * img.h * img.w + y * img.w + x];
                mat.data[y * step + x * img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}

Image resize_image(Image im, int w, int h) {
    if (im.w == w && im.h == h) return copy_image(im);

    Image resized = make_image(w, h, im.c);
    Image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = static_cast<float> ((im.w - 1) / (w - 1));
    float h_scale = static_cast<float> ((im.h - 1) / (h - 1));
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w-1 || im.w == 1) {
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c * w_scale;
                    int ix = static_cast<int> (sx);
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r * h_scale;
            int iy = static_cast<int> (sy);
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h-1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

cv::Mat ReadImage(const std::string& imageName, int IH, int IW, int* srcw, int* srch) {
    cv::Mat image = cv::imread(imageName, cv::IMREAD_COLOR);

    *srcw = image.size().width;
    *srch = image.size().height;
    if (IH == image.size().height && IW == image.size().width) {
         return image;
    }

    Image dimage = mat_to_image(image);
    Image rimage = resize_image(dimage, IW, IH);
    cv::Mat simage = image_to_mat(rimage);

    return simage;
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /**Loading extensions to the devices **/
        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        slog::info << "Loading network files:"
                "\n\t" << FLAGS_m <<
                "\n\t" << binFileName <<
        slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        /** Reading labels (if specified) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        /** Extracting model name and loading weights **/
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        auto inputInfoItem = *inputInfo.begin();
        auto inputName = inputInfo.begin()->first;
        int IC = inputInfoItem.second->getTensorDesc().getDims()[1];
        int IH = inputInfoItem.second->getTensorDesc().getDims()[2];
        int IW = inputInfoItem.second->getTensorDesc().getDims()[3];
        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);

        int srcw = 0;
        int srch = 0;
        cv::Mat simage = cv::imread(FLAGS_i, cv::IMREAD_COLOR);
        cv::Mat image = ReadImage(FLAGS_i, IH, IW, &srcw, &srch);
        /** Setting batch size using image count **/
        network.setBatchSize(1);
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }
        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;

        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);
            /** Filling input tensor with images. First b channel, then g and r channels **/
            auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            for (int ch = 0; ch < IC; ch++) {
                for (int row = 0; row < IH; row++) {
                    for (int col = 0; col < IW; col++) {
                        int dst_index = col + row * IW + ch * IH * IW;
                        data[dst_index] = image.at<cv::Vec3b>(row, col)[ch];
                        data[dst_index] = static_cast<float>(data[dst_index] / 255.0);
                    }
                }
            }
        }
        inputInfo = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        /** Start inference **/
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;
        std::vector<DetectionObject> objects;

        // Parsing outputs
        for (auto &output : outputInfo) {
            auto output_name = output.first;
            Blob::Ptr blob = infer_request.GetBlob(output_name);
            ParseYOLOV4Output(blob, IH, IW, srch, srcw, FLAGS_t, objects);
        }

        // Filtering overlapping boxes
        std::sort(objects.begin(), objects.end());
        for (unsigned int i = 0; i < objects.size(); ++i) {
            if (objects[i].confidence == 0)
                continue;
            for (unsigned int j = i + 1; j < objects.size(); ++j)
                //if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                if (box_diou(objects[i], objects[j]) >= FLAGS_iou_t && objects[i].class_id == objects[j].class_id)
                    objects[j].confidence = 0;
        }

         // Drawing boxes
        for (auto &object : objects) {
            if (object.confidence < FLAGS_t)
                continue;
            auto label = object.class_id;
            float confidence = object.confidence;
            if (confidence > FLAGS_t) {
                if (object.xmin < 0) object.xmin = 0;
                if (object.ymin < 0) object.ymin = 0;
                if (object.xmax > IW) object.xmax = IW;
                if (object.ymax > IH) object.ymax = IH;

                float bxmin = static_cast<float>((object.xmin) * srcw / IW);
                float bymin = static_cast<float>((object.ymin) * srch / IH);
                float bxmax = static_cast<float>((object.xmax) * srcw / IW);
                float bymax = static_cast<float>((object.ymax) * srch / IH);

                std::string str_label = label < static_cast<int>(labels.size()) ? labels[label] : std::string("label #") + std::to_string(label);
                std::cout << "[" << str_label << "] element, prob = " << confidence <<
                            "    (" << bxmin << "," << bymin << ")-(" << bxmax << "," << bymax << ")"
                            << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                /** Drawing only objects with confidence above threshold value **/
                //std::ostringstream conf;
                //conf << ":" << std::fixed << std::setprecision(3) << confidence;
                /*cv::putText(simage,
                        (label < static_cast<int>(labels.size()) ?
                                labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                            cv::Point2f(static_cast<float>(bxmin), static_cast<float>(bymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                            cv::Scalar(0, 0, 255));*/
                cv::rectangle(simage, cv::Point2f(static_cast<float>(bxmin), static_cast<float>(bymin)),
                        cv::Point2f(static_cast<float>(bxmax), static_cast<float>(bymax)), cv::Scalar(0, 0, 255));
            }
        }

        cv::imwrite("object_detection_demo_yolov4_output.jpg", simage);
        cv::waitKey(0);
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
