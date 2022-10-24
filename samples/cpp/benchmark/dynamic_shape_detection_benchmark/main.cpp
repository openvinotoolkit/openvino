// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <condition_variable>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "format_reader_ptr.h"
#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;
        if (argc != 3) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <path_to_image>" << slog::endl;
            return EXIT_FAILURE;
        }
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
        if (model->inputs().size() != 1) {
            throw std::runtime_error("Only modes with 1 input are supported");
        }

        std::string scores_name = "scores";
        std::vector<ov::Output<ov::Node>> outputs = model->outputs();
        // Each output may have multiple names. Check that one of them is scores_name.
        // This tensor contains confidence of each detected bounding boxes
        auto scores_out_iter =
            find_if(outputs.begin(), outputs.end(), [scores_name](const ov::Output<ov::Node>& output) {
                const std::unordered_set<std::string>& names = output.get_names();
                return names.find(scores_name) != names.end();
            });
        if (outputs.end() == scores_out_iter) {
            throw std::runtime_error("The model must have 'scores' as one of outputs");
        }
        const ov::Shape& scores_shape = scores_out_iter->get_shape();
        if (scores_shape.size() != 3) {
            throw std::runtime_error("Scores output rank must be 3");
        }
        if (scores_shape[2] != 92) {
            throw std::runtime_error("Scores output last dimension must be of size 92");
        }
        // Set dynamic input shape
        model->reshape(ov::PartialShape{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
        ov::preprocess::PrePostProcessor ppp(model);
        ov::element::Type input_type = ov::element::u8;

        // Set input tensor information:
        // - input() provides information about a single model input
        ppp.input().tensor().set_element_type(input_type).set_layout("NHWC");
        // Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout("NCHW");

        // Apply preprocessing modifying the original 'model'
        model = ppp.build();

        // Optimize for throughput. Best throughput can be reached by
        // running multiple ov::InferRequest instances asyncronously
        ov::AnyMap tput{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}};

        // Create ov::Core and use it to compile a model
        // Pick device by replacing CPU, for example MULTI:CPU(4),GPU(8)
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU", tput);

        // The image is going to be inferred multiple times, but read only once to get more stable results
        FormatReader::ReaderPtr reader(argv[2]);
        if (reader.get() == nullptr) {
            throw std::runtime_error("Image cannot be read!");
        }

        // Create optimal number of ov::InferRequest instances
        uint32_t nireq;
        try {
            // +1 to run postprocessing for one ireq while others are running
            nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests) + 1;
        } catch (const std::exception& ex) {
            throw std::runtime_error("Every used device must support " +
                                     std::string(ov::optimal_number_of_infer_requests.name()) +
                                     " Failed to query the property with error: " + ex.what());
        }
        std::vector<ov::InferRequest> ireqs;
        for (uint32_t i = 0; i < nireq; ++i) {
            ireqs.push_back(compiled_model.create_infer_request());
        }

        ov::Shape warm_up_input_shape = {1, reader->height(), reader->width(), 3};
        std::shared_ptr<unsigned char> warm_up_input_data = reader->getData();
        // just wrap image data by ov::Tensor without allocating of new memory
        ov::Tensor warm_up_input_tensor = ov::Tensor(input_type, warm_up_input_shape, warm_up_input_data.get());
        // Warm up
        for (ov::InferRequest& ireq : ireqs) {
            ireq.set_input_tensor(warm_up_input_tensor);
            ireq.start_async();
        }
        for (ov::InferRequest& ireq : ireqs) {
            ireq.wait();
        }
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{15};
        int init_niter = 12;
        int niter = ((init_niter + nireq - 1) / nireq) * nireq;
        if (init_niter != niter) {
            slog::warn << "Number of iterations was aligned by request number from " << init_niter << " to " << niter
                       << " using number of requests " << nireq << slog::endl;
        }
        size_t ndetections = 0;
        std::vector<double> latencies;
        std::mutex mutex;
        std::condition_variable cv;
        std::exception_ptr callback_exception;
        struct TimedIreq {
            ov::InferRequest& ireq;  // ref
            std::chrono::steady_clock::time_point start;
            bool has_start_time;
        };
        std::deque<TimedIreq> finished_ireqs;
        for (ov::InferRequest& ireq : ireqs) {
            finished_ireqs.push_back({ireq, std::chrono::steady_clock::time_point{}, false});
        }
        auto start = std::chrono::steady_clock::now();
        auto time_point_to_finish = start + seconds_to_run;
        // Once there’s a finished ireq wake up main thread.
        // Compute and save latency for that ireq and prepare for next inference by setting up callback.
        // Callback pushes that ireq again to finished ireqs when infrence is completed.
        // Start asynchronous infer with updated callback
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            while (!callback_exception && finished_ireqs.empty()) {
                cv.wait(lock);
            }
            if (callback_exception) {
                std::rethrow_exception(callback_exception);
            }
            if (!finished_ireqs.empty()) {
                auto time_point = std::chrono::steady_clock::now();
                if (time_point > time_point_to_finish && latencies.size() > niter) {
                    break;
                }
                TimedIreq timedIreq = finished_ireqs.front();
                finished_ireqs.pop_front();
                lock.unlock();
                ov::InferRequest& ireq = timedIreq.ireq;
                if (timedIreq.has_start_time) {
                    latencies.push_back(std::chrono::duration_cast<Ms>(time_point - timedIreq.start).count());
                }
                // Extract data and count detected objects
                ndetections = 0;
                ov::Tensor output_tensor = ireq.get_tensor(scores_name);
                float* data = output_tensor.data<float>();
                for (size_t i = 0; i < scores_shape[1]; ++i) {
                    float max_score = 0;
                    size_t max_id = 0;
                    for (size_t j = 0; j < scores_shape[2]; ++j) {
                        if (data[i * scores_shape[1] + j] > max_score) {
                            max_score = data[i * scores_shape[1] + j];
                            max_id = j;
                        }
                    }
                    float confidence_threshold = 0.5;
                    // The last class is no-object class. Filter it out
                    if (max_id != scores_shape[2] - 1 && max_score > confidence_threshold) {
                        ++ndetections;
                    }
                }
                // Prepare new inference
                ov::Shape input_shape = {1, reader->height(), reader->width(), 3};
                std::shared_ptr<unsigned char> input_data = reader->getData();
                // just wrap image data by ov::Tensor without allocating of new memory
                ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());
                ireq.set_callback(
                    // Make sure input_data is not destroyed before inference starts
                    // input_tensor doesn't own this data, thus capture input_data by value
                    [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv, input_data](
                        std::exception_ptr ex) {
                        // Keep callback small. This improves performance for fast (tens of thousands FPS) models
                        std::unique_lock<std::mutex> lock(mutex);
                        {
                            try {
                                if (ex) {
                                    std::rethrow_exception(ex);
                                }
                                finished_ireqs.push_back({ireq, time_point, true});
                            } catch (const std::exception&) {
                                if (!callback_exception) {
                                    callback_exception = std::current_exception();
                                }
                            }
                        }
                        cv.notify_one();
                    });
                ireq.start_async();
            }
        }
        auto end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        // Report results
        slog::info << "Number of detected objects: " << ndetections << slog::endl;

        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl;
        slog::info << "Duration:   " << duration << " ms" << slog::endl;
        slog::info << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics latency_metrics{latencies, "", percent};
        latency_metrics.write_to_slog();
        slog::info << "Throughput: " << double_to_string(1000 * latencies.size() / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
