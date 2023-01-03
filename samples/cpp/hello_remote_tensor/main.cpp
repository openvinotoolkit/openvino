// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <condition_variable>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
#include "samples/remote_tensors_helper.hpp"
#ifdef HAVE_DEVICE_MEM_SUPPORT
#    include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#    include <openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp>
#endif
// clang-format on

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

std::vector<std::string> parseDevices(std::string device) {
    size_t pos = 0;
    std::vector<std::string> devicelist;
    if (device.find("MULTI") == 0) {
        auto pos = device.find_first_of(":");
        auto substring = device.substr(pos + 1);
        while ((pos = substring.find(",")) != std::string::npos) {
            devicelist.push_back(substring.substr(0, pos));
            substring.erase(0, pos + 1);
        }
        if (!substring.empty())
            devicelist.push_back(substring);
    } else {
        if ((pos = device.find(",")) != std::string::npos) {
            slog::info << "will only load to " << device.substr(0, pos);
            devicelist.push_back(device.substr(0, pos));
        } else {
            devicelist.push_back(device);
        }
    }
    return devicelist;
}
int main(int argc, char* argv[]) {
    try {
        slog::info << "OpenVINO:" << slog::endl;
        slog::info << ov::get_openvino_version();
        if (argc != 3) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << argv[1]
                       << "target device, like GPU or MULTI:GPU.0,GPU.1" << slog::endl;
            return EXIT_FAILURE;
        }
        // parse the devices
        std::string device = argv[2];
        std::vector<std::string> hwtargets = parseDevices(device);
        if (hwtargets.size() == 1 && hwtargets.back().find("CPU") != std::string::npos) {
            slog::info << "for remote tensor usage, does not support CPU only" << slog::endl;
            return EXIT_FAILURE;
        }
        bool isMulti = device.find("MULTI") != std::string::npos;
        // Create ov::Core and use it to compile a model.
        // Pick a device by replacing CPU, for example MULTI:CPU(4),GPU(8).
        // It is possible to set CUMULATIVE_THROUGHPUT as ov::hint::PerformanceMode for AUTO device
        ov::Core core;
        // construct remote context for GPU
        // inference using remote tensor
        auto ocl_instance = std::make_shared<gpu::OpenCL>();
        // ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);
        cl_int err;
        std::vector<ov::RemoteContext> remote_contexts;
        for (auto iter = hwtargets.begin(); iter != hwtargets.end();) {
            if ((*iter).find("CPU") != std::string::npos) {
                iter++;
                continue;
            }
            std::string deviceid = "0";
            auto pos = (*iter).find('.');
            if (pos != std::string::npos) {
                deviceid = (*iter).substr(pos + 1, (*iter).size());
            }
            try {
                auto remote_context =
                    ov::intel_gpu::ocl::ClContext(core, ocl_instance->_context.get(), std::stoi(deviceid));
                remote_contexts.push_back(remote_context);
                iter++;
            } catch (...) {
                slog::info << "create context failed for target " << *iter << ", remove from target list" << slog::endl;
                hwtargets.erase(iter);
            }
        }
        auto model = core.read_model(argv[1]);

        ov::RemoteContext multi_context;
        if (isMulti) {
            ov::AnyMap context_list;
            for (auto& iter : remote_contexts) {
                context_list.insert({iter.get_device_name(), iter});
            }
            multi_context = core.create_context("MULTI", context_list);
        }
        ov::AnyMap loadConfig = {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)};
        if (isMulti) {
            std::string devicepriority;
            for (auto iter = hwtargets.begin(); iter != hwtargets.end(); iter++) {
                devicepriority += *iter;
                devicepriority += ((iter + 1) == hwtargets.end()) ? "" : ",";
            }
            loadConfig.insert({ov::device::priorities(devicepriority)});
        }
        ov::CompiledModel compiled_model =
            core.compile_model(model, isMulti ? multi_context : remote_contexts.back(), loadConfig);
        // Create optimal number of ov::InferRequest instances
        uint32_t nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        std::vector<ov::InferRequest> ireqs(nireq);
        std::generate(ireqs.begin(), ireqs.end(), [&] {
            return compiled_model.create_infer_request();
        });
        // Fill input data for ireqs
        std::map<std::string, ov::TensorVector> remoteTensors;
        for (ov::InferRequest& ireq : ireqs) {
            for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
                auto in_size = ov::shape_size(model_input.get_shape()) * model_input.get_element_type().bitwidth() / 8;
                // Allocate shared buffers for input and output data which will be set to infer request
                cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
                void* mappedPtr = ocl_instance->_queue.enqueueMapBuffer(shared_input_buffer,
                                                                        CL_TRUE,
                                                                        CL_MEM_READ_WRITE,
                                                                        0,
                                                                        (cl::size_type)in_size);
                auto createdContext = remote_contexts.back().as<ov::intel_gpu::ocl::ClContext>();
                auto createdTensor = createdContext.create_tensor(model_input.get_element_type(),
                                                                  model_input.get_shape(),
                                                                  shared_input_buffer);
                auto hosttensor = ov::Tensor(model_input.get_element_type(), model_input.get_shape(), mappedPtr);
                fill_tensor_random(hosttensor);
                ocl_instance->_queue.enqueueUnmapMemObject(shared_input_buffer, mappedPtr);
                remoteTensors[model_input.get_any_name()].push_back(createdTensor);
                ireq.set_tensor(model_input.get_any_name(), remoteTensors[model_input.get_any_name()].back());
            }
        }

        // Warm up
        for (ov::InferRequest& ireq : ireqs) {
            ireq.start_async();
        }
        for (ov::InferRequest& ireq : ireqs) {
            ireq.wait();
        }
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{10};
        size_t niter = 10;
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
                ireq.set_callback(
                    [&ireq, time_point, &mutex, &finished_ireqs, &callback_exception, &cv](std::exception_ptr ex) {
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
        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl
                   << "Duration:   " << duration << " ms" << slog::endl
                   << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics{latencies, "", percent}.write_to_slog();
        slog::info << "Throughput: " << double_to_string(1000 * latencies.size() / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
