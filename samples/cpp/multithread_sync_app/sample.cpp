#include <openvino/openvino.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>
#include <chrono>

static void print_usage(const char* exe) {
    std::cout << "Usage: " << exe << " -m <model_path.xml|.onnx> [--shape 1,3,224,224] [--layout NCHW|NHWC|...] [-d CPU|GPU]\n";
}

static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (flag == argv[i]) return true;
    return false;
}

static const char* get_opt(int argc, char** argv, const std::string& flag1, const std::string& flag2 = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (flag1 == argv[i] || (!flag2.empty() && flag2 == argv[i])) return argv[i + 1];
    }
    return nullptr;
}

static std::vector<size_t> parse_shape(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) if (c != '[' && c != ']') t.push_back(c);
    std::vector<size_t> dims;
    std::stringstream ss(t);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        size_t v = static_cast<size_t>(std::stoll(item));
        dims.push_back(v);
    }
    return dims;
}

static void fill_random_float(ov::Tensor& tensor, float low = 0.f, float high = 1.f) {
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(low, high);
    auto* data = tensor.data<float>();
    size_t count = tensor.get_size();
    for (size_t i = 0; i < count; ++i) data[i] = dist(rng);
}

int main(int argc, char** argv) {
    if (!has_flag(argc, argv, "-m") && !has_flag(argc, argv, "--model")) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = get_opt(argc, argv, "-m", "--model");
    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    const char* shape_opt  = get_opt(argc, argv, "--shape");
    const char* layout_opt = get_opt(argc, argv, "--layout");
    const char* device_opt = get_opt(argc, argv, "-d", "--device");
    std::string device     = device_opt ? device_opt : "CPU";
    std::vector<size_t> user_shape = shape_opt ? parse_shape(shape_opt) : std::vector<size_t>{};
    std::string layout_str = layout_opt ? std::string(layout_opt) : std::string();

    // Parse niter argument (-n or --niter). If >1, run timed loop and exit early.
    const char* niter_opt = get_opt(argc, argv, "-n", "--niter");
    int niter = 1;
    if (niter_opt) {
        try {
            niter = std::max(1, std::stoi(niter_opt));
        } catch (...) {
            std::cerr << "Invalid niter value.\n";
            return 1;
        }
    }

    if (niter > 1) {
        try {
            ov::Core ie;

            try {
                ov::element::Type et = ie.get_property(device, ov::hint::inference_precision);
                std::cout << " init default before read - inference_precision: " << et.get_type_name() << "\n";
            } catch (...) {}

            auto net = ie.read_model(model_path);

            if (net->inputs().empty()) {
                std::cerr << "Model has no inputs.\n";
                return 2;
            }

            auto in0 = net->input(0);
            if (!user_shape.empty()) {
                net->reshape({{net->input(0), ov::PartialShape(user_shape)}});
            } else if (in0.get_partial_shape().is_dynamic()) {
                std::cerr << "Model input shape is dynamic. Provide --shape, e.g., --shape 1,3,224,224\n";
                return 3;
            }

            ov::preprocess::PrePostProcessor prep(net);
            prep.input().tensor().set_element_type(ov::element::f32);
            if (!layout_str.empty()) {
                ov::Layout L(layout_str);
                prep.input().tensor().set_layout(L);
                prep.input().model().set_layout(L);
            }
            net = prep.build();

            ov::Shape shape = user_shape.empty()
                                ? net->input(0).get_shape()
                                : ov::Shape(user_shape.begin(), user_shape.end());

            ov::Tensor in_tensor{ov::element::f32, shape};
            fill_random_float(in_tensor);

            // Print element types and friendly names for all nodes (per output)

            std::cout << "Model nodes and their output element types before compilation:\n";
            for (const auto& node : net->get_ops()) {
                const auto& name = node->get_friendly_name();
                for (size_t i = 0; i < node->get_output_size(); ++i) {
                    ov::element::Type et = node->get_output_element_type(i);
                    std::cout << et.get_type_name() << " " << name << "\n";
                }
            }

            // inference_precision
            try {
                ov::element::Type et = ie.get_property(device, ov::hint::inference_precision);
                std::cout << " default - inference_precision: " << et.get_type_name() << "\n";
            } catch (...) {}

            // Set inference precision based on --infer_precision [f32|f16|bf16]
            {
                const char* infer_precision_opt = get_opt(argc, argv, "--infer_precision");
                
                if (infer_precision_opt) {
                    ov::element::Type infer_et = ov::element::f32;
                    std::string ip = infer_precision_opt;
                    std::transform(ip.begin(), ip.end(), ip.begin(), ::tolower);
                    if (ip == "f16") {
                        infer_et = ov::element::f16;
                    } else if (ip == "bf16") {
                        infer_et = ov::element::bf16;
                    } else if (ip == "f32") {
                        infer_et = ov::element::f32;
                    } else {
                        std::cerr << "Unknown infer_precision: " << ip << " (use f32, f16, or bf16). Falling back to f32.\n";
                    }
                    ie.set_property(device, ov::hint::inference_precision(infer_et));
                }
                
            }

            ie.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
            // If --perf is given, set CPU execution mode to PERFORMANCE
            if (has_flag(argc, argv, "--perf")) {
                ie.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
            }


            // Print all available config hints for the selected device
            std::cout << "Configuration hints for device " << device << ":\n";




            // inference_precision
            try {
                ov::element::Type et = ie.get_property(device, ov::hint::inference_precision);
                std::cout << " - inference_precision: " << et.get_type_name() << "\n";
            } catch (...) {}
            // num_requests
            try {
                int nr = ie.get_property(device, ov::hint::num_requests);
                std::cout << " - num_requests: " << nr << "\n";
            } catch (...) {}
            // model_priority
            try {
                auto pr = ie.get_property(device, ov::hint::model_priority);
                const char* ps = "UNKNOWN";
                switch (pr) {
                    case ov::hint::Priority::LOW:    ps = "LOW"; break;
                    case ov::hint::Priority::MEDIUM: ps = "MEDIUM"; break;
                    case ov::hint::Priority::HIGH:   ps = "HIGH"; break;
                }
                std::cout << " - model_priority: " << ps << "\n";
            } catch (...) {}
            // execution_mode (newer API)
            try {
                auto em = ie.get_property(device, ov::hint::execution_mode);
                const char* es = "UNKNOWN";
                switch (em) {
                    case ov::hint::ExecutionMode::PERFORMANCE: es = "PERFORMANCE"; break;
                    case ov::hint::ExecutionMode::ACCURACY:    es = "ACCURACY"; break;
                }
                std::cout << " - execution_mode: " << es << "\n";
            } catch (...) {}
            // performance_mode (older API, if available)
            try {
                auto pm = ie.get_property(device, ov::hint::performance_mode);
                const char* ps = "UNKNOWN";
                switch (pm) {
                    case ov::hint::PerformanceMode::LATENCY:              ps = "LATENCY"; break;
                    case ov::hint::PerformanceMode::THROUGHPUT:           ps = "THROUGHPUT"; break;
                    case ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT: ps = "CUMULATIVE_THROUGHPUT"; break;
                }
                std::cout << " - performance_mode: " << ps << "\n";
            } catch (...) {}

            auto exec = ie.compile_model(net, device);



            std::cout << "Model nodes and their output element types after compilation:\n";
            auto runtime = exec.get_runtime_model();
            for (const auto& node : runtime->get_ops()) {
                const auto& name = node->get_friendly_name();
                for (size_t i = 0; i < node->get_output_size(); ++i) {
                    ov::element::Type et = node->get_output_element_type(i);
                    std::cout << et.get_type_name() << " " << name << "\n";
                }
            }


            auto req  = exec.create_infer_request();
            req.set_input_tensor(in_tensor);

            // Optional warmup
            req.infer();

            using clk = std::chrono::high_resolution_clock;
            std::chrono::duration<double, std::micro> total{0};
            for (int i = 0; i < niter; ++i) {
                auto t0 = clk::now();
                req.infer();
                auto t1 = clk::now();
                total += (t1 - t0);
            }

            double avg_us = total.count() / static_cast<double>(niter);
            std::cout << "Average infer time over " << niter << " runs: " << avg_us << " microseconds\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 10;
        }
        return 0;
    }


    try {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        // Use the first input by default
        if (model->inputs().size() == 0) {
            std::cerr << "Model has no inputs.\n";
            return 2;
        }
        ov::Output<const ov::Node> input_port = model->input(0);

        // If user provided shape, try to reshape model
        if (!user_shape.empty()) {
            ov::PartialShape ps(user_shape);
            std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
            new_shapes[model->input(0)] = ps;
            model->reshape(new_shapes);
        } else {
            // If model input shape is dynamic and no --shape provided, fail
            if (input_port.get_partial_shape().is_dynamic()) {
                std::cerr << "Model input shape is dynamic. Provide --shape, e.g., --shape 1,3,224,224\n";
                return 3;
            }
        }

        // Build preprocess: use float32 random tensor and optional layout
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32);
        if (!layout_str.empty()) {
            ov::Layout L(layout_str);
            ppp.input().tensor().set_layout(L);
            ppp.input().model().set_layout(L);
        }
        model = ppp.build();

        // Determine final input shape for tensor allocation
        ov::Shape in_shape;
        if (!user_shape.empty()) {
            in_shape = ov::Shape(user_shape.begin(), user_shape.end());
        } else {
            in_shape = model->input(0).get_shape();
        }

        // Allocate and fill random input
        ov::Tensor input_tensor{ov::element::f32, in_shape};
        fill_random_float(input_tensor, 0.f, 1.f);

        // Compile and run
        ov::CompiledModel compiled = core.compile_model(model, device);
        ov::InferRequest infer_req = compiled.create_infer_request();
        infer_req.set_input_tensor(input_tensor);
        infer_req.infer();

        // Report output info
        auto output = infer_req.get_output_tensor(0);
        std::cout << "Inference done.\n";
        std::cout << "Input shape: ";
        for (size_t i = 0; i < in_shape.size(); ++i) {
            std::cout << in_shape[i] << (i + 1 < in_shape.size() ? "x" : "");
        }
        std::cout << "\n";
        std::cout << "Output shape: ";
        auto out_shape = output.get_shape();
        for (size_t i = 0; i < out_shape.size(); ++i) {
            std::cout << out_shape[i] << (i + 1 < out_shape.size() ? "x" : "");
        }
        std::cout << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 10;
    }
    return 0;
}