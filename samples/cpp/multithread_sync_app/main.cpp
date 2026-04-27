#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_cpu/properties.hpp>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <thread>
#include <vector>
#ifdef __linux__
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

static void pin_thread_to_core(size_t core_index) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<int>(core_index), &cpuset);
    auto native_handle = pthread_self();
    pthread_setaffinity_np(native_handle, sizeof(cpu_set_t), &cpuset);
#endif
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <num_threads> [-infer_precision <i8|u8|f32|bf16>] [-shape [1,64]] [-layout [NW]] [-core_base <N>]\n";
        std::cerr << "Purpose: multi-thread synchronous CPU inference benchmark with one compiled model and one infer request per app thread.\n";
        std::cerr << "  -core_base <N>  Starting core index for thread pinning (default: 0). Thread i is pinned to core (core_base + i).\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const int num_threads = std::stoi(argv[2]);
    if (num_threads <= 0) {
        std::cerr << "num_threads must be > 0\n";
        return 1;
    }

    // Optional args for multithread_sync_app:
    // -infer_precision <i8|u8|f32|bf16> -shape [1,64] -layout [NW]
    std::string shape_arg, layout_arg, infer_precision_arg;
    int core_base = 0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-core_base") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -core_base\n";
                return 1;
            }
            core_base = std::stoi(argv[++i]);
            if (core_base < 0) {
                std::cerr << "core_base must be >= 0\n";
                return 1;
            }
        } else if (a == "-shape") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -shape\n";
                return 1;
            }
            shape_arg = argv[++i];
        } else if (a == "-infer_precision") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -infer_precision\n";
                return 1;
            }
            infer_precision_arg = argv[++i];
        } else if (a == "-layout") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -layout\n";
                return 1;
            }
            layout_arg = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return 1;
        }
    }

    auto trim = [](std::string s) {
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c){ return std::isspace(c); }), s.end());
        if (!s.empty() && s.front() == '[') s.erase(s.begin());
        if (!s.empty() && s.back() == ']') s.pop_back();
        return s;
    };

    ov::Shape user_shape;
    if (!shape_arg.empty()) {
        std::string s = trim(shape_arg);
        if (s.empty()) {
            std::cerr << "Invalid -shape value\n";
            return 1;
        }
        std::vector<size_t> dims;
        size_t start = 0;
        while (start < s.size()) {
            size_t pos = s.find(',', start);
            std::string tok = s.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
            if (tok.empty()) { std::cerr << "Invalid -shape value\n"; return 1; }
            try {
                long long v = std::stoll(tok);
                if (v <= 0) { std::cerr << "Shape dims must be > 0\n"; return 1; }
                dims.push_back(static_cast<size_t>(v));
            } catch (...) {
                std::cerr << "Invalid -shape value token: " << tok << "\n";
                return 1;
            }
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        user_shape = ov::Shape{dims};
    }

    std::string user_layout;
    if (!layout_arg.empty()) {
        user_layout = trim(layout_arg);
        if (user_layout.empty()) {
            std::cerr << "Invalid -layout value\n";
            return 1;
        }
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to read model: " << e.what() << "\n";
        return 2;
    }

    // Apply shape/layout to dynamic model (first input)
    if (!user_shape.empty() || !user_layout.empty()) {
        ov::preprocess::PrePostProcessor ppp(model);
        auto& inp = ppp.input(0);
        if (!user_layout.empty()) {
            inp.tensor().set_layout(ov::Layout(user_layout));
            inp.model().set_layout(ov::Layout(user_layout));
        }
        if (!user_shape.empty()) {
            inp.tensor().set_shape(user_shape);
        }
        try {
            model = ppp.build();
        } catch (const std::exception& e) {
            std::cerr << "Failed to apply shape/layout: " << e.what() << "\n";
            return 2;
        }
    }

    ov::AnyMap cpu_props = {
        {ov::num_streams.name(), num_threads},
        {ov::intel_cpu::multi_app_thread_sync_execution.name(), true},
        {ov::hint::enable_cpu_pinning.name(), true},
        {ov::enable_profiling.name(), false}
    };

    if (!infer_precision_arg.empty()) {
        // Accept common precision strings like i8, u8, f32, bf16
        // Forward as string; OpenVINO will validate and convert.
        cpu_props.emplace(ov::hint::inference_precision.name(), infer_precision_arg);
    }

    ov::CompiledModel compiled;
    try {
        compiled = core.compile_model(model, "CPU", cpu_props);
    } catch (const std::exception& e) {
        std::cerr << "Failed to compile model on CPU: " << e.what() << "\n";
        return 3;
    }

    // Prepare one infer request per application thread (single compiled model).
    std::vector<ov::InferRequest> requests;
    requests.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        requests.push_back(compiled.create_infer_request());
    }

    // Prepare input tensors once and set them to all requests.
    auto inputs = compiled.inputs();
    for (auto& req : requests) {
        for (const auto& port : inputs) {
            const ov::element::Type elem_type = port.get_element_type();
            const ov::Shape shape = port.get_shape();
            ov::Tensor tensor(elem_type, shape);
            // Fill with zeros for deterministic behavior
            std::memset(tensor.data(), 0, tensor.get_byte_size());
            req.set_tensor(port, tensor);
        }
    }

    const uint64_t iters_per_thread = 1000000ULL;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> thread_durations(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        thread_durations[static_cast<size_t>(i)].reserve(static_cast<size_t>(iters_per_thread));
    }

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            pin_thread_to_core(static_cast<size_t>(core_base + i));
            auto& req = requests[i];
            auto& local = thread_durations[static_cast<size_t>(i)];
            for (uint64_t k = 0; k < iters_per_thread; ++k) {
                auto s = std::chrono::steady_clock::now();
                req.infer();
                auto e = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(e - s).count();
                local.push_back(us);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double total_iters = static_cast<double>(iters_per_thread) * static_cast<double>(num_threads);
    const double tps = total_iters / seconds;

    // Aggregate per-iteration durations across all threads.
    std::vector<double> all_us;
    all_us.reserve(static_cast<size_t>(iters_per_thread) * static_cast<size_t>(num_threads));
    double sum_us = 0.0;
    for (const auto& v : thread_durations) {
        all_us.insert(all_us.end(), v.begin(), v.end());
        sum_us = std::accumulate(v.begin(), v.end(), sum_us);
    }
    const size_t N = all_us.size();
    double avg_us = N ? (sum_us / static_cast<double>(N)) : 0.0;

    // Percentiles (ceil rank - 1), using nth_element
    double p90_us = 0.0, p99_us = 0.0;
    if (N > 0) {
        size_t idx90 = (N * 90 + 100 - 1) / 100; if (idx90 == 0) idx90 = 1; idx90 -= 1;
        size_t idx99 = (N * 99 + 100 - 1) / 100; if (idx99 == 0) idx99 = 1; idx99 -= 1;
        std::nth_element(all_us.begin(), all_us.begin() + static_cast<std::ptrdiff_t>(idx90), all_us.end());
        p90_us = all_us[idx90];
        std::nth_element(all_us.begin(), all_us.begin() + static_cast<std::ptrdiff_t>(idx99), all_us.end());
        p99_us = all_us[idx99];
    }

    std::cout << "Total iterations: " << total_iters << "\n";
    std::cout << "Elapsed seconds: " << seconds << "\n";
    std::cout << "Throughput (iters/sec): " << tps << "\n";
    std::cout << "Per-iteration avg (us): " << avg_us << "\n";
    std::cout << "Per-iteration p90 (us): " << p90_us << "\n";
    std::cout << "Per-iteration p99 (us): " << p99_us << "\n";

    return 0;
}
