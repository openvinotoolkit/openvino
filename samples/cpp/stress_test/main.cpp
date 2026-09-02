// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// stress_test -- OpenVINO runtime stress tester
//
// Exercises: repeated load/unload, parallel inference (each thread owns its
// InferRequest), concurrent model loading while inference is ongoing,
// import/export round-trip, new inference starting while another is running
// (via cancel), and simultaneous multi-model memory pressure.
//
// Inference follows the synchronous infer_req.infer() pattern used by aicore:
//   set_tensor(name, tensor) -> infer() -> get_tensor(name)
//
// Works with LLM (input_ids/attention_mask), non-LLM (BERT, audio, image
// encoder), and classic vision models (classification, detection).

#include <atomic>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "openvino/openvino.hpp"

#if defined(_WIN32)
#    include <direct.h>
#else
#    include <sys/stat.h>
#endif

// Global stop (SIGINT)

static std::atomic<bool> g_stop{false};
static void on_signal(int) {
    g_stop = true;
}

// Per-scenario statistics

struct Stats {
    std::atomic<uint64_t> pass{0};
    std::atomic<uint64_t> fail{0};
    std::atomic<uint64_t> cancelled{0};
    mutable std::mutex mtx;
    std::vector<std::string> errors;

    void record_error(const std::string& msg) {
        std::lock_guard<std::mutex> lk(mtx);
        if (errors.size() < 64)
            errors.push_back(msg);
    }

    void print(const std::string& label) const {
        std::cout << "\n[" << label << "]"
                  << "  pass=" << pass.load() << "  fail=" << fail.load() << "  cancelled=" << cancelled.load() << "\n";
        std::lock_guard<std::mutex> lk(mtx);
        for (const auto& e : errors)
            std::cout << "  ERR: " << e << "\n";
    }
};

// CLI options

struct Options {
    std::string model_path;
    std::string device = "CPU";
    int threads = 4;
    int iterations = 50;
    int duration_s = 0;  // 0 = use iterations
    // which scenarios to run (all by default)
    bool run_load_unload = true;
    bool run_parallel_infer = true;
    bool run_concurrent = true;
    bool run_import_export = true;
    bool run_mid_flight = true;
    bool run_memory_stress = true;
    bool run_destroy_live_req = true;
    bool run_multi_core = true;
    std::string log_dir = "stress_logs";
    int hang_timeout_s = 60;
    // sysfs path to NPU firmware log; adjust PCI address for the target system
    std::string fw_log_path = "/sys/kernel/debug/accel/0000:00:0b.0/fw_log";
    // key=value config for ZE/OV env vars; loaded before ov::Core is created
    std::string config_path = "stress_test.conf";
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -m <model> [options]\n"
              << "  -m  <path>      model file (.xml/.onnx)\n"
              << "  -d  <device>    target device (default: CPU)\n"
              << "  -t  <N>         worker threads (default: 4)\n"
              << "  -n  <N>         iterations per scenario (default: 50)\n"
              << "  --duration <s>  run each scenario for <s> seconds\n"
              << "  --only <name>   run only one scenario:\n"
              << "                  load_unload | parallel | concurrent |\n"
              << "                  import_export | mid_flight | memory |\n"
              << "                  destroy_live_req | multi_core\n"
              << "  --log-dir <dir>      log directory (default: stress_logs)\n"
              << "  --hang-timeout <s>   exit if no infer() returns for <s>s (default: 60)\n"
              << "  --fw-log <path>      NPU fw_log sysfs path\n"
              << "  --config <file>      key=value log config (default: stress_test.conf)\n";
}

static Options parse_args(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc)
            opt.model_path = argv[++i];
        else if (a == "-d" && i + 1 < argc)
            opt.device = argv[++i];
        else if (a == "-t" && i + 1 < argc)
            opt.threads = std::stoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc)
            opt.iterations = std::stoi(argv[++i]);
        else if (a == "--duration" && i + 1 < argc)
            opt.duration_s = std::stoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc) {
            std::string s = argv[++i];
            opt.run_load_unload = (s == "load_unload");
            opt.run_parallel_infer = (s == "parallel");
            opt.run_concurrent = (s == "concurrent");
            opt.run_import_export = (s == "import_export");
            opt.run_mid_flight = (s == "mid_flight");
            opt.run_memory_stress = (s == "memory");
            opt.run_destroy_live_req = (s == "destroy_live_req");
            opt.run_multi_core = (s == "multi_core");
        } else if (a == "--log-dir" && i + 1 < argc)
            opt.log_dir = argv[++i];
        else if (a == "--hang-timeout" && i + 1 < argc)
            opt.hang_timeout_s = std::stoi(argv[++i]);
        else if (a == "--fw-log" && i + 1 < argc)
            opt.fw_log_path = argv[++i];
        else if (a == "--config" && i + 1 < argc)
            opt.config_path = argv[++i];
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
    }
    return opt;
}

// Config file and console tee
// Config file (key=value) sets ZE/OV env vars BEFORE ov::Core is created so
// the NPU driver and Level Zero loader pick them up at initialisation time.
// TeeBuffer mirrors std::cout to console.txt for the full run.

using ConfigMap = std::map<std::string, std::string>;

static ConfigMap load_config(const std::string& path) {
    ConfigMap kv;
    std::ifstream f(path);
    if (!f)
        return kv;
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find('#');
        if (pos != std::string::npos)
            line.erase(pos);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\r' || line.back() == '\t'))
            line.pop_back();
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
            line.erase(0, 1);
        if (line.empty())
            continue;
        pos = line.find('=');
        if (pos == std::string::npos)
            continue;
        kv[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return kv;
}

static void apply_env(const ConfigMap& kv) {
    for (const auto& p : kv) {
        if (p.first == "LOG_LEVEL")
            continue;  // applied via ov::log::level after Core init
#if defined(_WIN32)
        _putenv_s(p.first.c_str(), p.second.c_str());
#else
        setenv(p.first.c_str(), p.second.c_str(), 1);
#endif
        std::cout << "  env: " << p.first << "=" << p.second << "\n";
    }
}

// Tees std::cout to a file for the full run without touching other code.
class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::ostream& stream, const std::string& path)
        : m_stream(stream),
          m_orig(stream.rdbuf()),
          m_file(path, std::ios::out | std::ios::trunc) {
        m_stream.rdbuf(this);
    }

    ~TeeBuffer() override {
        sync();
        m_stream.rdbuf(m_orig);
    }

protected:
    int overflow(int c) override {
        if (c == EOF)
            return c;
        std::lock_guard<std::mutex> lk(m_mtx);
        m_orig->sputc(static_cast<char>(c));
        if (m_file)
            m_file.put(static_cast<char>(c));
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_orig->sputn(s, n);
        if (m_file)
            m_file.write(s, n);
        return n;
    }
    int sync() override {
        std::lock_guard<std::mutex> lk(m_mtx);
        const int stream_status = m_orig->pubsync();
        if (m_file)
            m_file.flush();
        return stream_status;
    }

private:
    std::ostream& m_stream;
    std::streambuf* m_orig;
    std::ofstream m_file;
    std::mutex m_mtx;
};

// Log capture
// Saves NPU firmware log and kernel messages before/after each scenario and
// on hang detection. Paths and commands are platform-specific.

namespace log_capture {

static std::string run_cmd(const char* cmd) {
    std::string out;
#if !defined(_WIN32)
    FILE* fp = popen(cmd, "r");
    if (fp) {
        char buf[512];
        while (fgets(buf, sizeof(buf), fp))
            out += buf;
        pclose(fp);
    }
#else
    (void)cmd;
#endif
    return out;
}

static std::string read_fw_log(const std::string& path) {
    if (path.empty())
        return "";
    std::ifstream f(path);
    if (!f)
        return "";
    return std::string(std::istreambuf_iterator<char>(f), {});
}

static std::string read_dmesg() {
#if defined(__linux__) || defined(__ANDROID__)
    return run_cmd("dmesg 2>/dev/null");
#else
    return "";
#endif
}

static void ensure_dir(const std::string& dir) {
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
}

static void write_file(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (f)
        f << data;
}

static void save(const std::string& dir, const std::string& prefix, const std::string& fw_log_path) {
    if (dir.empty())
        return;
    ensure_dir(dir);
    const std::string fw = read_fw_log(fw_log_path);
    const std::string dm = read_dmesg();
    write_file(dir + "/" + prefix + "_fw_log.txt", fw.empty() ? "(unavailable)\n" : fw);
    write_file(dir + "/" + prefix + "_dmesg.txt", dm.empty() ? "(unavailable)\n" : dm);
    std::cout << "  logs: " << dir << "/" << prefix << "_{fw_log,dmesg}.txt\n";
}

}  // namespace log_capture

// Hang watchdog
// Background thread: if no infer() returns for hang_timeout_s seconds, saves
// fw_log + dmesg then calls _Exit(1). Targets "NPU driver stuck" failures
// where infer() blocks forever and the process would hang indefinitely.

namespace watchdog {

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static std::atomic<int64_t> g_last_beat_ms{0};
static std::atomic<bool> g_armed{false};
static std::atomic<bool> g_running{false};
static std::string g_scenario;
static std::string g_log_dir;
static std::string g_fw_log_path;
static int g_timeout_s{60};
static std::thread g_thread;

// Call after every infer() return (pass or exception) to reset the timer.
static void beat() {
    g_last_beat_ms.store(now_ms(), std::memory_order_relaxed);
}

static void arm(const std::string& scenario) {
    g_scenario = scenario;
    beat();
    g_armed.store(true, std::memory_order_release);
}

static void disarm() {
    g_armed.store(false, std::memory_order_release);
}

static void start(int timeout_s, const std::string& log_dir, const std::string& fw_log_path) {
    g_timeout_s = timeout_s;
    g_log_dir = log_dir;
    g_fw_log_path = fw_log_path;
    g_running.store(true, std::memory_order_release);
    g_thread = std::thread([]() {
        while (g_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (!g_armed.load(std::memory_order_acquire))
                continue;
            int64_t elapsed = now_ms() - g_last_beat_ms.load(std::memory_order_relaxed);
            if (elapsed < (int64_t)g_timeout_s * 1000)
                continue;
            g_armed.store(false, std::memory_order_release);  // prevent re-entry
            std::cerr << "\n*** HANG: infer() has not returned for " << g_timeout_s << "s  (scenario: " << g_scenario
                      << ")\n"
                      << "    Saving diagnostic logs to: " << g_log_dir << "\n";
            log_capture::save(g_log_dir, "hang_" + g_scenario, g_fw_log_path);
            std::_Exit(1);
        }
    });
}

static void stop() {
    g_running.store(false, std::memory_order_release);
    if (g_thread.joinable())
        g_thread.join();
}

}  // namespace watchdog

// Tensor helpers

// Resolve a partial shape to a concrete shape.
// Batch dim -> 1, sequence/other dynamic dims -> seq_len.
static ov::Shape resolve_shape(const ov::PartialShape& ps, size_t seq_len) {
    ov::Shape s;
    for (size_t i = 0; i < ps.size(); ++i) {
        if (ps[i].is_static())
            s.push_back(static_cast<size_t>(ps[i].get_length()));
        else
            s.push_back(i == 0 ? 1 : seq_len);
    }
    if (s.empty())
        s.push_back(1);
    return s;
}

static void fill_random(ov::Tensor& t) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    const size_t n = t.get_size();
    const auto et = t.get_element_type();

    if (et == ov::element::f32) {
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        auto* p = t.data<float>();
        for (size_t i = 0; i < n; ++i)
            p[i] = d(rng);
    } else if (et == ov::element::f16) {
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        auto* p = t.data<ov::float16>();
        for (size_t i = 0; i < n; ++i)
            p[i] = ov::float16(d(rng));
    } else if (et == ov::element::i64) {
        std::uniform_int_distribution<int64_t> d(0, 1024);
        auto* p = t.data<int64_t>();
        for (size_t i = 0; i < n; ++i)
            p[i] = d(rng);
    } else if (et == ov::element::i32) {
        std::uniform_int_distribution<int32_t> d(0, 1024);
        auto* p = t.data<int32_t>();
        for (size_t i = 0; i < n; ++i)
            p[i] = d(rng);
    } else {
        std::memset(t.data(), 0, t.get_byte_size());
    }
}

// Build and fill an input tensor for one model port.
static ov::Tensor make_tensor(const ov::Output<const ov::Node>& port, size_t seq_len) {
    ov::Shape shape = resolve_shape(port.get_partial_shape(), seq_len);
    ov::Tensor t(port.get_element_type(), shape);
    fill_random(t);
    return t;
}

// Set all input tensors on an InferRequest using the aicore pattern:
//   infer_req.set_tensor(name, tensor)
static void set_inputs(ov::InferRequest& req, const ov::CompiledModel& cm, size_t seq_len) {
    for (const auto& in : cm.inputs()) {
        req.set_tensor(in.get_any_name(), make_tensor(in, seq_len));
    }
}

// Detect LLM-style models from input port names.
static bool is_llm(const ov::CompiledModel& cm) {
    for (const auto& in : cm.inputs()) {
        const std::string& n = in.get_any_name();
        if (n.find("input_ids") != std::string::npos || n.find("attention_mask") != std::string::npos ||
            n.find("position_ids") != std::string::npos)
            return true;
    }
    return false;
}

// Iteration guard

struct Guard {
    int iterations;
    bool timed;
    std::chrono::steady_clock::time_point deadline;

    explicit Guard(const Options& opt)
        : iterations(opt.iterations),
          timed(opt.duration_s > 0),
          deadline(std::chrono::steady_clock::now() + std::chrono::seconds(opt.duration_s)) {}

    bool ok(int i) const {
        if (g_stop)
            return false;
        return timed ? std::chrono::steady_clock::now() < deadline : i < iterations;
    }
};

// Scenario 1: load / unload
//
// N threads each repeatedly compile the model, create an InferRequest,
// run one inference, then destroy everything. Stresses driver load/unload
// and resource tracking under concurrent compilation.

static void test_load_unload(ov::Core& core, const std::shared_ptr<ov::Model>& model, const Options& opt, Stats& st) {
    std::cout << "\n=== load/unload (" << opt.threads << " threads) ===\n";

    std::vector<std::thread> workers;
    for (int t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            Guard g(opt);
            for (int i = 0; g.ok(i); ++i) {
                try {
                    auto cm = core.compile_model(model, opt.device);
                    auto req = cm.create_infer_request();
                    set_inputs(req, cm, 16);
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    st.fail++;
                    st.record_error("load_unload[" + std::to_string(t) + "]: " + e.what());
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    st.print("load_unload");
}

// Scenario 2: parallel inference
//
// N threads share one CompiledModel. Each thread owns its InferRequest
// (as in aicore) and calls infer() synchronously in a tight loop.

static void test_parallel_infer(ov::CompiledModel& cm, const Options& opt, Stats& st) {
    const bool llm = is_llm(cm);
    std::cout << "\n=== parallel_infer (" << opt.threads << " threads" << (llm ? ", LLM" : "") << ") ===\n";

    std::vector<std::thread> workers;
    for (int t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            auto req = cm.create_infer_request();
            Guard g(opt);
            for (int i = 0; g.ok(i); ++i) {
                try {
                    // vary sequence length for LLM to cover prefill / decode
                    size_t seq = llm ? static_cast<size_t>(8 + (i % 16)) : 1;
                    set_inputs(req, cm, seq);
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    st.fail++;
                    st.record_error("parallel[" + std::to_string(t) + "]: " + e.what());
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    st.print("parallel_infer");
}

// Scenario 3: concurrent load + infer
//
// Thread A continuously compiles and destroys models.
// Thread B continuously runs inference on a pre-compiled model.
// Verifies stability when driver load/unload races with active inference.

static void test_concurrent_load_infer(ov::Core& core,
                                       const std::shared_ptr<ov::Model>& model,
                                       ov::CompiledModel& cm_steady,
                                       const Options& opt,
                                       Stats& st) {
    std::cout << "\n=== concurrent load + infer ===\n";
    std::atomic<bool> done{false};

    // inference thread
    std::thread infer_t([&]() {
        auto req = cm_steady.create_infer_request();
        while (!done && !g_stop) {
            try {
                set_inputs(req, cm_steady, 16);
                req.infer();
                watchdog::beat();
                (void)req.get_output_tensor(0);
                st.pass++;
            } catch (const std::exception& e) {
                st.fail++;
                st.record_error(std::string("concurrent infer: ") + e.what());
            }
        }
    });

    // load/unload thread
    Guard g(opt);
    for (int i = 0; g.ok(i); ++i) {
        try {
            auto cm = core.compile_model(model, opt.device);
            (void)cm;
            st.pass++;
        } catch (const std::exception& e) {
            st.fail++;
            st.record_error(std::string("concurrent load: ") + e.what());
        }
    }
    done = true;
    infer_t.join();
    st.print("concurrent_load_infer");
}

// Scenario 4: import / export round-trip
//
// Compile once, export blob to memory, then repeatedly:
//   import -> create InferRequest -> set_tensor -> infer() -> get_tensor
// Both single-threaded and multi-threaded import paths are exercised.

static void test_import_export(ov::Core& core, const std::shared_ptr<ov::Model>& model, const Options& opt, Stats& st) {
    std::cout << "\n=== import/export ===\n";

    std::ostringstream blob_buf;
    try {
        core.compile_model(model, opt.device).export_model(blob_buf);
    } catch (const std::exception& e) {
        std::cout << "  SKIP: export not supported on " << opt.device << " (" << e.what() << ")\n";
        return;
    }
    const std::string blob = blob_buf.str();

    std::vector<std::thread> workers;
    for (int t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            Guard g(opt);
            for (int i = 0; g.ok(i); ++i) {
                try {
                    std::istringstream iss(blob);
                    auto cm = core.import_model(iss, opt.device);
                    auto req = cm.create_infer_request();
                    set_inputs(req, cm, 16);
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    st.fail++;
                    st.record_error("import_export[" + std::to_string(t) + "]: " + e.what());
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    st.print("import_export");
}

// Scenario 5: new inference mid-flight
//
// Thread A executes a long infer() while Thread B immediately kicks off its
// own infer() on a second InferRequest from the same CompiledModel.
// Additionally, Thread C cancels Thread A's request mid-flight to verify
// cancellation does not corrupt shared device state.

static void test_mid_flight(ov::CompiledModel& cm, const Options& opt, Stats& st) {
    const bool llm = is_llm(cm);
    std::cout << "\n=== new infer mid-flight" << (llm ? " (LLM)" : "") << " ===\n";

    // Two persistent InferRequests per iteration pair: req_a and req_b.
    auto req_a = cm.create_infer_request();
    auto req_b = cm.create_infer_request();

    Guard g(opt);
    for (int i = 0; g.ok(i); ++i) {
        try {
            size_t seq = llm ? static_cast<size_t>(8 + (i % 8)) : 1;

            // Thread A: set inputs and infer (may be cancelled mid-flight)
            std::atomic<bool> a_done{false};
            std::thread a([&]() {
                try {
                    set_inputs(req_a, cm, seq);
                    req_a.infer();
                    watchdog::beat();
                    st.pass++;
                } catch (const std::exception&) {
                    watchdog::beat();  // cancelled, not hung
                    st.cancelled++;
                }
                a_done = true;
            });

            // Thread B: start its own inference immediately (no waiting for A)
            std::thread b([&]() {
                try {
                    set_inputs(req_b, cm, seq);
                    req_b.infer();
                    watchdog::beat();
                    (void)req_b.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    watchdog::beat();
                    st.fail++;
                    st.record_error(std::string("mid_flight B: ") + e.what());
                }
            });

            // Thread C: cancel req_a shortly after it starts
            std::thread c([&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (!a_done)
                    req_a.cancel();
            });

            a.join();
            b.join();
            c.join();
        } catch (const std::exception& e) {
            st.fail++;
            st.record_error(std::string("mid_flight: ") + e.what());
        }
    }
    st.print("mid_flight");
}

// Scenario 6: memory stress
//
// Load N CompiledModel instances simultaneously (one per thread), run
// inference on each concurrently, then unload all at once.
// Stresses device memory allocation, handle tables, and teardown ordering.

static void test_memory_stress(ov::Core& core, const std::shared_ptr<ov::Model>& model, const Options& opt, Stats& st) {
    std::cout << "\n=== memory stress (" << opt.threads << " simultaneous models) ===\n";

    Guard g(opt);
    for (int round = 0; g.ok(round); ++round) {
        std::vector<ov::CompiledModel> models;
        models.reserve(static_cast<size_t>(opt.threads));

        // load phase
        for (int t = 0; t < opt.threads && !g_stop; ++t) {
            try {
                models.push_back(core.compile_model(model, opt.device));
                st.pass++;
            } catch (const std::exception& e) {
                st.fail++;
                st.record_error(std::string("memory load: ") + e.what());
            }
        }

        // Concurrent inference phase; each thread uses its own model/request.
        std::vector<std::thread> workers;
        for (size_t t = 0; t < models.size(); ++t) {
            workers.emplace_back([&, t]() {
                try {
                    auto req = models[t].create_infer_request();
                    set_inputs(req, models[t], 16);
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    st.fail++;
                    st.record_error("memory infer[" + std::to_string(t) + "]: " + e.what());
                }
            });
        }
        for (auto& w : workers)
            w.join();

        models.clear();  // explicit unload all
    }
    st.print("memory_stress");
}

// Scenario 7: destroy CompiledModel while InferRequest is live
//
// Compiles a fresh model each iteration, creates an InferRequest, starts
// infer() in a thread, then immediately drops the CompiledModel handle.
// The InferRequest holds an internal reference so inference must complete
// cleanly. Exercises driver-side handle and context lifetime tracking.

static void test_destroy_live_req(ov::Core& core,
                                  const std::shared_ptr<ov::Model>& model,
                                  const Options& opt,
                                  Stats& st) {
    std::cout << "\n=== destroy_live_req ===\n";

    Guard g(opt);
    for (int i = 0; g.ok(i); ++i) {
        try {
            // shared_ptr lets us reset() the handle at a precise point
            auto cm = std::make_shared<ov::CompiledModel>(core.compile_model(model, opt.device));
            auto req = cm->create_infer_request();
            set_inputs(req, *cm, 16);

            std::thread infer_t([&req, &st]() {
                try {
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                } catch (const std::exception& e) {
                    watchdog::beat();
                    st.fail++;
                    st.record_error(std::string("destroy_live req: ") + e.what());
                }
            });

            cm.reset();  // drop handle while infer() may be in flight

            infer_t.join();
        } catch (const std::exception& e) {
            st.fail++;
            st.record_error(std::string("destroy_live compile: ") + e.what());
        }
    }
    st.print("destroy_live_req");
}

// Scenario 8: multiple ov::Core instances
//
// N threads each create their own ov::Core, open a fresh device connection,
// compile the model, run inference, then destroy Core + CompiledModel.
// Stresses the Level Zero loader global state and driver reference counting
// under simultaneous Core creation and deletion.

static void test_multi_core(const Options& opt, Stats& st) {
    std::cout << "\n=== multi_core (" << opt.threads << " threads) ===\n";

    std::vector<std::thread> workers;
    for (int t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            Guard g(opt);
            for (int i = 0; g.ok(i); ++i) {
                try {
                    ov::Core local_core;
                    auto mdl = local_core.read_model(opt.model_path);
                    auto cm = local_core.compile_model(mdl, opt.device);
                    auto req = cm.create_infer_request();
                    set_inputs(req, cm, 16);
                    req.infer();
                    watchdog::beat();
                    (void)req.get_output_tensor(0);
                    st.pass++;
                    // req, cm, mdl, local_core all destroyed at scope exit
                } catch (const std::exception& e) {
                    st.fail++;
                    st.record_error("multi_core[" + std::to_string(t) + "]: " + e.what());
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
    st.print("multi_core");
}

template <typename Callable>
static void run_scenario(const Options& opt, const std::string& name, Callable scenario) {
    if (g_stop)
        return;

    Stats stats;
    log_capture::save(opt.log_dir, name + "_start", opt.fw_log_path);
    watchdog::arm(name);
    try {
        scenario(stats);
    } catch (...) {
        watchdog::disarm();
        throw;
    }
    watchdog::disarm();
    log_capture::save(opt.log_dir, name + "_end", opt.fw_log_path);
}

// Main

int main(int argc, char* argv[]) {
    std::signal(SIGINT, on_signal);

    Options opt = parse_args(argc, argv);
    if (opt.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load config and apply env vars BEFORE ov::Core is created so the NPU
    // driver and Level Zero loader see them during initialisation.
    const ConfigMap cfg = load_config(opt.config_path);
    if (!cfg.empty())
        std::cout << "config: " << opt.config_path << "\n";
    apply_env(cfg);

    // Mirror all std::cout output to <log_dir>/console.txt for the full run.
    log_capture::ensure_dir(opt.log_dir);
    TeeBuffer tee_buf(std::cout, opt.log_dir + "/console.txt");

    std::cout << "OpenVINO " << ov::get_openvino_version() << "\n"
              << "device:     " << opt.device << "\n"
              << "model:      " << opt.model_path << "\n"
              << "threads:    " << opt.threads << "\n";
    if (opt.duration_s > 0)
        std::cout << "duration:   " << opt.duration_s << "s per scenario\n";
    else
        std::cout << "iterations: " << opt.iterations << " per scenario\n";

    ov::Core core;

    // Capture OV internal diagnostics to <log_dir>/ov_log.txt.
    {
        static std::ofstream ov_log_file(opt.log_dir + "/ov_log.txt", std::ios::out | std::ios::trunc);
        static std::mutex ov_log_mtx;
        ov::util::set_log_callback([](std::string_view msg) {
            std::lock_guard<std::mutex> lk(ov_log_mtx);
            if (ov_log_file)
                ov_log_file << msg;
        });
    }

    // Apply OV core log level from config (LOG_LEVEL key).
    auto it = cfg.find("LOG_LEVEL");
    if (it != cfg.end()) {
        try {
            ov::log::Level lvl;
            std::istringstream{it->second} >> lvl;
            core.set_property(ov::log::level(lvl));
            std::cout << "  OV log level: " << it->second << "\n";
        } catch (const std::exception& e) {
            std::cerr << "WARNING: cannot set OpenVINO log level: " << e.what() << "\n";
        }
    }

    std::shared_ptr<ov::Model> model;
    try {
        model = core.read_model(opt.model_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR reading model: " << e.what() << "\n";
        return 2;
    }

    ov::CompiledModel cm_base;
    try {
        cm_base = core.compile_model(model, opt.device);
    } catch (const std::exception& e) {
        std::cerr << "ERROR compiling model: " << e.what() << "\n";
        return 2;
    }

    std::cout << "model type: " << (is_llm(cm_base) ? "LLM" : "classic/non-LLM") << "\n"
              << "log dir:    " << opt.log_dir << "\n"
              << "hang timeout: " << opt.hang_timeout_s << "s\n";

    log_capture::save(opt.log_dir, "baseline", opt.fw_log_path);
    watchdog::start(opt.hang_timeout_s, opt.log_dir, opt.fw_log_path);

    if (opt.run_load_unload)
        run_scenario(opt, "load_unload", [&](Stats& stats) {
            test_load_unload(core, model, opt, stats);
        });
    if (opt.run_parallel_infer)
        run_scenario(opt, "parallel_infer", [&](Stats& stats) {
            test_parallel_infer(cm_base, opt, stats);
        });
    if (opt.run_concurrent) {
        run_scenario(opt, "concurrent", [&](Stats& stats) {
            test_concurrent_load_infer(core, model, cm_base, opt, stats);
        });
    }
    if (opt.run_import_export)
        run_scenario(opt, "import_export", [&](Stats& stats) {
            test_import_export(core, model, opt, stats);
        });
    if (opt.run_mid_flight)
        run_scenario(opt, "mid_flight", [&](Stats& stats) {
            test_mid_flight(cm_base, opt, stats);
        });
    if (opt.run_memory_stress)
        run_scenario(opt, "memory_stress", [&](Stats& stats) {
            test_memory_stress(core, model, opt, stats);
        });
    if (opt.run_destroy_live_req) {
        run_scenario(opt, "destroy_live_req", [&](Stats& stats) {
            test_destroy_live_req(core, model, opt, stats);
        });
    }
    if (opt.run_multi_core)
        run_scenario(opt, "multi_core", [&](Stats& stats) {
            test_multi_core(opt, stats);
        });

    watchdog::stop();
    std::cout << "\n=== stress test complete ===\n";
    return 0;
}
