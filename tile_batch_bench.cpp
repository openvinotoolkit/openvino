// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dev-loop replacement for the python tile_batch_bench.py repro script.
//
// Loads a pre-tiled (N,3,H,W) float32 .npy batch (see prepare_tiles.py) and
// runs it through a dynamic-batch IR and a matching static-batch IR on GPU,
// printing per-image latency for both so you can see the dynamic-vs-static
// slowdown directly.
//
// Unlike a python script using `pip install`-ed openvino, this is compiled
// and linked straight against the incremental build output
// (bin/intel64/Release/libopenvino.so + plugin .so's), so a `ninja
// openvino_intel_gpu_plugin` (or a full rebuild) is picked up immediately --
// no wheel packaging/(re)installation needed. Use build_and_run.sh to
// compile+run in one step.
//
// Usage:
//   ./tile_batch_bench <batch_size> [data_dir] [device]
//
// data_dir must contain:
//   best-ort_dynamic.xml/.bin   (dynamic-batch IR)
//   best-ort_b<N>.xml/.bin      (static-batch IR for this N, optional)
//   tiles_b<N>.npy              (input tensor, produced by prepare_tiles.py)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace {

// Minimal .npy reader: supports flat, C-order uint8 ('|u1') or float32
// ('<f4') arrays -- the two dtypes prepare_tiles.py / this tool care about.
// Returns raw bytes, the element type, and the shape.
struct NpyArray {
    std::vector<uint8_t> raw;
    ov::element::Type_t dtype;
    std::vector<size_t> shape;
};

NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open npy file: " + path);
    }

    std::string magic(6, ' ');
    f.read(&magic[0], magic.size());
    if (magic != "\x93NUMPY") {
        throw std::runtime_error("Not a valid .npy file: " + path);
    }
    f.ignore(2);  // version bytes

    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    std::string header(header_len, ' ');
    f.read(&header[0], header.size());

    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("Expected C-order (fortran_order: False) npy array in " + path);
    }

    size_t elem_size;
    ov::element::Type_t dtype;
    if (header.find("'descr': '<f4'") != std::string::npos) {
        elem_size = 4;
        dtype = ov::element::f32;
    } else if (header.find("'descr': '|u1'") != std::string::npos) {
        elem_size = 1;
        dtype = ov::element::u8;
    } else {
        throw std::runtime_error("Unsupported npy dtype (expected <f4 or |u1) in " + path);
    }

    NpyArray arr;
    arr.dtype = dtype;
    auto shape_key = header.find("'shape':");
    auto open_paren = header.find('(', shape_key);
    auto close_paren = header.find(')', open_paren);
    std::string shape_str = header.substr(open_paren + 1, close_paren - open_paren - 1);
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) {
            arr.shape.push_back(static_cast<size_t>(std::stoul(tok)));
        }
    }

    size_t count = 1;
    for (auto d : arr.shape) {
        count *= d;
    }
    arr.raw.resize(count * elem_size);
    f.read(reinterpret_cast<char*>(arr.raw.data()), static_cast<std::streamsize>(arr.raw.size()));
    if (!f) {
        throw std::runtime_error("Truncated npy data in " + path);
    }
    return arr;
}

double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Runs `iters` inferences (after `warmup`) with the given input data already
// bound, returns {avg_total_ms, avg_per_image_ms}.
std::pair<double, double> run_batch(ov::InferRequest& req, size_t n, size_t warmup, size_t iters) {
    for (size_t i = 0; i < warmup; ++i) {
        req.infer();
    }
    std::vector<double> totals;
    totals.reserve(iters);
    for (size_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        req.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double avg = mean(totals);
    return {avg, avg / static_cast<double>(n)};
}

double bench(const std::string& tag,
             const std::string& xml_path,
             size_t n,
             const NpyArray& tiles,
             const std::string& device,
             size_t warmup,
             size_t iters) {
    ov::Core core;
    auto model = core.read_model(xml_path);
    auto compiled = core.compile_model(model, device);
    auto req = compiled.create_infer_request();

    ov::Shape shape{n, tiles.shape[1], tiles.shape[2], tiles.shape[3]};
    const auto port_type = compiled.input().get_element_type();

    ov::Tensor input;
    if (port_type == tiles.dtype) {
        input = ov::Tensor(port_type, shape, const_cast<uint8_t*>(tiles.raw.data()));
    } else if (port_type == ov::element::f32 && tiles.dtype == ov::element::u8) {
        // Model expects float input but we only loaded uint8 pixels: convert.
        input = ov::Tensor(ov::element::f32, shape);
        auto* dst = input.data<float>();
        const auto* src = tiles.raw.data();
        for (size_t i = 0; i < input.get_size(); ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        throw std::runtime_error("Unhandled combination: npy dtype=" +
                                  std::string(tiles.dtype == ov::element::u8 ? "u8" : "f32") +
                                  " vs model port dtype=" + port_type.get_type_name());
    }
    req.set_input_tensor(input);

    auto [avg, per] = run_batch(req, n, warmup, iters);
    std::cout << "[" << tag << "] batch=" << n << "  total=" << avg << "ms  per_image=" << per
              << "ms  throughput=" << (n / (avg / 1000.0)) << " img/s" << std::endl;
    return per;
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace

int main(int argc, char** argv) {
    // All arguments are optional -- sensible defaults let you just run
    // `./tile_batch_bench` with no arguments at all.
    const size_t n = argc > 1 ? static_cast<size_t>(std::stoul(argv[1])) : 32;
    const std::string data_dir = argc > 2 ? argv[2] : "/home/gta/dynamic_batch_repro";
    const std::string device = argc > 3 ? argv[3] : "GPU";
    const size_t warmup = argc > 4 ? static_cast<size_t>(std::stoul(argv[4])) : 2;
    const size_t iters = argc > 5 ? static_cast<size_t>(std::stoul(argv[5])) : 10;

    std::cout << "Usage: " << argv[0] << " [batch_size=32] [data_dir=" << data_dir
              << "] [device=GPU] [warmup=2] [iters=10]" << std::endl;

    const std::string xml_dyn = data_dir + "/best-ort_dynamic.xml";
    const std::string xml_static = data_dir + "/best-ort_b" + std::to_string(n) + ".xml";
    const std::string npy_path = data_dir + "/tiles_b" + std::to_string(n) + ".npy";

    if (!file_exists(npy_path)) {
        // Auto-generate the input tensor .npy the first time this batch size is used,
        // so you don't have to remember to run prepare_tiles.py yourself.
        std::cout << "Missing " << npy_path << " -- generating it via prepare_tiles.py ..." << std::endl;
        std::string script_dir = std::string(argv[0]);
        auto slash = script_dir.find_last_of('/');
        script_dir = slash == std::string::npos ? "." : script_dir.substr(0, slash);
        // Prefer the user's ~/venv (has numpy/opencv) if present, else fall back to plain python3.
        std::string python = file_exists(std::string(getenv("HOME") ? getenv("HOME") : "") + "/venv/bin/python3")
                                  ? std::string(getenv("HOME")) + "/venv/bin/python3"
                                  : "python3";
        std::string cmd = python + " \"" + script_dir + "/prepare_tiles.py\" " + std::to_string(n) +
                           " --data-dir \"" + data_dir + "\"";
        if (std::system(cmd.c_str()) != 0 || !file_exists(npy_path)) {
            std::cerr << "Failed to auto-generate " << npy_path << ". Run manually:\n  " << cmd << std::endl;
            return 1;
        }
    }

    std::cout << "OpenVINO " << ov::get_openvino_version().buildNumber << std::endl;
    NpyArray tiles = load_npy(npy_path);
    std::cout << "Loaded tiles: batch=" << tiles.shape[0] << " (using " << n << ")" << std::endl;

    std::cout << "===== batch=" << n << " device=" << device << " =====" << std::endl;
    double p_dyn = bench("Dynamic IR", xml_dyn, n, tiles, device, warmup, iters);

    if (file_exists(xml_static)) {
        double p_st = bench("Static IR", xml_static, n, tiles, device, warmup, iters);
        std::cout << "\nper_image: dynamic " << p_dyn << "ms vs static " << p_st << "ms -> static is "
                  << (p_dyn / p_st) << "x faster" << std::endl;
    } else {
        std::cout << "(no matching static IR " << xml_static << ", skipping static comparison)" << std::endl;
    }
    return 0;
}
