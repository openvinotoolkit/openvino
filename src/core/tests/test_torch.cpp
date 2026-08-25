// test_torch: pytorch pipeline without torch вЂ” a handcrafted bundle exactly
// as torch_export.py produces it (linear(8->4) -> relu, weights in
// safetensors) is loaded via pt_r::load_export and executed on CPU and GPU
// against an analytic reference.

#include "cpu_engine.hpp"
#include "pytorch_reader.hpp"
#include "runtime/execution_config.hpp"
#include "vk_dispatch.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace ov::core::vulkan::cross_platform;

namespace {

int failures = 0;

void check_vec(const char* name, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    bool ok = got.size() == want.size();
    if (ok)
        for (size_t i = 0; i < want.size(); ++i)
            if (std::fabs(got[i] - want[i]) > tol) {
                ok = false;
                break;
            }
    std::printf("%-44s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

void put_u64(std::string& s, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        s.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

}  // namespace

int main() {
    try {
        const std::string dir = std::getenv("TEMP") ? std::getenv("TEMP") : ".";
        const std::string graph_path = dir + "\\vk_torch_test.graph.vktorch";
        const std::string weights_path = dir + "\\vk_torch_test.weights.safetensors";

        // fc weight [N=4, K=8] (torch.Linear layout), input x [B=2, K=8]
        std::vector<float> w(32);
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = static_cast<float>(static_cast<int>(i) % 5) - 2.f;

        // ---- weights.safetensors ------------------------------------------------
        std::string header =
            "{\"p_fc_weight\": {\"dtype\":\"F32\",\"shape\":[4,8],\"data_offsets\":[0,128]}}";
        while ((8 + header.size()) % 8)
            header += ' ';
        {
            std::ofstream f(weights_path, std::ios::binary);
            std::string len8;
            put_u64(len8, header.size());
            f.write(len8.data(), 8);
            f.write(header.data(), static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char*>(w.data()), sizeof(float) * w.size());
        }

        // ---- graph.vktorch (as the exporter writes it) --------------------------
        {
            std::ofstream f(graph_path);
            f << "# vktorch v1\n";
            f << "OP in_x parameter 2x8 -\n";
            f << "CONST p_fc_weight 4x8 p_fc_weight\n";
            f << "OP fc linear 2x4 in_x,p_fc_weight\n";
            f << "OP r relu 2x4 fc\n";
            f << "OUT r\n";
        }

        auto g = pt_r::load_export(graph_path, weights_path);

        std::vector<float> x(16);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<float>(i) / 8.f - 1.f;

        // Reference: y = relu(x @ w^T)
        std::vector<float> ref(8, 0.f);
        for (size_t b = 0; b < 2; ++b)
            for (size_t n = 0; n < 4; ++n) {
                float acc = 0.f;
                for (size_t k = 0; k < 8; ++k)
                    acc += x[b * 8 + k] * w[n * 8 + k];
                ref[b * 4 + n] = std::max(0.f, acc);
            }

        const auto cpu = cpu_execute(g, {{"in_x", x}}).at("r");
        check_vec("torch-bundle linear+relu CPU vs ref", cpu, ref, 1e-5f);
        const auto gpu = vk_execute(g, {{"in_x", x}}, "GPU").at("r");
        check_vec("torch-bundle linear+relu GPU vs ref", gpu, ref, 1e-4f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}

