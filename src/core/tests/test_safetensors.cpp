// test_safetensors: batch 8 — writes a .safetensors file from code (F32 and
// F16 tensors), reads it back through the st_r reader, then runs a matmul
// whose weights came straight from the file. Both executors.

#include "cpu_engine.hpp"
#include "runtime/execution_config.hpp"
#include "safetensors_reader.hpp"
#include "vk_dispatch.hpp"
#include "vk_engine_factory.hpp"
#include "vk_ir.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace ov::core::vulkan::cross_platform;

namespace {

int failures = 0;

void check(const char* name, bool ok) {
    std::printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

void check_vec(const char* name, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    bool ok = got.size() == want.size();
    if (ok)
        for (size_t i = 0; i < want.size(); ++i)
            if (std::fabs(got[i] - want[i]) > tol) {
                ok = false;
                break;
            }
    std::printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

void put_u64(std::string& s, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        s.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    return n;
}

}  // namespace

int main() {
    try {
        const std::string path = (std::getenv("TEMP") ? std::getenv("TEMP") : ".") +
                                 std::string("\\vk_st_test.safetensors");

        // ---- write ------------------------------------------------------------
        // w: F32 [4,8] (128 bytes); u: F16 [2,4] (16 bytes)
        std::vector<float> wf(32);
        for (size_t i = 0; i < wf.size(); ++i)
            wf[i] = static_cast<float>(static_cast<int>(i) % 7) - 3.f;
        std::vector<uint16_t> uh(8);
        for (size_t i = 0; i < uh.size(); ++i)
            uh[i] = static_cast<uint16_t>(0x3800u + i);  // 0.5, ~0.516, ...

        std::string header =
            "{\"w\": {\"dtype\":\"F32\",\"shape\":[4,8],\"data_offsets\":[0,128]},"
            "\"u\": {\"dtype\":\"F16\",\"shape\":[2,4],\"data_offsets\":[128,144]}}";
        while ((8 + header.size()) % 8 != 0)
            header += ' ';  // keep the data block 8-byte aligned

        {
            std::ofstream f(path, std::ios::binary);
            std::string len8;
            put_u64(len8, header.size());
            f.write(len8.data(), 8);
            f.write(header.data(), static_cast<std::streamsize>(header.size()));
            f.write(reinterpret_cast<const char*>(wf.data()), sizeof(float) * wf.size());
            f.write(reinterpret_cast<const char*>(uh.data()), sizeof(uint16_t) * uh.size());
        }

        // ---- read -------------------------------------------------------------
        const auto tensors = st_r::load_safetensors(path);
        check("file contains both tensors", tensors.count("w") == 1 && tensors.count("u") == 1);
        check_vec("F32 payload round-trips", tensors.at("w").data, wf, 0.0f);
        check("w shape [4,8]", tensors.at("w").shape == std::vector<size_t>{4, 8});
        check("u shape [2,4]", tensors.at("u").shape == std::vector<size_t>{2, 4});
        // F16 reference values.
        bool f16_ok = true;
        for (size_t i = 0; i < uh.size(); ++i) {
            const uint32_t bits = static_cast<uint32_t>(uh[i]) << 13;
            const float exp_bits_f = [bits] {
                float v;
                const uint32_t sign = bits & 0x80000000u;
                const uint32_t exp = ((bits >> 23) & 0xff) - 15 + 127;
                const uint32_t man = (bits >> 13) & 0x3ff;
                uint32_t b = sign | (exp << 23) | (man << 13);
                std::memcpy(&v, &b, 4);
                return v;
            }();
            f16_ok &= std::fabs(tensors.at("u").data[i] - exp_bits_f) < 1e-6f;
        }
        check("F16 converts to f32", f16_ok);

        // ---- weights from the file drive a real graph --------------------------
        auto g = [] {
            ir_graph g;
            g.nodes.push_back(nd("x", ir_op::parameter, {}));
            g.nodes.push_back(nd("w", ir_op::constant, {}));
            g.nodes.push_back(nd("mm", ir_op::matmul, {"x", "w"}));
            g.nodes.push_back(nd("out", ir_op::result, {"mm"}));
            g.tensor_shapes["x"] = {2, 8};
            g.tensor_shapes["w"] = {4, 8};
            g.tensor_shapes["mm"] = {2, 8};
            g.inputs = {"x"};
            g.outputs = {"mm"};
            return g;
        }();
        // MatMul is [M,K]x[K,N]; the file tensor w is [4,8] -> use it as the
        // transposed operand by storing it [N,K]-style via transpose_b=false
        // with K=8,N=... simplest: treat w as [K=4? mismatch]. Instead feed
        // w^T data manually: build [8,4] from the file rows.
        std::vector<float> wt(32);
        for (size_t k = 0; k < 4; ++k)
            for (size_t n = 0; n < 8; ++n)
                wt[n * 4 + k] = tensors.at("w").data[k * 8 + n];
        // Use shape [8,4]: x[2,8] x w[8,4] -> [2,4]
        g.tensor_shapes["w"] = {8, 4};
        g.tensor_shapes["mm"] = {2, 4};
        g.constant_data["w"] = wt;

        std::vector<float> x(16);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<float>(i) / 8.f - 1.f;
        auto ref = cpu_execute(g, {{"x", x}}).at("mm");
        auto gpu = vk_execute(g, {{"x", x}}, "GPU").at("mm");
        check_vec("matmul-from-file CPU vs GPU", gpu, ref, 1e-5f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
