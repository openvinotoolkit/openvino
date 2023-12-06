#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

struct ReadModelArgs {
    std::string model_path;
    std::string bin_path;
    std::string model_str;
    ov::Tensor weight_tensor;

    ReadModelArgs() {}
    ReadModelArgs(const Napi::CallbackInfo& info) {
        if (!is_valid_read_model_input(info))
            throw std::runtime_error("Invalid arguments of read model function");

        const size_t argsLength = info.Length();
        std::shared_ptr<ov::Model> model;

        if (info[0].IsBuffer()) {
            Napi::Buffer<uint8_t> model_data = info[0].As<Napi::Buffer<uint8_t>>();
            model_str = std::string(reinterpret_cast<char*>(model_data.Data()), model_data.Length());

            if (argsLength == 2) {
                Napi::Buffer<uint8_t> weights = info[1].As<Napi::Buffer<uint8_t>>();
                const uint8_t* bin = reinterpret_cast<const uint8_t*>(weights.Data());

                size_t bin_size = weights.Length();
                weight_tensor = ov::Tensor(ov::element::Type_t::u8, {bin_size});
                std::memcpy(weight_tensor.data(), bin, bin_size);
            }
            else {
                weight_tensor = ov::Tensor(ov::element::Type_t::u8, {0});
            }
        } else {
            model_path = std::string(info[0].ToString());

            if (argsLength == 2) bin_path = info[1].ToString();
        }
    }

    bool is_valid_read_model_input(const Napi::CallbackInfo& info) {
        const size_t argsLength = info.Length();
        const size_t is_buffers_input = info[0].IsBuffer()
            && (argsLength == 1 || info[1].IsBuffer());

        if (is_buffers_input) return true;

        return info[0].IsString() && (argsLength == 1 || info[1].IsString());
    }
};
