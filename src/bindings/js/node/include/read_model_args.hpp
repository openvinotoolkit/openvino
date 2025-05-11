#pragma once

#include <napi.h>

#include "node/include/helper.hpp"
#include "node/include/type_validation.hpp"
#include "openvino/runtime/core.hpp"

/**
 * @brief This struct retrieves data from Napi::CallbackInfo.
 */
struct ReadModelArgs {
    std::string model_path;
    std::string bin_path;
    std::string model_str;
    ov::Tensor weight_tensor;

    ReadModelArgs() {}
    ReadModelArgs(const Napi::CallbackInfo& info) {
        std::vector<std::string> allowed_signatures;

        if (ov::js::validate<Napi::String>(info, allowed_signatures)) {
            model_path = info[0].ToString();
        } else if (ov::js::validate<Napi::String, Napi::String>(info, allowed_signatures)) {
            model_path = info[0].ToString();
            bin_path = info[1].ToString();
        } else if (ov::js::validate<Napi::Buffer<uint8_t>>(info, allowed_signatures)) {
            model_str = buffer_to_string(info[0]);
            weight_tensor = ov::Tensor(ov::element::Type_t::u8, {0});
        } else if (ov::js::validate<Napi::Buffer<uint8_t>, Napi::Buffer<uint8_t>>(info, allowed_signatures)) {
            model_str = buffer_to_string(info[0]);
            Napi::Buffer<uint8_t> weights = info[1].As<Napi::Buffer<uint8_t>>();
            const uint8_t* bin = reinterpret_cast<const uint8_t*>(weights.Data());

            size_t bin_size = weights.Length();
            weight_tensor = ov::Tensor(ov::element::Type_t::u8, {bin_size});
            std::memcpy(weight_tensor.data(), bin, bin_size);
        } else if (ov::js::validate<Napi::String, TensorWrap>(info, allowed_signatures)) {
            model_str = info[0].ToString();
            weight_tensor = cast_to_tensor(info, 1);
        } else {
            OPENVINO_THROW("'readModel'", ov::js::get_parameters_error_msg(info, allowed_signatures));
        }
    }
};
