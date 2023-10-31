// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <openvino/runtime/core.hpp>

#include "async_reader.hpp"
#include "errors.hpp"

class CoreWrap : public Napi::ObjectWrap<CoreWrap> {
public:
    /**
     * @brief Constructs CoreWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     */
    CoreWrap(const Napi::CallbackInfo& info);
    /**
     * @brief Defines a Javascript Core class with constructor, static and instance properties and methods.
     * @param env The environment in which to construct a JavaScript class.
     * @return Napi::Function representing the constructor function for the Javascript Core class.
     */
    static Napi::Function GetClassConstructor(Napi::Env env);
    /** @brief This method is called during initialization of OpenVino native add-on.
     * It exports JavaScript Core class.
     */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    /**
     * @brief Reads a model synchronously.
     * @param info contains passed arguments.
     * One argument is passed:
     * @param info[0] path to a model. (model_path string or Buffer<UInt8Array>)
     * Two arguments are passed:
     * @param info[0] path to a model. (model_path string or Buffer<UInt8Array>)
     * @param info[1] path to a data file. (e.g. bin_path string or Buffer<UInt8Array>)
     * @return A Javascript Model object.
     */
    Napi::Value read_model_sync(const Napi::CallbackInfo& info);

    /**
     * @brief Asynchronously reads a model.
     * @param info contains passed arguments.
     * One argument is passed:
     * @param info[0] path to a model. (model_path)
     * Two arguments are passed:
     * @param info[0] path to a model. (model_path)
     * @param info[1] path to a data file. (e.g. bin_path)
     * @return A Javascript Promise.
     */
    Napi::Value read_model_async(const Napi::CallbackInfo& info);

    /**
     * @brief Creates and loads a compiled model from a source model.
     * @param info contains two passed arguments.
     * @param info[0] Javascript Model object acquired from CoreWrap::read_model
     * @param info[1] string with propetries e.g. device
     * @return A Javascript CompiledModel object.
     */
    Napi::Value compile_model_sync_dispatch(const Napi::CallbackInfo& info);

    /**
     * @brief Asynchronously creates and loads a compiled model from a source model.
     * @param info contains two passed arguments.
     * @param info[0] Javascript Model object acquired from CoreWrap::read_model
     * @param info[1] string with propetries e.g. device
     * @return A Javascript CompiledModel object.
     */
    Napi::Value compile_model_async(const Napi::CallbackInfo& info);

protected:
    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                          const Napi::Object& model,
                                          const Napi::String& device);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                          const Napi::String& model_path,
                                          const Napi::String& device);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                          const Napi::Object& model,
                                          const Napi::String& device,
                                          const std::map<std::string, ov::Any>& config);

    Napi::Value compile_model_sync(const Napi::CallbackInfo& info,
                                          const Napi::String& model_path,
                                          const Napi::String& device,
                                          const std::map<std::string, ov::Any>& config);

private:
    ov::Core _core;
    Napi::Env env;
};

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
