// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "espcn_x2_model.hpp"
#include "openvino/openvino.hpp"

namespace ov::test::behavior {

// Reference outputs depend only on the reference compiled model and the input data, never on the NPU configuration
// under test, so each distinct input is inferred once per cache instance.
class EspcnReferenceCache {
public:
    using CompiledModelProvider = std::function<ov::CompiledModel()>;

    // TEMPLATE model compiled once per process and shared by all tests; throws ov::Exception if TEMPLATE is unavailable.
    static ov::CompiledModel getCompiledModel(const std::string& modelName) {
        static ReferenceModels models;
        std::lock_guard<std::mutex> lock(models.mutex);
        auto it = models.compiled.find(modelName);
        if (it == models.compiled.end()) {
            auto compiledModel =
                models.core.compile_model(createESPCNX2ModelByName(modelName), ov::test::utils::DEVICE_TEMPLATE);
            it = models.compiled.emplace(modelName, std::move(compiledModel)).first;
        }
        return it->second;
    }

    // The provider is resolved on the first cache miss, so the reference model is only compiled when needed.
    explicit EspcnReferenceCache(CompiledModelProvider provider) : _provider(std::move(provider)) {}

    // Thread-safe; concurrent lookups of the same input wait for a single reference inference.
    ov::Tensor getOutput(const ov::Tensor& input) {
        const std::string_view bytes(static_cast<const char*>(input.data()), input.get_byte_size());
        const Key key{input.get_shape().to_string(), std::hash<std::string_view>{}(bytes)};

        std::promise<ov::Tensor> promise;
        std::shared_future<ov::Tensor> output;
        bool owner = false;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            auto [it, inserted] = _outputs.try_emplace(key);
            if (inserted) {
                it->second = promise.get_future().share();
                owner = true;
            }
            output = it->second;
        }

        if (owner) {
            try {
                promise.set_value(infer(input));
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        }
        return output.get();
    }

private:
    // Reference inferences run one at a time on a single request, so a reference is never produced concurrently.
    ov::Tensor infer(const ov::Tensor& input) {
        std::lock_guard<std::mutex> lock(_inferMutex);
        if (!_request) {
            _request = _provider().create_infer_request();
        }
        _request.set_input_tensor(input);
        _request.infer();
        const auto result = _request.get_output_tensor();
        ov::Tensor copy(result.get_element_type(), result.get_shape());
        result.copy_to(copy);
        return copy;
    }

    struct ReferenceModels {
        ReferenceModels() {
#if !defined(OPENVINO_STATIC_LIBRARY) && !defined(USE_STATIC_IE)
            ov::test::utils::register_template_plugin(core);
#endif  // !OPENVINO_STATIC_LIBRARY && !USE_STATIC_IE
        }

        ov::Core core;
        std::mutex mutex;
        std::map<std::string, ov::CompiledModel> compiled;
    };

    // Input shape and hash of the input bytes.
    using Key = std::tuple<std::string, size_t>;

    CompiledModelProvider _provider;
    std::mutex _inferMutex;
    ov::InferRequest _request;
    std::mutex _mutex;
    std::map<Key, std::shared_future<ov::Tensor>> _outputs;
};

}  // namespace ov::test::behavior
