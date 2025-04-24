// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines remote tensor
 *
 * @file template/remote_tensor.hpp
 */

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace template_plugin {

/**
 * @brief Template plugin remote tensor which wraps memory from the vector
 */
// ! [remote_tensor:public_header]
class VectorTensor : public ov::RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(
            tensor,
            {{ov::device::full_name.name(), {"TEMPLATE"}}, {"vector_data_ptr", {}}, {"vector_data", {}}});
    }

    /**
     * @brief Returns the underlying vector
     * @return const reference to vector if T is compatible with element type
     */
    template <class T>
    const std::vector<T>& get_data() const {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto& vec = params.at("vector_data").as<const std::vector<T>>();
            return vec;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Vector type is incorrect!");
        }
    }

    /**
     * @brief Returns the underlying vector
     * @return reference to vector if T is compatible with element type
     */
    template <class T>
    std::vector<T>& get_data() {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto& vec = params.at("vector_data").as<std::vector<T>>();
            return vec;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Vector type is incorrect!");
        }
    }

    /**
     * @brief Returns the const pointer to the data
     *
     * @return const pointer to the tensor data
     */
    const void* get_data() const {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<const void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }

    /**
     * @brief Returns the pointer to the data
     *
     * @return pointer to the tensor data
     */
    void* get_data() {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }
};
// ! [remote_tensor:public_header]

}  // namespace template_plugin
}  // namespace ov
