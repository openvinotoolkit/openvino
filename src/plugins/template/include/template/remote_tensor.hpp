// Copyright (C) 2018-2023 Intel Corporation
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

class RemoteContext;

class VectorTensor : public ov::RemoteTensor {
private:
    class VectorImpl;
    std::shared_ptr<VectorImpl> get_impl() const;
    friend class RemoteContext;

public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(tensor, {{ov::device::id.name(), {"TEMPLATE"}}});
    }

    /**
     * @brief Returns the underlying vector
     * @return vector if T is compatible with element type
     */
    template <class T>
    const std::vector<T>& get() const {
        return *get_impl();
    }
};

}  // namespace template_plugin
}  // namespace ov
