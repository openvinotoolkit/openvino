//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <string>

#include "backend.hpp"
#include "ie_tensor.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

class Handle;

namespace ngraph
{
    namespace runtime
    {
        namespace ie
        {
            class IE_Backend final : public runtime::Backend
            {
            public:
                IE_Backend(const std::string& configuration_string);
                virtual ~IE_Backend() = default;

                std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                                    bool enable_performance_data = false) override;
                bool is_supported(const Node& node) const override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_dynamic_tensor(const ngraph::element::Type& type,
                                          const ngraph::PartialShape& shape) override;

                std::shared_ptr<ngraph::runtime::Tensor> create_tensor() override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) final override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* data) final override;

                template <typename T>
                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(ngraph::element::Type type, ngraph::Shape shape, T* data)
                {
                    auto tensor = std::make_shared<IETensor>(type, shape);
                    size_t size = shape_size(shape);
                    tensor->write(data, size * sizeof(T));
                    return tensor;
                }

            private:
                std::string m_device;
            };
        }
    }
}
