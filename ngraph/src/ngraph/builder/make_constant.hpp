//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/node.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace builder
    {
        template <class T>
        std::shared_ptr<Node>
            make_constant(const element::Type& type, const Shape& shape, const T& num)
        {
            std::shared_ptr<Node> val = nullptr;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
            switch (type)
            {
            case element::Type_t::f32:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<float>{static_cast<float>(num)});
                break;
            case element::Type_t::f64:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<double>{static_cast<double>(num)});
                break;
            case element::Type_t::f16:
                val = std::make_shared<ngraph::op::Constant>(
                    type,
                    ngraph::Shape{},
                    std::vector<ngraph::float16>{ngraph::float16(static_cast<float>(num))});
                break;
            case element::Type_t::bf16:
                val = std::make_shared<ngraph::op::Constant>(
                    type,
                    ngraph::Shape{},
                    std::vector<ngraph::bfloat16>{ngraph::bfloat16(static_cast<float>(num))});
                break;
            case element::Type_t::i64:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int64_t>{static_cast<int64_t>(num)});
                break;
            case element::Type_t::i32:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int32_t>{static_cast<int32_t>(num)});
                break;
            case element::Type_t::i16:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int16_t>{static_cast<int16_t>(num)});
                break;
            case element::Type_t::i8:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<int8_t>{static_cast<int8_t>(num)});
                break;
            case element::Type_t::u64:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint64_t>{static_cast<uint64_t>(num)});
                break;
            case element::Type_t::u32:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint32_t>{static_cast<uint32_t>(num)});
                break;
            case element::Type_t::u16:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint16_t>{static_cast<uint16_t>(num)});
                break;
            case element::Type_t::u8:
                val = std::make_shared<ngraph::op::Constant>(
                    type, ngraph::Shape{}, std::vector<uint8_t>{static_cast<uint8_t>(num)});
                break;
            case element::Type_t::dynamic:
                throw ngraph_error("make_constant: Unsupported element type 'dynamic'");
            case element::Type_t::boolean:
                throw ngraph_error("make_constant: Unsupported element type 'boolean'");
            case element::Type_t::u1:
                throw ngraph_error("make_constant: Unsupported element type 'u1'");
            case element::Type_t::undefined:
                throw ngraph_error("make_constant: Unsupported element type 'undefined'");
            }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

            if (shape.size() > 0)
            {
                ngraph::AxisSet axes;
                for (size_t i = 0; i < shape.size(); i++)
                {
                    axes.insert(i);
                }
                val = std::make_shared<ngraph::op::v0::Broadcast>(val, shape, axes);
            }

            return val->add_provenance_group_members_above({});
        }
    }
}
