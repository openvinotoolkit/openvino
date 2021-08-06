// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pdpd_utils.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            std::shared_ptr<TensorPlacePDPD> castToTensorPlace(const Place::Ptr& place)
            {
                if (auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place))
                {
                    return var_place;
                }
                else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlacePDPD>(place))
                {
                    return in_port_place->get_source_tensor_pdpd();
                }
                else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlacePDPD>(place))
                {
                    return out_port_place->get_target_tensor_pdpd();
                }
                FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacePDPD.");
            }

            void traverse_down(
                const std::vector<Place::Ptr>& start_tensors,
                const std::function<void(const std::shared_ptr<OpPlacePDPD>& op_place,
                                         const std::shared_ptr<TensorPlacePDPD>& tensor_place)>&
                    callback)
            {
                std::queue<OpPlacePDPD*> q;
                std::unordered_set<OpPlacePDPD*> visited;

                auto check_and_update = [&](const std::shared_ptr<TensorPlacePDPD>& tensor) {
                    if (tensor)
                    {
                        if (!tensor->is_output())
                        {
                            bool is_the_same_tensor = false;
                            const auto& consuming_ops = tensor->get_consuming_operations();
                            for (const auto& op : consuming_ops)
                            {
                                auto pdpd_op = std::dynamic_pointer_cast<OpPlacePDPD>(op);
                                if (pdpd_op && !visited.count(pdpd_op.get()))
                                {
                                    callback(pdpd_op, !is_the_same_tensor ? tensor : nullptr);
                                    is_the_same_tensor = true;

                                    q.push(pdpd_op.get());
                                    visited.insert(pdpd_op.get());
                                }
                            }
                        }
                        else
                        {
                            callback(nullptr, tensor);
                        }
                    }
                };

                for (const auto& tensor : start_tensors)
                {
                    check_and_update(std::dynamic_pointer_cast<TensorPlacePDPD>(tensor));
                }

                while (!q.empty())
                {
                    auto cur_op = q.front();
                    q.pop();
                    for (const auto& map_pair : cur_op->get_output_ports())
                    {
                        for (const auto& port : map_pair.second)
                        {
                            check_and_update(std::dynamic_pointer_cast<TensorPlacePDPD>(
                                port->get_target_tensor()));
                        }
                    }
                }
            }

            void traverse_up(
                const std::vector<Place::Ptr>& start_tensors,
                const std::function<void(const std::shared_ptr<OpPlacePDPD>& op_place,
                                         const std::shared_ptr<TensorPlacePDPD>& tensor_place)>&
                    callback)
            {
                std::queue<OpPlacePDPD*> q;
                std::unordered_set<OpPlacePDPD*> visited;

                auto check_and_update = [&](const std::shared_ptr<TensorPlacePDPD>& tensor) {
                    if (!tensor->is_input())
                    {
                        const auto& producing_op = tensor->get_producing_operation();
                        auto pdpd_op = std::dynamic_pointer_cast<OpPlacePDPD>(producing_op);
                        if (pdpd_op && !visited.count(pdpd_op.get()))
                        {
                            callback(pdpd_op, tensor);

                            q.push(pdpd_op.get());
                            visited.insert(pdpd_op.get());
                        }
                    }
                    else
                    {
                        callback(nullptr, tensor);
                    }
                };

                for (const auto& tensor : start_tensors)
                {
                    check_and_update(std::dynamic_pointer_cast<TensorPlacePDPD>(tensor));
                }

                while (!q.empty())
                {
                    auto cur_op = q.front();
                    q.pop();
                    for (const auto& map_pair : cur_op->get_input_ports())
                    {
                        for (const auto& port : map_pair.second)
                        {
                            check_and_update(std::dynamic_pointer_cast<TensorPlacePDPD>(
                                port->get_source_tensor()));
                        }
                    }
                }
            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph