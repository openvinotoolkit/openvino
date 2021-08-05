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

            void traverse_down(const std::vector<Place::Ptr>& start_nodes,
                               std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                               std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors)
            {
                std::queue<OpPlacePDPD*> q;
                std::unordered_set<OpPlacePDPD*> visited;

                auto check_and_update = [&](const std::shared_ptr<OpPlacePDPD>& op) -> bool {
                    if (op && !visited.count(op.get()))
                    {
                        visited.insert(op.get());
                        q.push(op.get());
                        if (ordered_ops)
                            ordered_ops->push_back(op);
                        return true;
                    }
                    return false;
                };

                for (const auto& node : start_nodes)
                {
                    if (!check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(node)) &&
                        !node->is_output())
                    {
                        if (ordered_tensors)
                            ordered_tensors->push_back(pdpd::castToTensorPlace(node));
                        for (const auto& op : node->get_consuming_operations())
                        {
                            auto pdpd_output_op = std::dynamic_pointer_cast<OpPlacePDPD>(op);
                            PDPD_ASSERT(pdpd_output_op != nullptr, "Invalid consuming operation");
                            check_and_update(pdpd_output_op);
                        }
                    }
                }
                while (!q.empty())
                {
                    auto p_op = q.front();
                    q.pop();
                    for (const auto& map_pair : p_op->get_output_ports())
                    {
                        for (const auto& port : map_pair.second)
                        {
                            auto tensor = std::dynamic_pointer_cast<TensorPlacePDPD>(
                                port->get_target_tensor());
                            if (tensor && !tensor->is_output())
                            {
                                if (ordered_tensors)
                                    ordered_tensors->push_back(tensor);
                                for (const auto& op : tensor->get_consuming_operations())
                                {
                                    check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(op));
                                }
                            }
                        }
                    }
                }
            }

            void traverse_up(const std::vector<Place::Ptr>& start_nodes,
                             std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                             std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors)
            {
                std::queue<OpPlacePDPD*> q;
                std::unordered_set<OpPlacePDPD*> visited;

                auto check_and_update = [&](const std::shared_ptr<OpPlacePDPD>& op) -> bool {
                    if (op && !visited.count(op.get()))
                    {
                        visited.insert(op.get());
                        q.push(op.get());
                        if (ordered_ops)
                            ordered_ops->push_back(op);
                        return true;
                    }
                    return false;
                };

                for (const auto& node : start_nodes)
                {
                    if (!check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(node)) &&
                        !node->is_input())
                    {
                        if (ordered_tensors)
                            ordered_tensors->push_back(pdpd::castToTensorPlace(node));
                        auto pdpd_output_op =
                            std::dynamic_pointer_cast<OpPlacePDPD>(node->get_producing_operation());
                        FRONT_END_GENERAL_CHECK(pdpd_output_op != nullptr,
                                                "Output doesn't have producing operation");
                        check_and_update(pdpd_output_op);
                    }
                }
                while (!q.empty())
                {
                    auto p_op = q.front();
                    q.pop();
                    for (const auto& map_pair : p_op->get_input_ports())
                    {
                        for (const auto& port : map_pair.second)
                        {
                            auto tensor = std::dynamic_pointer_cast<TensorPlacePDPD>(
                                port->get_source_tensor());
                            if (tensor && !tensor->is_input())
                            {
                                if (ordered_tensors)
                                    ordered_tensors->push_back(tensor);
                                check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(
                                    tensor->get_producing_operation()));
                            }
                        }
                    }
                }
            }
        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph