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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ngraph
{
    namespace test
    {
        class ValueHolder
        {
            template <typename T>
            T& invalid()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }

        public:
            virtual ~ValueHolder() {}
            virtual operator bool&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator float&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator double&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::string&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int8_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int16_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int32_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator int64_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint8_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint16_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint32_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator uint64_t&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<std::string>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<float>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<double>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int8_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int16_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int32_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<int64_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<uint8_t>&() { NGRAPH_CHECK(false, "Invalid type access"); }
            virtual operator std::vector<uint16_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<uint32_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator std::vector<uint64_t>&()
            {
                NGRAPH_CHECK(false, "Invalid type access");
            }
            virtual operator HostTensorPtr&() { NGRAPH_CHECK(false, "Invalid type access"); }
            uint64_t get_index() { return m_index; }

        protected:
            uint64_t m_index{0};
        };

        template <typename T>
        class ValueHolderImp : public ValueHolder
        {
        public:
            ValueHolderImp(const T& value, uint64_t index)
                : m_value(value)
            {
                m_index = index;
            }
            operator T&() override { return m_value; }

        protected:
            T m_value;
        };

        class ValueMap
        {
            using map_type = std::unordered_map<std::string, std::shared_ptr<ValueHolder>>;

        public:
            /// \brief Set to print serialization information
            void set_print(bool value) { m_print = value; }
            template <typename T>
            void insert(const std::string& name, const T& value)
            {
                std::pair<map_type::iterator, bool> result = m_values.insert(map_type::value_type(
                    name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
                NGRAPH_CHECK(result.second, name, " is already in use");
            }
            template <typename T>
            void insert_scalar(const std::string& name, const T& value)
            {
                std::pair<map_type::iterator, bool> result = m_values.insert(map_type::value_type(
                    name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
                NGRAPH_CHECK(result.second, name, " is already in use");
                if (m_print)
                {
                    std::cerr << "SER: " << name << " = " << value << std::endl;
                }
            }
            template <typename T>
            void insert_vector(const std::string& name, const T& value)
            {
                std::pair<map_type::iterator, bool> result = m_values.insert(map_type::value_type(
                    name, std::make_shared<ValueHolderImp<T>>(value, m_write_count++)));
                NGRAPH_CHECK(result.second, name, " is already in use");
                if (m_print)
                {
                    std::cerr << "SER: " << name << " = [";
                    std::string comma = "";
                    for (auto val : value)
                    {
                        std::cerr << comma << val;
                        comma = ", ";
                    }
                    std::cerr << "]" << std::endl;
                }
            }
            template <typename T>
            T& get(const std::string& name)
            {
                auto& value_holder = *m_values.at(name);
                NGRAPH_CHECK(m_read_count++ == value_holder.get_index());
                return static_cast<T&>(*m_values.at(name));
            }

        protected:
            map_type m_values;
            uint64_t m_write_count{0};
            uint64_t m_read_count{0};
            bool m_print{false};
        };

        class DeserializeAttributeVisitor : public AttributeVisitor
        {
        public:
            DeserializeAttributeVisitor(ValueMap& value_map)
                : m_values(value_map)
            {
            }
            void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
            {
                if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<
                        std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter))
                {
                    auto& data = m_values.get<HostTensorPtr>(name);
                    data->read(a->get()->get_ptr(), a->get()->size());
                }
                else
                {
                    NGRAPH_CHECK(false, "Attribute \"", name, "\" cannot be unmarshalled");
                }
            }
            // The remaining adapter methods fall back on the void adapter if not implemented
            void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
            {
                adapter.set(m_values.get<std::string>(name));
            };
            void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
            {
                adapter.set(m_values.get<bool>(name));
            };
            void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
            {
                adapter.set(m_values.get<int64_t>(name));
            }
            void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
            {
                adapter.set(m_values.get<double>(name));
            }

            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int8_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int8_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int16_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int16_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int32_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int32_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int64_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<int64_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint8_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint8_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint16_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint16_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint32_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint32_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint64_t>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<uint64_t>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<std::string>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<std::string>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<float>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<float>>(name));
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<double>>& adapter) override
            {
                adapter.set(m_values.get<std::vector<double>>(name));
            }
            void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override
            {
                HostTensorPtr& data = m_values.get<HostTensorPtr>(name);
                data->read(adapter.get_ptr(), adapter.size());
            }

        protected:
            ValueMap& m_values;
        };

        class SerializeAttributeVisitor : public AttributeVisitor
        {
        public:
            SerializeAttributeVisitor(ValueMap& value_map)
                : m_values(value_map)
            {
            }

            void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
            {
                if (auto a = ::ngraph::as_type<::ngraph::AttributeAdapter<
                        std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter))
                {
                    HostTensorPtr data =
                        std::make_shared<HostTensor>(element::u8, Shape{a->get()->size()});
                    data->write(a->get()->get_ptr(), a->get()->size());
                    m_values.insert(name, data);
                }
                else
                {
                    NGRAPH_CHECK(false, "Attribute \"", name, "\" cannot be marshalled");
                }
            }
            // The remaining adapter methods fall back on the void adapter if not implemented
            void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
            {
                m_values.insert_scalar(name, adapter.get());
            };
            void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
            {
                m_values.insert_scalar(name, adapter.get());
            };

            void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
            {
                m_values.insert_scalar(name, adapter.get());
            }
            void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
            {
                m_values.insert_scalar(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<std::string>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<float>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<double>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int8_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int16_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int32_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<int64_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint8_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint16_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint32_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name,
                            ValueAccessor<std::vector<uint64_t>>& adapter) override
            {
                m_values.insert_vector(name, adapter.get());
            }
            void on_adapter(const std::string& name, ValueAccessor<void*>& adapter) override
            {
                HostTensorPtr data =
                    std::make_shared<HostTensor>(element::u8, Shape{adapter.size()});
                data->write(adapter.get_ptr(), adapter.size());
                m_values.insert(name, data);
            }

        protected:
            ValueMap& m_values;
        };

        class NodeBuilder : public ValueMap, public DeserializeAttributeVisitor
        {
        public:
            NodeBuilder()
                : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this))
                , m_serializer(*this)
            {
            }

            NodeBuilder(const std::shared_ptr<Node>& node)
                : DeserializeAttributeVisitor(static_cast<ValueMap&>(*this))
                , m_serializer(*this)
            {
                save_node(node);
            }

            void save_node(std::shared_ptr<Node> node)
            {
                m_node_type_info = node->get_type_info();
                node->visit_attributes(m_serializer);
            }

            // Does not validate, since inputs aren't set
            std::shared_ptr<Node> create()
            {
                std::shared_ptr<Node> node(get_ops().create(m_node_type_info));
                node->visit_attributes(*this);
                return node;
            }
            AttributeVisitor& get_node_saver() { return m_serializer; }
            AttributeVisitor& get_node_loader() { return *this; }
            static FactoryRegistry<Node>& get_ops()
            {
                static std::shared_ptr<FactoryRegistry<Node>> registry;
                static std::mutex init_guard;
                if (!registry)
                {
                    std::lock_guard<std::mutex> guard(init_guard);
                    if (!registry)
                    {
                        registry = std::make_shared<FactoryRegistry<Node>>();
#define NGRAPH_OP(NAME, NAMESPACE, VERSION) registry->register_factory<NAMESPACE::NAME>();
#include "op_version_tbl.hpp"
#undef NGRAPH_OP
                    }
                }
                return *registry;
            }

        protected:
            Node::type_info_t m_node_type_info;
            SerializeAttributeVisitor m_serializer;
        };
    }
}
