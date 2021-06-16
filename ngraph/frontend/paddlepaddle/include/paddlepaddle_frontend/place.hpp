// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include <paddlepaddle_frontend/exceptions.hpp>

namespace paddle
{
    namespace framework
    {
        namespace proto
        {
            class OpDesc;
            class VarDesc;

        } // namespace proto
    }     // namespace framework
} // namespace paddle

namespace ngraph
{
    namespace frontend
    {
        class TensorPlacePDPD;
        class OpPlacePDPD;

        class PlacePDPD : public Place
        {
        public:
            PlacePDPD(const InputModel& input_model, const std::vector<std::string>& names)
                : m_input_model(input_model)
                , m_names(names)
            {
            }

            explicit PlacePDPD(const InputModel& input_model)
                : PlacePDPD(input_model, std::vector<std::string>{})
            {
            }

            ~PlacePDPD() override = default;

            bool is_input() const override;
            bool is_output() const override;
            bool is_equal(Ptr another) const override { return this == another.get(); }

            std::vector<std::string> get_names() const override { return m_names; }

        private:
            const InputModel& m_input_model;
            std::vector<std::string> m_names;
        };

        class InPortPlacePDPD : public PlacePDPD
        {
        public:
            explicit InPortPlacePDPD(const InputModel& input_model)
                : PlacePDPD(input_model)
            {
            }

            void set_op(const std::weak_ptr<OpPlacePDPD>& op) { m_op = op; }
            void set_source_tensor(const std::weak_ptr<TensorPlacePDPD>& source_tensor);

            // Internal usage
            std::shared_ptr<TensorPlacePDPD> get_source_tensor_pdpd() const;
            std::shared_ptr<OpPlacePDPD> get_op();

            // External usage
            std::vector<Ptr> get_consuming_operations() const override;
            Ptr get_producing_operation() const override;
            Place::Ptr get_source_tensor() const override;
            Ptr get_producing_port() const override;

            bool is_equal_data(Ptr another) const override;

        private:
            std::weak_ptr<TensorPlacePDPD> m_source_tensor;
            std::weak_ptr<OpPlacePDPD> m_op;
        };

        class OutPortPlacePDPD : public PlacePDPD
        {
        public:
            explicit OutPortPlacePDPD(const InputModel& input_model)
                : PlacePDPD(input_model)
            {
            }

            void set_op(const std::weak_ptr<OpPlacePDPD>& op) { m_op = op; }
            void set_target_tensor(const std::weak_ptr<TensorPlacePDPD>& target_tensor);

            std::shared_ptr<TensorPlacePDPD> get_target_tensor_pdpd() const;

            // External usage
            std::vector<Ptr> get_consuming_operations() const override;
            Place::Ptr get_producing_operation() const override;
            std::vector<Place::Ptr> get_consuming_ports() const override;
            Ptr get_target_tensor() const override;
            bool is_equal_data(Ptr another) const override;

        private:
            std::weak_ptr<OpPlacePDPD> m_op;
            std::weak_ptr<TensorPlacePDPD> m_target_tensor;
        };

        class OpPlacePDPD : public PlacePDPD
        {
        public:
            OpPlacePDPD(const InputModel& input_model,
                        const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc,
                        const std::vector<std::string>& names);

            OpPlacePDPD(const InputModel& input_model,
                        const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

            void add_in_port(const std::shared_ptr<InPortPlacePDPD>& input,
                             const std::string& name);
            void add_out_port(const std::shared_ptr<OutPortPlacePDPD>& output,
                              const std::string& name);

            // Internal usage
            const std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>>&
                get_output_ports() const;
            const std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>>&
                get_input_ports() const;
            std::shared_ptr<OutPortPlacePDPD> get_output_port_pdpd(const std::string& name,
                                                                   int idx) const;
            std::shared_ptr<InPortPlacePDPD> get_input_port_pdpd(const std::string& name,
                                                                 int idx) const;
            const std::shared_ptr<paddle::framework::proto::OpDesc>& get_desc() const;

            // External API methods
            std::vector<Place::Ptr> get_consuming_ports() const override;

            Ptr get_output_port(const std::string& outputPortName,
                                int outputPortIndex) const override;
            Ptr get_output_port(const std::string& outputPortName) const override;
            Ptr get_output_port() const override;

            Ptr get_input_port(const std::string& inputName, int inputPortIndex) const override;
            Ptr get_input_port(const std::string& inputName) const override;
            Ptr get_input_port() const override;

            std::vector<Place::Ptr> get_consuming_operations() const override;
            std::vector<Place::Ptr>
                get_consuming_operations(const std::string& outputPortName) const override;
            std::vector<Place::Ptr> get_consuming_operations(const std::string& outputPortName,
                                                             int outputPortIndex) const override;

            Place::Ptr get_producing_operation() const override;
            Place::Ptr get_producing_operation(const std::string& inputName) const override;
            Place::Ptr get_producing_operation(const std::string& inputName,
                                               int inputPortIndex) const override;

            Place::Ptr get_source_tensor() const override;
            Place::Ptr get_source_tensor(const std::string& inputName) const override;
            Place::Ptr get_source_tensor(const std::string& inputName,
                                         int inputPortIndex) const override;

            Place::Ptr get_target_tensor() const override;
            Place::Ptr get_target_tensor(const std::string& outputName) const override;
            Place::Ptr get_target_tensor(const std::string& outputName,
                                         int outputPortIndex) const override;

        private:
            std::shared_ptr<paddle::framework::proto::OpDesc> m_op_desc;
            std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>> m_input_ports;
            std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>> m_output_ports;
        };

        class TensorPlacePDPD : public PlacePDPD
        {
        public:
            TensorPlacePDPD(const InputModel& input_model,
                            const std::vector<std::string>& names,
                            const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

            TensorPlacePDPD(const InputModel& input_model,
                            const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc);

            void add_producing_port(const std::shared_ptr<OutPortPlacePDPD>& out_port);
            void add_consuming_port(const std::shared_ptr<InPortPlacePDPD>& in_port);

            // Internal usage
            const PartialShape& get_partial_shape() const { return m_pshape; }
            const element::Type& get_element_type() const { return m_type; }
            void set_partial_shape(const PartialShape& pshape) { m_pshape = pshape; }
            void set_element_type(const element::Type& type) { m_type = type; }
            const std::shared_ptr<paddle::framework::proto::VarDesc>& get_desc() const;

            // External usage
            Ptr get_producing_operation() const override;
            std::vector<Place::Ptr> get_consuming_operations() const override;
            std::vector<Place::Ptr> get_consuming_ports() const override;
            Ptr get_producing_port() const override;
            bool is_equal_data(Ptr another) const override;

        private:
            std::shared_ptr<paddle::framework::proto::VarDesc> m_var_desc;
            PartialShape m_pshape;
            element::Type m_type;

            std::vector<std::weak_ptr<OutPortPlacePDPD>> m_producing_ports;
            std::vector<std::weak_ptr<InPortPlacePDPD>> m_consuming_ports;
        };

    } // namespace frontend
} // namespace ngraph
