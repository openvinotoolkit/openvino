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

            void setOp(const std::weak_ptr<OpPlacePDPD>& op) { m_op = op; }

            void setSourceTensor(const std::weak_ptr<TensorPlacePDPD>& source_tensor)
            {
                m_source_tensor = source_tensor;
            }

            std::shared_ptr<TensorPlacePDPD> getSourceTensorPDPD() const;

            std::shared_ptr<OpPlacePDPD> getOp();

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

            void setOp(const std::weak_ptr<OpPlacePDPD>& op) { m_op = op; }

            void setTargetTensor(const std::weak_ptr<TensorPlacePDPD>& target_tensor)
            {
                m_target_tensor = target_tensor;
            }

            std::shared_ptr<TensorPlacePDPD> getTargetTensorPDPD() const;

        private:
            std::weak_ptr<OpPlacePDPD> m_op;
            std::weak_ptr<TensorPlacePDPD> m_target_tensor;
        };

        class OpPlacePDPD : public PlacePDPD
        {
        public:
            OpPlacePDPD(const InputModel& input_model,
                        const std::vector<std::string>& names,
                        const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

            OpPlacePDPD(const InputModel& input_model,
                        const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc);

            void addInPort(const std::shared_ptr<InPortPlacePDPD>& input, const std::string& name)
            {
                m_input_ports[name].push_back(input);
            }

            void addOutPort(const std::shared_ptr<OutPortPlacePDPD>& output,
                            const std::string& name)
            {
                m_output_ports[name].push_back(output);
            }

            const std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>>&
                getOutputPorts() const
            {
                return m_output_ports;
            }

            const std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>>&
                getInputPorts() const
            {
                return m_input_ports;
            }

            std::shared_ptr<OutPortPlacePDPD> getOutputPortPDPD(const std::string& name, int idx)
            {
                return m_output_ports[name][idx];
            }

            std::shared_ptr<InPortPlacePDPD> getInputPortPDPD(const std::string& name, int idx)
            {
                return m_input_ports[name][idx];
            }

            const std::shared_ptr<paddle::framework::proto::OpDesc>& getDesc() const
            {
                return m_op_desc;
            }

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

            void addProducingPort(const std::shared_ptr<OutPortPlacePDPD>& out_port)
            {
                m_producing_ports.push_back(out_port);
            }

            void addConsumingPort(const std::shared_ptr<InPortPlacePDPD>& in_port)
            {
                m_consuming_ports.push_back(in_port);
            }

            std::vector<Place::Ptr> get_consuming_ports() const override;

            Ptr get_producing_port() const override;

            const PartialShape& getPartialShape() const { return m_pshape; }

            const element::Type& getElementType() const { return m_type; }

            void setPartialShape(const PartialShape& pshape) { m_pshape = pshape; }

            void setElementType(const element::Type& type) { m_type = type; }

            const std::shared_ptr<paddle::framework::proto::VarDesc>& getDesc() const
            {
                return m_var_desc;
            }

        private:
            std::shared_ptr<paddle::framework::proto::VarDesc> m_var_desc;
            PartialShape m_pshape;
            element::Type m_type;

            std::vector<std::weak_ptr<OutPortPlacePDPD>> m_producing_ports;
            std::vector<std::weak_ptr<InPortPlacePDPD>> m_consuming_ports;
        };

    } // namespace frontend
} // namespace ngraph
