// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_manager.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph
{
    namespace frontend
    {
        class OpPlacePDPD;
        class TensorPlacePDPD;
        class PDPD_API InputModelPDPD : public InputModel
        {
            friend class FrontEndPDPD;
            class InputModelPDPDImpl;
            std::shared_ptr<InputModelPDPDImpl> _impl;

            std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces() const;
            std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarPlaces() const;
            std::map<std::string, Output<Node>> getTensorValues() const;

        public:
            explicit InputModelPDPD(const std::string& path);
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            explicit InputModelPDPD(const std::wstring& path);
#endif
            explicit InputModelPDPD(const std::vector<std::istream*>& streams);
            std::vector<Place::Ptr> get_inputs() const override;
            std::vector<Place::Ptr> get_outputs() const override;
            Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
            void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
            void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
            void extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                  const std::vector<Place::Ptr>& outputs) override;
            void set_partial_shape(Place::Ptr place, const ngraph::PartialShape&) override;
            ngraph::PartialShape get_partial_shape(Place::Ptr place) const override;
            void set_element_type(Place::Ptr place, const ngraph::element::Type&) override;
            void set_tensor_value(Place::Ptr place, const void* value) override;
        };

    } // namespace frontend
} // namespace ngraph
