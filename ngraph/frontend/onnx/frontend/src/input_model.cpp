// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>
#include <input_model.hpp>
#include <place.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

InputModelONNX::InputModelONNX(const std::string& path)
    : m_editor(path)
{
}

std::vector<Place::Ptr> InputModelONNX::get_inputs() const
{
    auto inputs = m_editor.model_inputs();
    std::vector<Place::Ptr> ret;
    ret.reserve(inputs.size());
    for (const auto& input : inputs)
    {
        ret.push_back(std::make_shared<PlaceTensorONNX>(input, m_editor));
    }
    return ret;
}

Place::Ptr InputModelONNX::get_place_by_tensor_name(const std::string& tensor_name) const
{
    return std::make_shared<PlaceTensorONNX>(tensor_name, m_editor);
}

void InputModelONNX::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& shape)
{
    std::map<std::string, ngraph::PartialShape> m;
    m[place->get_names()[0]] = shape;
    m_editor.set_input_shapes(m);
}

void InputModelONNX::set_element_type(Place::Ptr place, const ngraph::element::Type& type)
{
    std::map<std::string, ngraph::element::Type_t> m;
    m[place->get_names()[0]] = type;
    m_editor.set_input_types(m);
}

std::shared_ptr<Function> InputModelONNX::decode()
{
    return m_editor.decode();
}

std::shared_ptr<Function> InputModelONNX::convert()
{
    return m_editor.get_function();
}
