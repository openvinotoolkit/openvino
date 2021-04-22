// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "frontend_manager/frontend_manager.hpp"
#include <iostream>

using namespace ngraph::frontend;

//------------------ PLACE wrappers and trampolines -------------------
// Base class for deriving on Python side, not derived from Place
class PyPlace
{
public:
    PyPlace() {}
    virtual ~PyPlace() {}

    virtual bool is_input() const {
        PYBIND11_OVERLOAD_PURE(bool, PyPlace, is_input);
    }

    virtual bool is_output() const {
        PYBIND11_OVERLOAD_PURE(bool, PyPlace, is_output);
    }

    virtual bool is_equal(std::shared_ptr<PyPlace> other) const {
        PYBIND11_OVERLOAD_PURE(bool, PyPlace, is_equal, other);
    }

    virtual std::vector<std::string> get_names() const {
        PYBIND11_OVERLOAD_PURE(std::vector<std::string>, PyPlace, get_names);
    }
};

class InputModelWrapper;

class PlaceWrapper : public Place
{
    std::shared_ptr<PyPlace> m_actual;
    std::weak_ptr<const InputModelWrapper> m_model;
public:
    PlaceWrapper(std::shared_ptr<PyPlace> actual,
                 std::shared_ptr<const InputModelWrapper> model): m_actual(actual), m_model(model) {}
    ~PlaceWrapper() override {}

    std::shared_ptr<PyPlace> getActual() { return m_actual; }
    std::shared_ptr<const InputModelWrapper> getModel() { return m_model.lock(); }

    bool isInput() const override {
        return m_actual->is_input();
    }

    bool isOutput() const override {
        return m_actual->is_output();
    }

    std::vector<std::string> getNames() const override {
        return m_actual->get_names();
    }

    bool isEqual(Place::Ptr place) const override {
        auto w = std::dynamic_pointer_cast<PlaceWrapper>(place);
        if (!w) return false;
        return m_actual->is_equal(w->m_actual);
    }
    friend class InputModelWrapper;
};

//------------------ Input Model wrappers and trampolines -------------------
// Base class for deriving on Python side, not derived from InputModel
class PyInputModel
{
public:
    PyInputModel() {}
    virtual ~PyInputModel() {}

    virtual std::vector<std::shared_ptr<PyPlace>> get_inputs () const {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_inputs);
    }

    virtual std::vector<std::shared_ptr<PyPlace>> get_outputs () const {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_outputs);
    }

    virtual std::shared_ptr<PyPlace> get_place_by_tensor_name (const std::string& tensorName) const {
        PYBIND11_OVERLOAD(std::shared_ptr<PyPlace>, PyInputModel, get_place_by_tensor_name, tensorName);
    }

    virtual void override_all_inputs (const std::vector<std::shared_ptr<PyPlace>>& inputs) {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_inputs, inputs);
    }

    virtual void override_all_outputs (const std::vector<std::shared_ptr<PyPlace>>& outputs) {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_outputs, outputs);
    }

    virtual void extract_subgraph(const std::vector<std::shared_ptr<PyPlace>>& inputs,
                                  const std::vector<std::shared_ptr<PyPlace>>& outputs) {
        PYBIND11_OVERLOAD(void, PyInputModel, extract_subgraph, inputs, outputs);
    }

    virtual void set_partial_shape(std::shared_ptr<PyPlace> place, const ngraph::PartialShape& shape) {
        PYBIND11_OVERLOAD(void, PyInputModel, set_partial_shape, place, shape);
    }
};

class FrontEndWrapper;
class InputModelWrapper : public InputModel, public std::enable_shared_from_this<InputModelWrapper>
{
    std::shared_ptr<PyInputModel> m_actual;
    std::weak_ptr<const FrontEndWrapper> m_frontEnd;
public:
    InputModelWrapper(std::shared_ptr<PyInputModel> actual,
                      std::shared_ptr<const FrontEndWrapper> fe):
            m_actual(actual), m_frontEnd(fe) {}
    ~InputModelWrapper() override {}

    std::shared_ptr<PyInputModel> getActual() { return m_actual; }
    std::shared_ptr<const FrontEndWrapper> getFrontEnd() { return m_frontEnd.lock(); }

    std::vector<Place::Ptr> getInputs() const override {
        auto inputs = m_actual->get_inputs();
        std::vector<Place::Ptr> res;
        for (const auto& input : inputs) {
            res.push_back(std::make_shared<PlaceWrapper>(input, shared_from_this()));
        }
        return res;
    }

    std::vector<Place::Ptr> getOutputs() const override {
        auto outputs = m_actual->get_outputs();
        std::vector<Place::Ptr> res;
        for (const auto& output : outputs) {
            res.push_back(std::make_shared<PlaceWrapper>(output, shared_from_this()));
        }
        return res;
    }

    Place::Ptr getPlaceByTensorName(const std::string& tensorName) const override {
        auto place = m_actual->get_place_by_tensor_name(tensorName);
        return std::make_shared<PlaceWrapper>(place, shared_from_this());
    }

    void setPartialShape(Place::Ptr place, const ngraph::PartialShape& newShape) override {
        auto placeWrapper = std::dynamic_pointer_cast<PlaceWrapper>(place);
        if (!placeWrapper || this != placeWrapper->getModel().get()) {
            throw std::runtime_error("Invalid Place object to override");
        }
        m_actual->set_partial_shape(placeWrapper->m_actual, newShape);
    }

    void overrideAllInputs (const std::vector<Place::Ptr>& places) override {
        std::vector<std::shared_ptr<PyPlace>> pyPlaces;
        for (const auto& place : places) {
            auto placeWrapper = std::dynamic_pointer_cast<PlaceWrapper>(place);
            if (!placeWrapper || this != placeWrapper->getModel().get()) {
                throw std::runtime_error("Invalid Place object to override");
            }
            pyPlaces.push_back(placeWrapper->m_actual);
        }
        m_actual->override_all_inputs(pyPlaces);
    }

    void overrideAllOutputs (const std::vector<Place::Ptr>& places) override {
        std::vector<std::shared_ptr<PyPlace>> pyPlaces;
        for (const auto& place : places) {
            auto placeWrapper = std::dynamic_pointer_cast<PlaceWrapper>(place);
            if (!placeWrapper || this != placeWrapper->getModel().get()) {
                throw std::runtime_error("Invalid Place object to override");
            }
            pyPlaces.push_back(placeWrapper->m_actual);
        }
        m_actual->override_all_outputs(pyPlaces);
    }

    void extractSubgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override {
        std::vector<std::shared_ptr<PyPlace>> pyInputs, pyOutputs;
        for (const auto& input : inputs) {
            auto placeWrapper = std::dynamic_pointer_cast<PlaceWrapper>(input);
            if (!placeWrapper || this != placeWrapper->getModel().get()) {
                throw std::runtime_error("Invalid Input Place object to extract");
            }
            pyInputs.push_back(placeWrapper->m_actual);
        }
        for (const auto& output : outputs) {
            auto placeWrapper = std::dynamic_pointer_cast<PlaceWrapper>(output);
            if (!placeWrapper || this != placeWrapper->getModel().get()) {
                throw std::runtime_error("Invalid Output Place object to extract");
            }
            pyOutputs.push_back(placeWrapper->m_actual);
        }
        m_actual->extract_subgraph(pyInputs, pyOutputs);
    }
};

// -------------- FRONTEND wrappers and trampolines -------------------
// Base class for deriving on Python side, not derived from FrontEnd
class PyFrontEnd
{
public:
    PyFrontEnd() {}
    virtual ~PyFrontEnd() {}

    virtual std::shared_ptr<PyInputModel> load_from_file(const std::string& path) const {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<PyInputModel>, PyFrontEnd, load_from_file, path);
    }

    virtual std::shared_ptr<ngraph::Function> do_convert(std::shared_ptr<PyInputModel> model) const {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<ngraph::Function>, PyFrontEnd, do_convert, model);
    }
};

class FrontEndWrapper : public FrontEnd, public std::enable_shared_from_this<FrontEndWrapper>
{
    std::shared_ptr<PyFrontEnd> m_actual;
public:
    FrontEndWrapper(std::shared_ptr<PyFrontEnd> actual): m_actual(actual) {}
    ~FrontEndWrapper() override {}

    InputModel::Ptr loadFromFile (const std::string& path) const override {
        auto pyModel = m_actual->load_from_file(path);
        return std::make_shared<InputModelWrapper>(pyModel, shared_from_this());
    }

    std::shared_ptr<ngraph::Function> convert (InputModel::Ptr model) const override {
        auto mdlWrapper = std::dynamic_pointer_cast<InputModelWrapper>(model);
        if (!mdlWrapper || this != mdlWrapper->getFrontEnd().get()) {
            throw std::runtime_error("Invalid model to convert with this FrontEnd");
        }
        return m_actual->do_convert(mdlWrapper->getActual());
    }
};

namespace py = pybind11;

void regclass_pyngraph_FrontEndManager(py::module m)
{
    py::class_<ngraph::frontend::FrontEndManager,
               std::shared_ptr<ngraph::frontend::FrontEndManager>>
        fem(m, "FrontEndManager", py::dynamic_attr());
    fem.doc() = "ngraph.impl.FrontEndManager wraps ngraph::frontend::FrontEndManager";

    fem.def(py::init<>());

    fem.def("availableFrontEnds", &ngraph::frontend::FrontEndManager::availableFrontEnds);
    fem.def("registerFrontEnd", [](FrontEndManager& self,
                                          const std::string& name,
                                          std::function<std::shared_ptr<PyFrontEnd>(FrontEndCapabilities)> creator) {
        self.registerFrontEnd(name, [=](FrontEndCapabilities fec) {
            auto pyFE = creator(fec);
            return std::make_shared<FrontEndWrapper>(pyFE);
        });
    });

    fem.def("loadByFramework",
            &ngraph::frontend::FrontEndManager::loadByFramework,
            py::arg("framework"),
            py::arg("capabilities") = ngraph::frontend::FEC_DEFAULT);
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<PyFrontEnd, std::shared_ptr<PyFrontEnd>> pyFE(
        m, "FrontEnd", py::dynamic_attr());
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> wrapper(
            m, "FrontEndWrapper", py::dynamic_attr());
    pyFE.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";
    pyFE.def(py::init<>());

    wrapper.def("loadFromFile", &FrontEnd::loadFromFile, py::arg("path"));
    wrapper.def("convert", [](FrontEnd& self,
            InputModel::Ptr model) -> std::shared_ptr<ngraph::Function> {
        return self.convert(model);
    });
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<Place, std::shared_ptr<Place>> place(
            m, "PlaceWrapper", py::dynamic_attr());
    py::class_<PyPlace, std::shared_ptr<PyPlace>> pyPlace(
            m, "Place", py::dynamic_attr());
    pyPlace.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";
    pyPlace.def(py::init<>());

    place.def("isInput", &ngraph::frontend::Place::isInput);
    place.def("isOutput", &ngraph::frontend::Place::isOutput);
    place.def("getNames", &ngraph::frontend::Place::getNames);
    place.def("isEqual", [](Place& self, std::shared_ptr<Place> other) -> bool {
        return self.isEqual(other);
    });
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<InputModel, std::shared_ptr<InputModel>> im(
            m, "InputModelWrapper", py::dynamic_attr());
    py::class_<PyInputModel, std::shared_ptr<PyInputModel>> pyIM(
        m, "InputModel", py::dynamic_attr());
    pyIM.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";
    pyIM.def(py::init<>());
    im.def("extractSubgraph", &InputModel::extractSubgraph);
    im.def("getPlaceByTensorName", &InputModel::getPlaceByTensorName);
    im.def("setPartialShape", &InputModel::setPartialShape);
    im.def("getInputs", &InputModel::getInputs);
    im.def("getOutputs", &InputModel::getOutputs);
    im.def("overrideAllInputs", &InputModel::overrideAllInputs);
    im.def("overrideAllOutputs", &InputModel::overrideAllOutputs);
}

void regclass_pyngraph_FEC(py::module m)
{
    py::class_<ngraph::frontend::FrontEndCapabilities,
               std::shared_ptr<ngraph::frontend::FrontEndCapabilities>>
        type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = ngraph::frontend::FEC_DEFAULT;
    type.attr("CUT") = ngraph::frontend::FEC_CUT;
    type.attr("NAMES") = ngraph::frontend::FEC_NAMES;
    type.attr("REPLACE") = ngraph::frontend::FEC_REPLACE;
    type.attr("TRAVERSE") = ngraph::frontend::FEC_TRAVERSE;
    type.attr("WILDCARDS") = ngraph::frontend::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const ngraph::frontend::FrontEndCapabilities& a,
           const ngraph::frontend::FrontEndCapabilities& b) { return a == b; },
        py::is_operator());

    type.def("__str__", [](const ngraph::frontend::FrontEndCapabilities& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });
}
