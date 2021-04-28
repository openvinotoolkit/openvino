// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/ifrontend_manager.hpp"

using namespace ngraph::frontend;

//------------------ PLACE wrapper -------------------
// Base class for deriving on Python side, derived from IPlace
class PyPlace : public IPlace
{
public:
    PyPlace() {}
    ~PyPlace() override {}

    bool isInput() const override { PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_input", isInput); }

    bool isOutput() const override { PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_output", isOutput); }

    bool isEqual(IPlace::Ptr other) const override
    {
        PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_equal", isEqual, other);
    }

    std::vector<std::string> getNames() const override
    {
        PYBIND11_OVERLOAD_NAME(std::vector<std::string>, PyPlace, "get_names", getNames);
    }
};

//------------------ Input Model wrapper -------------------
// Base class for deriving on Python side, derived from IInputModel
class PyInputModel : public IInputModel
{
public:
    PyInputModel() {}
    ~PyInputModel() override {}

    std::vector<IPlace::Ptr> getInputs() const override
    {
        auto inputs =
            get_inputs(); // have to do a conversion of vector<PyPlace::Ptr> to vector<IPlace::Ptr>
        std::vector<IPlace::Ptr> res{inputs.begin(), inputs.end()};
        return res;
    }

    virtual std::vector<std::shared_ptr<PyPlace>> get_inputs() const
    {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_inputs);
    }

    std::vector<IPlace::Ptr> getOutputs() const override
    {
        auto outputs = get_outputs(); // have to do a manual conversion of vector<PyPlace::Ptr> to
                                      // vector<IPlace::Ptr>
        std::vector<IPlace::Ptr> res{outputs.begin(), outputs.end()};
        return res;
    }

    virtual std::vector<std::shared_ptr<PyPlace>> get_outputs() const
    {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_outputs);
    }

    IPlace::Ptr getPlaceByTensorName(const std::string& tensorName) const override
    {
        PYBIND11_OVERLOAD_NAME(std::shared_ptr<PyPlace>,
                               PyInputModel,
                               "get_place_by_tensor_name",
                               getPlaceByTensorName,
                               tensorName);
    }

    void overrideAllInputs(const std::vector<IPlace::Ptr>& inputs) override
    {
        std::vector<std::shared_ptr<PyPlace>> res;
        for (const auto& input : inputs)
        {
            auto pyInput = std::dynamic_pointer_cast<PyPlace>(input);
            if (!pyInput)
            {
                throw "Cannot cast input place to Python representation";
            }
            res.push_back(pyInput);
        }
        override_all_inputs(res);
    }

    virtual void override_all_inputs(const std::vector<std::shared_ptr<PyPlace>>& inputs) const
    {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_inputs, inputs);
    }

    void overrideAllOutputs(const std::vector<IPlace::Ptr>& outputs) override
    {
        std::vector<std::shared_ptr<PyPlace>> res;
        for (const auto& output : outputs)
        {
            auto pyOutput = std::dynamic_pointer_cast<PyPlace>(output);
            if (!pyOutput)
            {
                throw "Cannot cast output place to Python representation";
            }
            res.push_back(pyOutput);
        }
        override_all_outputs(res);
    }

    virtual void override_all_outputs(const std::vector<std::shared_ptr<PyPlace>>& outputs) const
    {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_outputs, outputs);
    }

    void extractSubgraph(const std::vector<IPlace::Ptr>& inputs,
                         const std::vector<IPlace::Ptr>& outputs) override
    {
        std::vector<std::shared_ptr<PyPlace>> resIn, resOut;
        for (const auto& input : inputs)
        {
            auto pyInput = std::dynamic_pointer_cast<PyPlace>(input);
            if (!pyInput)
            {
                throw "Cannot cast input place to Python representation";
            }
            resIn.push_back(pyInput);
        }
        for (const auto& output : outputs)
        {
            auto pyOutput = std::dynamic_pointer_cast<PyPlace>(output);
            if (!pyOutput)
            {
                throw "Cannot cast output place to Python representation";
            }
            resOut.push_back(pyOutput);
        }
        extract_subgraph(resIn, resOut);
    }

    virtual void extract_subgraph(const std::vector<std::shared_ptr<PyPlace>>& inputs,
                                  const std::vector<std::shared_ptr<PyPlace>>& outputs) const
    {
        PYBIND11_OVERLOAD(void, PyInputModel, extract_subgraph, inputs, outputs);
    }

    void setPartialShape(IPlace::Ptr place, const ngraph::PartialShape& shape) override
    {
        PYBIND11_OVERLOAD_NAME(
            void, PyInputModel, "set_partial_shape", setPartialShape, place, shape);
    }
};

// -------------- FRONTEND wrapper -------------------
// Base class for deriving on Python side, derived from IFrontEnd
class PyFrontEnd : public IFrontEnd
{
public:
    PyFrontEnd() {}
    ~PyFrontEnd() override {}

    IInputModel::Ptr loadFromFile(const std::string& path) const override
    {
        PYBIND11_OVERLOAD_NAME(
            std::shared_ptr<PyInputModel>, PyFrontEnd, "load_from_file", loadFromFile, path);
    }

    std::shared_ptr<ngraph::Function> convert(IInputModel::Ptr model) const override
    {
        PYBIND11_OVERLOAD(std::shared_ptr<ngraph::Function>, PyFrontEnd, convert, model);
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

    fem.def("available_front_ends", &ngraph::frontend::FrontEndManager::availableFrontEnds);
    fem.def(
        "register_front_end",
        [](FrontEndManager& self,
           const std::string& name,
           std::function<std::shared_ptr<PyFrontEnd>(FrontEndCapabilities)> creator) {
            self.registerFrontEnd(
                name, [=](FrontEndCapabilities fec) -> IFrontEnd::Ptr { return creator(fec); });
        },
        py::arg("name"),
        py::arg("creator"));

    fem.def(
        "load_by_framework",
        &FrontEndManager::loadByFramework,
        py::arg("framework"),
        py::arg_v("capabilities", ngraph::frontend::FrontEndCapabilities::FEC_DEFAULT, "FrontEndCapabilities.DEFAULT"));
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<PyFrontEnd, std::shared_ptr<PyFrontEnd>> pyFrontEnd(
        m, "IFrontEnd", py::dynamic_attr());
    pyFrontEnd.doc() = "Base class for FrontEnd custom implementation";
    pyFrontEnd.def(py::init<>());

    py::class_<FrontEnd> frontEnd(m, "FrontEnd", py::dynamic_attr());
    frontEnd.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    frontEnd.def("load_from_file", &FrontEnd::loadFromFile, py::arg("path"));
    frontEnd.def(
        "convert",
        [](FrontEnd& self, const InputModel& model) -> std::shared_ptr<ngraph::Function> {
            return self.convert(model);
        },
        py::arg("model"));
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<PyInputModel, std::shared_ptr<PyInputModel>> pyInputModel(
        m, "IInputModel", py::dynamic_attr());
    pyInputModel.doc() = "Base class for custom input model";
    pyInputModel.def(py::init<>());

    py::class_<InputModel> inputModel(m, "InputModel", py::dynamic_attr());
    inputModel.doc() = "ngraph.impl.InputModel wraps ngraph::frontend::InputModel";

    inputModel.def("get_place_by_tensor_name", &InputModel::getPlaceByTensorName, py::arg("name"));
    inputModel.def(
        "set_partial_shape", &InputModel::setPartialShape, py::arg("place"), py::arg("shape"));
    inputModel.def("get_inputs", &InputModel::getInputs);
    inputModel.def("get_outputs", &InputModel::getOutputs);
    inputModel.def("override_all_inputs", &InputModel::overrideAllInputs, py::arg("inputs"));
    inputModel.def("override_all_outputs", &InputModel::overrideAllOutputs, py::arg("outputs"));
    inputModel.def(
        "extract_subgraph", &InputModel::extractSubgraph, py::arg("inputs"), py::arg("outputs"));
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<PyPlace, std::shared_ptr<PyPlace>> pyPlace(m, "IPlace", py::dynamic_attr());
    pyPlace.doc() = "Base class for custom place implementation";
    pyPlace.def(py::init<>());

    py::class_<Place, std::shared_ptr<Place>> place(m, "Place", py::dynamic_attr());
    place.doc() = "ngraph.impl.Place wraps ngraph::frontend::Place";

    place.def("is_input", &ngraph::frontend::Place::isInput);
    place.def("is_output", &ngraph::frontend::Place::isOutput);
    place.def("get_names", &ngraph::frontend::Place::getNames);
    place.def("is_equal", &ngraph::frontend::Place::isEqual, py::arg("other"));
}

void regclass_pyngraph_FEC(py::module m)
{
    py::class_<FrontEndCapabilities,
               std::shared_ptr<FrontEndCapabilities>>
        type(m, "FrontEndCapabilities");
    // type.doc() = "FrontEndCapabilities";
    type.attr("DEFAULT") = FrontEndCapabilities::FEC_DEFAULT;
    type.attr("CUT") = FrontEndCapabilities::FEC_CUT;
    type.attr("NAMES") = FrontEndCapabilities::FEC_NAMES;
    type.attr("REPLACE") = FrontEndCapabilities::FEC_REPLACE;
    type.attr("TRAVERSE") = FrontEndCapabilities::FEC_TRAVERSE;
    type.attr("WILDCARDS") = FrontEndCapabilities::FEC_WILDCARDS;

    type.def(
        "__eq__",
        [](const FrontEndCapabilities& a,
           const FrontEndCapabilities& b) { return a == b; },
        py::is_operator());

    type.def("__str__", [](const FrontEndCapabilities& self) -> std::string {
        std::stringstream ss;
        ss << static_cast<int>(self);
        return ss.str();
    });
}
