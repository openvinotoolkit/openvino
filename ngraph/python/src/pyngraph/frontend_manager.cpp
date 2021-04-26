// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/ifrontend_manager.hpp"

using namespace ngraph::frontend;

//------------------ PLACE wrapper -------------------
// Base class for deriving on Python side, derived from IPlace
class PyPlace: public IPlace
{
public:
    PyPlace() {}
    ~PyPlace() override {}

    bool isInput() const override {
        PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_input", isInput);
    }

    bool isOutput() const override {
        PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_output", isOutput);
    }

    bool isEqual(IPlace::Ptr other) const override {
        PYBIND11_OVERLOAD_NAME(bool, PyPlace, "is_equal", isEqual, other);
    }

    std::vector<std::string> getNames() const override {
        PYBIND11_OVERLOAD_NAME(std::vector<std::string>, PyPlace, "get_names", getNames);
    }
};

//------------------ Input Model wrapper -------------------
// Base class for deriving on Python side, derived from IInputModel
class PyInputModel: public IInputModel
{
public:
    PyInputModel() {}
    ~PyInputModel() override {}

    std::vector<IPlace::Ptr> getInputs () const override {
        auto inputs = get_inputs(); // have to do a conversion of vector<PyPlace::Ptr> to vector<IPlace::Ptr>
        std::vector<IPlace::Ptr> res {inputs.begin(), inputs.end()};
        return res;
    }

    virtual std::vector<std::shared_ptr<PyPlace>> get_inputs() const {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_inputs);
    }

    std::vector<IPlace::Ptr> getOutputs () const override {
        auto outputs = get_outputs(); // have to do a manual conversion of vector<PyPlace::Ptr> to vector<IPlace::Ptr>
        std::vector<IPlace::Ptr> res {outputs.begin(), outputs.end()};
        return res;
    }

    virtual std::vector<std::shared_ptr<PyPlace>> get_outputs() const {
        PYBIND11_OVERLOAD(std::vector<std::shared_ptr<PyPlace>>, PyInputModel, get_outputs);
    }

    IPlace::Ptr getPlaceByTensorName (const std::string& tensorName) const override {
        PYBIND11_OVERLOAD_NAME(std::shared_ptr<PyPlace>, PyInputModel, "get_place_by_tensor_name", getPlaceByTensorName, tensorName);
    }

    void overrideAllInputs (const std::vector<IPlace::Ptr>& inputs) override {
        std::vector<std::shared_ptr<PyPlace>> res;
        for (const auto& input : inputs) {
            auto pyInput = std::dynamic_pointer_cast<PyPlace>(input);
            if (!pyInput) {
                throw "Cannot cast input place to Python representation";
            }
            res.push_back(pyInput);
        }
        override_all_inputs(res);
    }

    virtual void override_all_inputs(const std::vector<std::shared_ptr<PyPlace>>& inputs) const {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_inputs, inputs);
    }

    void overrideAllOutputs (const std::vector<IPlace::Ptr>& outputs) override {
        std::vector<std::shared_ptr<PyPlace>> res;
        for (const auto& output : outputs) {
            auto pyOutput = std::dynamic_pointer_cast<PyPlace>(output);
            if (!pyOutput) {
                throw "Cannot cast output place to Python representation";
            }
            res.push_back(pyOutput);
        }
        override_all_outputs(res);
    }

    virtual void override_all_outputs(const std::vector<std::shared_ptr<PyPlace>>& outputs) const {
        PYBIND11_OVERLOAD(void, PyInputModel, override_all_outputs, outputs);
    }

    void extractSubgraph(const std::vector<IPlace::Ptr>& inputs,
                         const std::vector<IPlace::Ptr>& outputs) override{
        std::vector<std::shared_ptr<PyPlace>> resIn, resOut;
        for (const auto& input : inputs) {
            auto pyInput = std::dynamic_pointer_cast<PyPlace>(input);
            if (!pyInput) {
                throw "Cannot cast input place to Python representation";
            }
            resIn.push_back(pyInput);
        }
        for (const auto& output: outputs) {
            auto pyOutput = std::dynamic_pointer_cast<PyPlace>(output);
            if (!pyOutput) {
                throw "Cannot cast output place to Python representation";
            }
            resOut.push_back(pyOutput);
        }
        extract_subgraph(resIn, resOut);
    }

    virtual void extract_subgraph(const std::vector<std::shared_ptr<PyPlace>>& inputs,
                                  const std::vector<std::shared_ptr<PyPlace>>& outputs) const {
        PYBIND11_OVERLOAD(void, PyInputModel, extract_subgraph, inputs, outputs);
    }

    void setPartialShape(IPlace::Ptr place, const ngraph::PartialShape& shape) override {
        PYBIND11_OVERLOAD_NAME(void, PyInputModel, "set_partial_shape", setPartialShape, place, shape);
    }
};

// -------------- FRONTEND wrapper -------------------
// Base class for deriving on Python side, derived from IFrontEnd
class PyFrontEnd: public IFrontEnd
{
public:
    PyFrontEnd() {}
    ~PyFrontEnd() override {}

    IInputModel::Ptr loadFromFile(const std::string& path) const override {
        PYBIND11_OVERLOAD_NAME(std::shared_ptr<PyInputModel>, PyFrontEnd, "load_from_file", loadFromFile, path);
    }

    std::shared_ptr<ngraph::Function> convert(IInputModel::Ptr model) const override {
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
    fem.def("register_front_end", [](FrontEndManager& self,
                                          const std::string& name,
                                          std::function<std::shared_ptr<PyFrontEnd>(FrontEndCapabilities)> creator) {
        self.registerFrontEnd(name, [=](FrontEndCapabilities fec) -> IFrontEnd::Ptr {
            return creator(fec);
        });
    });

    fem.def("load_by_framework", [](FrontEndManager& self, const std::string& name, FrontEndCapabilities caps) {
        return to_shared(self.loadByFramework(name, caps));
    }, py::arg("framework"), py::arg("capabilities") = ngraph::frontend::FEC_DEFAULT);
}

void regclass_pyngraph_FrontEnd(py::module m)
{
    py::class_<PyFrontEnd, std::shared_ptr<PyFrontEnd>> pyFE(
        m, "IFrontEnd", py::dynamic_attr());
    py::class_<FrontEndShared, std::shared_ptr<FrontEndShared>> wrapper(
            m, "FrontEnd", py::dynamic_attr());
    pyFE.doc() = "Base class for FrontEnd custom implementation";
    pyFE.def(py::init<>());

    wrapper.def("load_from_file", [](FrontEndShared& self, const std::string& path) -> std::shared_ptr<InputModelShared> {
        auto res = to_shared(self.frontEnd.loadFromFile(path));
        return res;
        }, py::arg("path"));
    wrapper.def("convert", [](FrontEndShared& self,
            const InputModelShared& model) -> std::shared_ptr<ngraph::Function> {
        return self.frontEnd.convert(model.inputModel);
    });
}

void regclass_pyngraph_Place(py::module m)
{
    py::class_<Place, std::shared_ptr<Place>> place(
            m, "Place", py::dynamic_attr());
    py::class_<PyPlace, std::shared_ptr<PyPlace>> pyPlace(
            m, "IPlace", py::dynamic_attr());
    pyPlace.doc() = "Base class for custom place implementation";
    pyPlace.def(py::init<>());

    place.def("is_input", &ngraph::frontend::Place::isInput);
    place.def("is_output", &ngraph::frontend::Place::isOutput);
    place.def("get_names", &ngraph::frontend::Place::getNames);
    place.def("is_equal", &ngraph::frontend::Place::isEqual, py::arg("other"));
}

void regclass_pyngraph_InputModel(py::module m)
{
    py::class_<InputModelShared, std::shared_ptr<InputModelShared>> im(
            m, "InputModel", py::dynamic_attr());
    py::class_<PyInputModel, std::shared_ptr<PyInputModel>> pyIM(
        m, "IInputModel", py::dynamic_attr());
    pyIM.doc() = "Base class for custom input model";
    pyIM.def(py::init<>());
    im.def("extract_subgraph", [](InputModelShared& self, const std::vector<std::shared_ptr<Place>>& inPtrs,
            const std::vector<std::shared_ptr<Place>>& outPtrs) {
        std::vector<Place> inputs, outputs;
        for (const auto& inPtr : inPtrs) {
            inputs.push_back(*inPtr);
        }
        for (const auto& outPtr : outPtrs) {
            outputs.push_back(*outPtr);
        }
        self.inputModel.extractSubgraph(inputs, outputs);
    });

    im.def("get_place_by_tensor_name", [](InputModelShared& self, const std::string& name) -> std::shared_ptr<Place> {
        return std::make_shared<Place>(self.inputModel.getPlaceByTensorName(name));
    });

    im.def("set_partial_shape", [](InputModelShared& self, std::shared_ptr<Place> place,
                                        const ngraph::PartialShape& shape) {
        self.inputModel.setPartialShape(*place, shape);
    });

    im.def("get_inputs", [](InputModelShared& self) {
        return self.inputModel.getInputs();
    });

    im.def("get_outputs", [](InputModelShared& self) {
        return self.inputModel.getOutputs();
    });

    im.def("override_all_inputs", [](InputModelShared& self,
                                   const std::vector<Place>& inputs) {
        return self.inputModel.overrideAllInputs(inputs);
    });

    im.def("override_all_outputs", [](InputModelShared& self,
                                   const std::vector<Place>& outputs) {
        return self.inputModel.overrideAllOutputs(outputs);
    });
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
