// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/env_util.hpp>
#include <ngraph/except.hpp>

#include "frontend_manager/frontend_manager.hpp"
#include "plugin_loader.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

#define FRONT_END_NOT_IMPLEMENTED(NAME)                                                            \
    throw std::runtime_error(#NAME " is not implemented for this FrontEnd class")
#define FRONT_END_ASSERT(EXPRESSION)                                                               \
    {                                                                                              \
        if (!(EXPRESSION))                                                                         \
            throw "AssertionFailed";                                                               \
    }

//----------- FrontEndManager ---------------------------
class FrontEndManager::Impl
{
    std::vector<PluginHandle> m_loadedLibs; // must be a first class member (destroyed last)
    std::map<std::string, FrontEndFactory> m_factories;

public:
    Impl() { registerPlugins(); }

    ~Impl() = default;

    FrontEnd::Ptr loadByFramework(const std::string& framework, FrontEndCapFlags fec)
    {
        FRONT_END_ASSERT(m_factories.count(framework))
        return m_factories[framework](fec);
    }

    std::vector<std::string> availableFrontEnds() const
    {
        std::vector<std::string> keys;

        std::transform(
            m_factories.begin(),
            m_factories.end(),
            std::back_inserter(keys),
            [](const std::pair<std::string, FrontEndFactory>& item) { return item.first; });
        return keys;
    }

    FrontEnd::Ptr loadByModel(const std::string& path, FrontEndCapFlags fec)
    {
        FRONT_END_NOT_IMPLEMENTED(loadByModel);
    }

    void registerFrontEnd(const std::string& name, FrontEndFactory creator)
    {
        m_factories.insert({name, creator});
    }

private:
    void registerPlugins()
    {
        auto registerFromDir = [&](const std::string& dir) {
            if (!dir.empty())
            {
                auto plugins = loadPlugins(dir);
                for (auto& plugin : plugins)
                {
                    registerFrontEnd(plugin.m_pluginInfo.m_name, plugin.m_pluginInfo.m_creator);
                    m_loadedLibs.push_back(std::move(plugin.m_libHandle));
                }
            }
        };
        std::string envPath = ngraph::getenv_string("OV_FRONTEND_PATH");
        if (!envPath.empty())
        {
            auto start = 0u;
            auto sepPos = envPath.find(PathSeparator, start);
            while (sepPos != std::string::npos)
            {
                registerFromDir(envPath.substr(start, sepPos - start));
                start = sepPos + 1;
                sepPos = envPath.find(PathSeparator, start);
            }
            registerFromDir(envPath.substr(start, sepPos));
        }
        else
        {
            registerFromDir(".");
        }
    }
};

FrontEndManager::FrontEndManager()
    : m_impl(new Impl())
{
}

FrontEndManager::FrontEndManager(FrontEndManager&&) = default;
FrontEndManager& FrontEndManager::operator=(FrontEndManager&&) = default;

FrontEndManager::~FrontEndManager() = default;

FrontEnd::Ptr FrontEndManager::load_by_framework(const std::string& framework, FrontEndCapFlags fec)
{
    return m_impl->loadByFramework(framework, fec);
}

FrontEnd::Ptr FrontEndManager::load_by_model(const std::string& path, FrontEndCapFlags fec)
{
    return m_impl->loadByModel(path, fec);
}

std::vector<std::string> FrontEndManager::get_available_front_ends() const
{
    return m_impl->availableFrontEnds();
}

void FrontEndManager::register_front_end(const std::string& name, FrontEndFactory creator)
{
    m_impl->registerFrontEnd(name, creator);
}

//----------- FrontEnd ---------------------------

FrontEnd::FrontEnd() = default;

FrontEnd::~FrontEnd() = default;

InputModel::Ptr FrontEnd::load_from_file(const std::string& path) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_file);
}

InputModel::Ptr FrontEnd::load_from_files(const std::vector<std::string>& paths) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_files);
}

InputModel::Ptr FrontEnd::load_from_memory(const void* model) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_memory);
}

InputModel::Ptr
    FrontEnd::load_from_memory_fragments(const std::vector<const void*>& modelParts) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_memory_fragments);
}

InputModel::Ptr FrontEnd::load_from_stream(std::istream& path) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_stream);
}

InputModel::Ptr FrontEnd::load_from_streams(const std::vector<std::istream*>& paths) const
{
    FRONT_END_NOT_IMPLEMENTED(load_from_streams);
}

std::shared_ptr<ngraph::Function> FrontEnd::convert(InputModel::Ptr model) const
{
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<ngraph::Function> FrontEnd::convert(std::shared_ptr<ngraph::Function>) const
{
    FRONT_END_NOT_IMPLEMENTED(convert);
}

std::shared_ptr<ngraph::Function> FrontEnd::convert_partially(InputModel::Ptr model) const
{
    FRONT_END_NOT_IMPLEMENTED(convert_partially);
}

std::shared_ptr<ngraph::Function> FrontEnd::decode(InputModel::Ptr model) const
{
    FRONT_END_NOT_IMPLEMENTED(convertDecodingOnly);
}

void FrontEnd::normalize(std::shared_ptr<ngraph::Function> function) const
{
    FRONT_END_NOT_IMPLEMENTED(normalize);
}

//----------- InputModel ---------------------------
std::vector<Place::Ptr> InputModel::get_inputs() const
{
    FRONT_END_NOT_IMPLEMENTED(get_inputs);
}

std::vector<Place::Ptr> InputModel::get_outputs() const
{
    FRONT_END_NOT_IMPLEMENTED(get_outputs);
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const
{
    FRONT_END_NOT_IMPLEMENTED(get_place_by_tensor_name);
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operationName)
{
    FRONT_END_NOT_IMPLEMENTED(get_place_by_operation_name);
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operationName,
                                                                  int inputPortIndex)
{
    FRONT_END_NOT_IMPLEMENTED(get_place_by_operation_name_and_input_port);
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operationName,
                                                                   int outputPortIndex)
{
    FRONT_END_NOT_IMPLEMENTED(get_place_by_operation_name_and_output_port);
}

void InputModel::set_name_for_tensor(Place::Ptr tensor, const std::string& newName)
{
    FRONT_END_NOT_IMPLEMENTED(set_name_for_tensor);
}

void InputModel::add_name_for_tensor(Place::Ptr tensor, const std::string& newName)
{
    FRONT_END_NOT_IMPLEMENTED(add_name_for_tensor);
}

void InputModel::set_name_for_operation(Place::Ptr operation, const std::string& newName)
{
    FRONT_END_NOT_IMPLEMENTED(set_name_for_operation);
}

void InputModel::free_name_for_tensor(const std::string& name)
{
    FRONT_END_NOT_IMPLEMENTED(free_name_for_tensor);
}

void InputModel::free_name_for_operation(const std::string& name)
{
    FRONT_END_NOT_IMPLEMENTED(free_name_for_operation);
}

void InputModel::set_name_for_dimension(Place::Ptr place,
                                        size_t shapeDimIndex,
                                        const std::string& dimName)
{
    FRONT_END_NOT_IMPLEMENTED(set_name_for_dimension);
}

void InputModel::cut_and_add_new_input(Place::Ptr place, const std::string& newNameOptional)
{
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_input);
}

void InputModel::cut_and_add_new_output(Place::Ptr place, const std::string& newNameOptional)
{
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_output);
}

Place::Ptr InputModel::add_output(Place::Ptr place)
{
    FRONT_END_NOT_IMPLEMENTED(add_output);
}

void InputModel::remove_output(Place::Ptr place)
{
    FRONT_END_NOT_IMPLEMENTED(remove_output);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs)
{
    FRONT_END_NOT_IMPLEMENTED(override_all_outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs)
{
    FRONT_END_NOT_IMPLEMENTED(override_all_inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                  const std::vector<Place::Ptr>& outputs)
{
    FRONT_END_NOT_IMPLEMENTED(extract_subgraph);
}

// Setting tensor properties
void InputModel::set_partial_shape(Place::Ptr place, const ngraph::PartialShape&)
{
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

ngraph::PartialShape InputModel::get_partial_shape(Place::Ptr place) const
{
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

void InputModel::set_element_type(Place::Ptr place, const ngraph::element::Type&)
{
    FRONT_END_NOT_IMPLEMENTED(set_element_type);
}

void InputModel::set_tensor_value(Place::Ptr place, const void* value)
{
    FRONT_END_NOT_IMPLEMENTED(set_tensor_value);
}

void InputModel::set_tensor_partial_value(Place::Ptr place,
                                          const void* minValue,
                                          const void* maxValue)
{
    FRONT_END_NOT_IMPLEMENTED(set_tensor_partial_value);
}

//----------- Place ---------------------------
std::vector<std::string> Place::get_names() const
{
    FRONT_END_NOT_IMPLEMENTED(get_names);
}

std::vector<Place::Ptr> Place::get_consuming_operations() const
{
    FRONT_END_NOT_IMPLEMENTED(get_consuming_operations);
}

std::vector<Place::Ptr> Place::get_consuming_operations(int outputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_consuming_operations);
}

Place::Ptr Place::get_target_tensor() const
{
    FRONT_END_NOT_IMPLEMENTED(get_target_tensor);
}

Place::Ptr Place::get_target_tensor(int outputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_target_tensor);
}

Place::Ptr Place::get_producing_operation() const
{
    FRONT_END_NOT_IMPLEMENTED(get_producing_operation);
}

Place::Ptr Place::get_producing_operation(int inputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_producing_operation);
}

Place::Ptr Place::get_producing_port() const
{
    FRONT_END_NOT_IMPLEMENTED(get_producing_port);
}

Place::Ptr Place::get_input_port() const
{
    FRONT_END_NOT_IMPLEMENTED(get_input_port);
}

Place::Ptr Place::get_input_port(int inputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_input_port);
}

Place::Ptr Place::get_input_port(const std::string& inputName) const
{
    FRONT_END_NOT_IMPLEMENTED(get_input_port);
}

Place::Ptr Place::get_input_port(const std::string& inputName, int inputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_input_port);
}

Place::Ptr Place::get_output_port() const
{
    FRONT_END_NOT_IMPLEMENTED(get_output_port);
}

Place::Ptr Place::get_output_port(int outputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_output_port);
}

Place::Ptr Place::get_output_port(const std::string& outputName) const
{
    FRONT_END_NOT_IMPLEMENTED(get_output_port);
}

Place::Ptr Place::get_output_port(const std::string& outputName, int outputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_output_port);
}

std::vector<Place::Ptr> Place::get_consuming_ports() const
{
    FRONT_END_NOT_IMPLEMENTED(get_consuming_ports);
}

bool Place::is_input() const
{
    FRONT_END_NOT_IMPLEMENTED(is_input);
}

bool Place::is_output() const
{
    FRONT_END_NOT_IMPLEMENTED(is_output);
}

bool Place::is_equal(Ptr another) const
{
    FRONT_END_NOT_IMPLEMENTED(is_equal);
}

bool Place::is_equal_data(Ptr another) const
{
    FRONT_END_NOT_IMPLEMENTED(is_equal_data);
}

Place::Ptr Place::get_source_tensor() const
{
    FRONT_END_NOT_IMPLEMENTED(get_source_tensor);
}

Place::Ptr Place::get_source_tensor(int inputPortIndex) const
{
    FRONT_END_NOT_IMPLEMENTED(get_source_tensor);
}
