# NPU Properties
What is a property from NPU Plugin's POV and Step-by-step guide on how to add one  
Practical manual for NPU plugin properties

## Table of Contents

- [Glossary](#glossary)
- [Structure of a property (Class hierarchy)](#structure-of-a-property-class-hierarchy)
  - [Properties With Option vs Properties Without Option](#properties-with-option-vs-properties-without-option)
  - [OptionBase<T>](#optionbaset)
  - [OptionDesc](#optiondesc)
  - [Config](#config)
  - [FilteredConfig](#filteredconfig)
  - [Properties](#properties)
- [How to add a new public (option backed) property](#how-to-add-a-new-public-option-backed-property)
  - [Step 1. Define the new property](#step-1-define-the-new-property)
  - [Step 2. Define the internal option descriptor](#step-2-define-the-internal-option-descriptor)
  - [Step 3. Register the new option](#step-3-register-the-new-option)
  - [Step 4. Link the new property to the new option](#step-4-link-the-new-property-to-the-new-option)
    - [For plugin](#for-plugin)
    - [For compiled-model (if required)](#for-compiled-model-if-required)
  - [Step 5. Python bindings](#step-5-python-bindings)
  - [Step 6. Update documentation](#step-6-update-documentation)
- [How to add a new public property without option (callback-backed/read-only)](#how-to-add-a-new-public-property-without-option-callback-backedread-only)
  - [Step 1. Define the new property](#step-1-define-the-new-property-1)
  - [Step 2. Define and register a callback function for the new property](#step-2-define-and-register-a-callback-function-for-the-new-property)
  - [Step 3. Python bindings](#step-3-python-bindings)
  - [Step 4. Update documentation](#step-4-update-documentation)
- [Compiled-model properties](#compiled-model-properties)
- [Removing a public property](#removing-a-public-property)

## Glossary

| Name     | Description | Example |
|:--------:|:---------   |:--------: |
| Property | a plugin interface which can be set or read | `ov::log::level` | 
| Option   | a configuration entry in our internal configuration.</br>Consists of an OptionBase template descriptor + a template OptionValue value. | `LOG_LEVEL` |
| Config   | Our internal database of configuration keys and their values | |
| Option-backed property | A property mapped to an Option entry in Config through `register_property`. | `ov::hint::performance_mode` |
| Property without option (callback-backed/read-only) | A property that is not mapped to Config and is implemented by callback logic (usually runtime/backend queries). | `ov::device::pci_info` |
| Compiler | Npu compiler as viewed from the plugin's perspective.</br>Can be Compiler-In-Driver or Compiler-In-Plugin | `CID` |
| "Anonymous" property</br>OR</br>compiler-private property | A setting from application level which the plugin has no knowledge of</br>(it is not registered, plugin is not aware of its datatype)</br> but which the compiler reports as supported via is_supported() API. | N/A |

<br>

## Structure of a property (Class hierarchy)

### Properties With Option vs Properties Without Option
As shown in the class hierarchy, a property is the public interface to internal information. A property can be
implemented either through a `Config` option or through callback logic.

Option-backed properties are stored in `Config` and can be updated through their registered setter. Properties without
an option are computed on demand, typically from the backend, driver, or operating system. Examples include driver
version, compiler version, device architecture, PCI information, GOPS, UUID, and LUID.

To summarize:
A property can be implemented in **one** of the 2 ways:
- Option-backed property: has an entry in internal Config and is managed through Option descriptors/helpers.
- Property without option: implemented through callback logic (typically read-only runtime/backend information).

### OptionBase\<T\> 
Implements the option descriptor. This class contains all the details of a config option: name, datatype, default value, parser, public/private, mutability, compiler version (for legacy support), etc. This serves as the key in our configuration map. 

Class definition in npu_plugin/al/include/config/config.hpp:  
```cpp
struct OptionBase { 
    using ValueType = T; 

    // `ActualOpt` must implement the following method: 
    // static std::string_view key() 

    static constexpr std::string_view getTypeName() { 
        if constexpr (TypePrinter<T>::hasName()) { 
            return TypePrinter<T>::name(); 
        } 
        static_assert(TypePrinter<T>::hasName(), 
                      "Options type is not a standard type, please add `getTypeName()` to your option"); 
    } 
    // Overload this to provide environment variable support. 
    static std::string_view envVar() { 
        return ""; 
    } 

    // Overload this to provide deprecated keys names. 
    static std::vector<std::string_view> deprecatedKeys() { 
        return {}; 
    } 

    // Overload this to provide default value if it wasn't specified by user. 
    // If it is std::nullopt - exception will be thrown in case of missing option access. 
    static std::optional<T> defaultValue() { 
        return std::nullopt; 
    } 

    // Overload this to provide more specific parser. 
    static ValueType parse(std::string_view val) { 
        return OptionParser<ValueType>::parse(val); 
    } 

    // Overload this to provide more specific validation 
    static void validateValue(const ValueType&) {} 

    // Overload this to provide more specific implementation. 
    static OptionMode mode() { 
        return OptionMode::Both; 
    } 

    // Overload this for private options. 
    static bool isPublic() { 
        return false; 
    } 

    // Overload this for read-only properties 
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 

    static std::string toString(const ValueType& val) { 
        return OptionPrinter<ValueType>::toString(val); 
    } 
}; 
```

### OptionDesc
is storage for the registered options. This is the base map which stores the available OptionBase descriptors.
This layer implements the option database manipulation functions: add/has/reset.
The plugin creates it once, registers the plugin options, and lets the backend add its compiler-specific options before
passing the finalized descriptor to the plugin property manager.

### Config
is the highlevel configuration "database" which implements the mapping between OptionBase and templatized OptionValue.
Maps and stores the user-defined values for each entry in OptionsDesc layer.
Implements the top level configuration manipulation functions:
get/update/updateAny/has/getString/toString/fromString and handles typecasts, type verification, parsing and conversions.
```` Note: This layer is initialized once in the plugin from the finalized OptionsDesc and passed to the plugin property manager. ````

### FilteredConfig
is a derivative class of Config, used only by NPU Plugin, which implements additional filtering layers atop of the base config,
such as enabling/disabling keys based on their availability/support on the current system configuration.
```` Note: This layer dynamically changes based on system configuration and compiler_type. ````

The initialization order is:
1. `Plugin` creates the complete `OptionsDesc` and registers all plugin options.
2. The backend adds its options through `backend->registerOptions(*options)`.
3. `Plugin` constructs the `FilteredConfig` from the finalized descriptor, parses environment variables, and passes it to `PluginPropertyManager`.
4. `PluginPropertyManager` resolves the effective compiler type and registers property descriptors.

### Properties
is the top level class and serves as the NPU Plugin's interface to OpenVino and the application layer.
It's main purpose is to implement get_property and set_property interfaces and the callback functions of each property.
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

<br><br>

# How to add a new public (option backed) property

The following steps how to add a new simple property which maps to a compiler configuration option.  
_*simple in this context means that it has no special callback function required for it, just set/get_  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RW> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Write
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 }  

## Step 2. Define the internal option descriptor
Second step is to define the optionDesc class for this property in  
```bash
src/plugins/intel_npu/src/al/include/intel_npu/config/options.hpp
```  
Example:  
```cpp
// 
// EXAMPLE_PROPERTY 
//  
struct EXAMPLE_PROPERTY final : OptionBase<EXAMPLE_PROPERTY, ov::intel_npu::ExampleType> {  

    static std::string_view key() { 
        return ov::intel_npu::example_property.name();
        } 

    static constexpr std::string_view getTypeName() { 
        return "ov::intel_npu::ExampleType"; 
    } 

    static ov::intel_npu::ExampleType defaultValue() { 
        return ov::intel_npu::ExampleType::VAL3; 
    } 

    static bool isPublic() { 
        return true; 
    } 

    static OptionMode mode() { 
        return OptionMode::Both; 
    } 
     
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 
     
    static std::string_view envVar() { 
        return "IE_NPU_EXAMPLE_PROPERTY"; 
    } 

    static ov::intel_npu::ExampleType parse(std::string_view val) { 
        if (val == "VAL1") { 
            return ov::intel_npu::ExampleType::VAL1; 
        } else if (val == "VAL2") { 
            return ov::intel_npu::ExampleType::VAL2; 
        } else if (val == "VAL3") { 
            return ov::intel_npu::ExampleType::VAL3; 
        } 

        OPENVINO_THROW("Value '", val, "'is not a valid EXAMPLE_PROPERTY option"); 
    } 

    static std::string toString(const ov::intel_npu::ExampleType& val) { 
        std::stringstream strStream; 

        strStream << val; 

        return strStream.str(); 
    } 
}; 
```
Notes:  
- key(): needs to return the string name of the property (the NPU_EXAMPLE_PROPERTY defined in the property at step 1)  
- getTypeName: returns the type name as a human-readable string  
- defaultValue: returns the option's default value (if there was no user-defined value set, config.get or get_property(EXAMPLE_PROPERTY) will call this function)
- isPublic: defines whether the option is a **public or a private** one  
- mode: defines the OptionMode of this option. Can be:  
    - CompileTime (for options used ONLY by the compiler)  
    - Runtime (for options only used by plugin and runtime)  
    - Both (for options used by both).  
    **Only options of CompileTime and Both will be sent to compiler at model compilation.**  
- mutability: whether the option is **Read-Write** or **Read-Only**  
- envVar: environment variable (if needed) for this property. The config manager will check if the options have envVar defined. For each option which has envVar, it will look in environment variables and update the option value from there at init.  
- parse: string to custom datatype parser. If the property will be set with a string value, this parser will convert it into the internal datatype.  
- toString: for converting the option value from the custom datatype to string 

**(!!)** None of the member functions are mandatory to be defined.  
If any is missing, the default function will be used for its call, as defined in the OptionsBase class  
(see class **OptionBase** in `src/plugins/intel_npu/src/al/include/intel_npu/config/config.hpp` or Class Hierarchy section above)

## Step 3. Register the new option
Third step is to register the new option in the plugin:
**src/plugins/intel_npu/src/plugin/src/plugin.cpp > function register_options(...)**
```cpp
    REGISTER_OPTION(EXAMPLE_PROPERTY);
``` 
Notes:  
The plugin registers the option in `OptionsDesc` before the backend adds its compiler-specific options. The plugin then
constructs `FilteredConfig` from the completed descriptor, parses environment variables, and passes the configuration to
`PluginPropertyManager`.

## Step 4. Link the new property to the new option
Fourth step is to create and register the Property (which is basically the interface to this configuration option) for both Plugin and CompiledModel (if needed) 
### For plugin
src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
```cpp
register_property(
    EXAMPLE_PROPERTY::key(),
    true,
    ov::PropertyMutability::RW,
    [this](const ov::AnyMap&) {
        return _config->hasOpt(EXAMPLE_PROPERTY::key());
    },
    [this](const ov::AnyMap&) {
        return _config->get<EXAMPLE_PROPERTY>();
    },
    [this](const ov::Any& value) {
        _config->updateAny({{EXAMPLE_PROPERTY::key(), value}});
    });
```
**Explanation:**
`register_property` stores the property name, visibility, mutability, support predicate, getter, and setter in one
descriptor. The support predicate determines whether the property is exposed. The getter reads the typed option from
`FilteredConfig`, and the setter validates and stores the supplied value through `updateAny`.
### For compiled-model (if required)
src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()
```cpp
const auto hasPropertyValue = [this](const std::string& propertyName) {
    return _config->has(propertyName);
};

register_property(
    EXAMPLE_PROPERTY::key(),
    true,
    ov::PropertyMutability::RO,
    [this, hasPropertyValue](const ov::AnyMap&) {
        return _config->hasOpt(EXAMPLE_PROPERTY::key()) && hasPropertyValue(EXAMPLE_PROPERTY::key());
    },
    [this](const ov::AnyMap&) {
        return _config->get<EXAMPLE_PROPERTY>();
    },
    [](const ov::Any&) {
        OPENVINO_THROW("READ-ONLY configuration key");
    });
```
**Explanation:**
Compiled-model properties are normally read-only. The support predicate above exposes the property only when the
option is registered and has an explicitly stored value in the compiled model's configuration. The setter is still
required by the common descriptor, but rejects writes.

## Step 5. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RW(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 6. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
src/plugins/intel_npu/README.md 
```

<br><br>

# How to add a new public property without option (callback-backed/read-only)
As described in the first paragraph, these properties do not have an entry in our internal Config and are returned through callback logic at get_property calls.  
This means we do not need to define an Option nor register an Option for them.  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RO> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Only
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 } 

## Step 2. Define and register a callback function for the new property
You need to register the new property and define a callback function in the owner-specific property manager.
For plugin: src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
For compiled-model: src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()

For properties without option, prefer a support-gated registration so the getter is only used when backend/runtime requirements are available:
```cpp
    const auto has_backend = [this](const ov::AnyMap&) {
        return _backend != nullptr;
    };

    register_property(
        ov::intel_npu::example_property.name(),
        true,
        ov::PropertyMutability::RO,
        has_backend,
        [this](const ov::AnyMap&) {
            return utils::getDriverVersion(_backend);
        },
        [](const ov::Any&) {
            OPENVINO_THROW("READ-ONLY configuration key");
        });
```
**Explanation**
`register_property` registers a property with the name **ov::intel_npu::example_property (NPU_EXAMPLE_PROPERTY)**, which is public and included in supported_properties when the support predicate returns true.  
and calls `utils::getDriverVersion(_backend)` each time get_property is queried. The getter receives query-time
arguments as `ov::AnyMap`; use them when the property supports extra inputs, or ignore them when it does not.
Note: the first argument is the property name string (`property.name()`), not the property object itself.

## Step 3. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RO(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 4. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst
src/plugins/intel_npu/README.md
```

<br><br>

# Compiled-model properties
Compiled-model properties describe settings that were used to create an already compiled model. By convention:
- options with `OptionMode::CompileTime` or `OptionMode::Both` may be exposed;
- most properties are exposed only when their value was explicitly stored before compilation;
- compiled-model properties are normally `RO`, because changing them cannot recompile the model.

The compiled-model manager uses the same six-argument `register_property` function as the plugin manager:
```cpp
register_property(
    propertyName,
    isPublic,
    mutability,
    isSupported,
    getter,
    setter);
```

For properties that require an explicitly stored value, use a support predicate such as:
```cpp
const auto hasPropertyValue = [this](const std::string& propertyName) {
    return _config->has(propertyName);
};

register_property(
    EXAMPLE_PROPERTY::key(),
    true,
    ov::PropertyMutability::RO,
    [this, hasPropertyValue](const ov::AnyMap&) {
        return _config->hasOpt(EXAMPLE_PROPERTY::key()) && hasPropertyValue(EXAMPLE_PROPERTY::key());
    },
    [this](const ov::AnyMap&) {
        return _config->get<EXAMPLE_PROPERTY>();
    },
    [](const ov::Any&) {
        OPENVINO_THROW("READ-ONLY configuration key");
    });
```

Some compiled-model properties are intentionally available whenever their option is registered, using
`_config->hasOpt(propertyName)` without requiring `_config->has(propertyName)`. This applies to properties whose
default value is part of the compiled-model contract.

<br><br>

# Removing a public property
Removing a public property can be done by removing everything added in section "How to add a new public property" step-by-step.
