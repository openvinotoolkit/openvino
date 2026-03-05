# NPU Properties
What is a property from NPU Plugin's POV and Step-by-step guide on how to add one  
Practical manual for NPU plugin properties

## Table of Contents

- [Glossary](#glossary)
- [Structure of a property (Class hierarchy)](#structure-of-a-property-class-hierarchy)
  - [Property vs Option vs Metric](#property-vs-option-vs-metric)
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
- [How to add a new public (metric backed) property](#how-to-add-a-new-public-metric-backed-property)
  - [Step 1. Define the new property](#step-1-define-the-new-property-1)
  - [Step 2. Define and register a callback function for the new (metric) Property](#step-2-define-and-register-a-callback-function-for-the-new-metric-property)
  - [Step 3. Python bindings](#step-3-python-bindings)
  - [Step 4. Update documentation](#step-4-update-documentation)
- [Compiled-model properties](#compiled-model-properties)
- [Special cases](#special-cases)
  - [SC.1 Adding a new property which requires custom functions](#sc1-adding-a-new-property-which-requires-custom-functions)
  - [SC.2 Adding a new (metric-backed) property which requires customization](#sc2-adding-a-new-metric-backed-property-which-requires-customization)
  - [SC.3 Filtering out options at registration phase](#sc3-filtering-out-options-at-registration-phase)
- [Removing a public property](#removing-a-public-property)

## Glossary

| Name     | Description | Example |
|:--------:|:---------   |:--------: |
| Property | a plugin interface which can be set or read. Can map to options or metrics` | `ov::log::level` | 
| Option   | a configuration entry in our internal configuration.</br>Consists of an OptionBase template descriptor + a template OptionValue value. | `LOG_LEVEL` |
| Config   | Our internal database of configuration keys and their values | `_globalConfig` |
| Metric   | A property which does not map to any configuration key and configuration entry in our internal config.</br>Usually a static value directly read from driver or OS. | `ov::device::pci_info` |
| Compiler | Npu compiler as viewed from the plugin’s perspective.</br>Can be Compiler-In-Driver or Compiler-In-Plugin | `CID` | 
| “Anonymous” property</br>OR</br>compiler-private property | A setting from application level which the plugin has no knowledge of</br>(it is not registered, plugin is not aware of its datatype)</br> but which the compiler reports as supported via is_supported() API. | N/A |

<br>

## Structure of a property (Class hierarchy)

![Properties Class Hierarchy](./img/properties_class_hierarchy.png)

<br>

### Property vs Option vs Metric
As it can be observed in the above class hierarchy diagram, a Property is a public interface to an internal information, the top layer of abstraction.  
A property can connect internally to an Option or to a Metric.  
The main difference between Option and Metric is that while Options are entries in our internal database (Config) which can be modified at any time, Metrics are static read-only pieces of information  
which do not exist in the internal database. Metrics can not be set or changed externally, their values are not stored in the plugin and they most often just a direct system or driver call.  
Example of Metrics in NPU Plugin are: driver version, compiler version, device architecture, pci information, gops, uuid, luid, etc.
To summarize:  
A property can map internally to **one** of the 2 types:  
- either a Option, if it has an entry in our internal Config, it is a mutable setting used either by compiler or by inference
- either a Metric, if it is a static immutable piece of information that only needs to be returned on request (system/platform/hardware etc information) 

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

    // Overload this for read-only options (metrics) 
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 

    /// Overload this for options conditioned by compiler version 
    static uint32_t compilerSupportVersion() { 
        return ONEAPI_MAKE_VERSION(0, 0); 
    } 

    static std::string toString(const ValueType& val) { 
        return OptionPrinter<ValueType>::toString(val); 
    } 
}; 
```

### OptionDesc  
is storage for the registered options This is the base map which stores the available optionBase descriptors.  
This layer implements the option database manipulation functions: add/remove/has/reset  
```` Note: This layer is static, intialized once in plugin constructor with all the options the Plugin has knowledge of. ````

### Config 
is the highlevel configuration “database” which implements the mapping between OptionBase and templatized OptionValue.  
Maps and stores the user-defined values for each entry in OptionsDesc layer.  
Implements the top level configuration manipulation functions:  
get/update/has/getString/toString/fromString and handles typecasts, typeverification, parsing and conversions.  
```` Note: This layer is static, initialized once in plugin constructor with the provided (also static) OptionDesc. ````

### FilteredConfig 
is a derivative class of Config, used only by NPU Plugin, which implements additional filtering layers atop of the base config,  
such as enabling/disabling keys based on their availability/support on the current system configuration.  
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

### Properties
is the top level class and serves as the NPU Plugin’s interface to OpenVino and the application layer.  
It’s main purpose is to implement get_property and set_property interfaces and the callback functions of each property.  
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

<br><br>

# How to add a new public (option backed) property

The following steps how to add a new simple property which maps to a compiler configuration option.  
_*simple in this context means that it has no special callback function required for it, just set/get_  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
openvino/src/inference/include/openvino/runtime/intel_npu/properties.hpp
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
openvino/src/plugins/intel_npu/al/config/options.hpp
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

    static uint32_t compilerSupportVersion() { 
        return ONEAPI_MAKE_VERSION(5, 5); 
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
- defaultValue: returns the option's default value (if there was no user-defined value set, config.get or get_property(EXAMPLE_PROPERTY) will call this function  
- compilerSupportVersion: the compiler version from which this key is supported by compiler  
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
(see class **OptionBase** in *npu_plugin/al/include/config/config.hpp* or Class Hierarchy section above)

## Step 3. Register the new option
Third step is to register the new option in the plugin:  
**plugin.cpp > function init_options()**:
```cpp
    REGISTER_OPTION(EXAMPLE_PROPERTY);
``` 
Notes:  
at this point, the npu plugin will take care of registering and managing the option in the internal configuration.  
It ensures that it is enabled or disabled based in the current system/environment/application configuration. 

## Step 4. Link the new property to the new option
Fourth step is to create and register the Property (which is basicly the interface to this configuration option) for both Plugin and CompiledModel (if needed) 
### For plugin
npu_plugin/plugin/src/properties.cpp > function registerPluginProperties()
```cpp
TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::example_property, EXAMPLE_PROPERTY);
```
**Explanation:**
this macro will register (if all conditions are met) a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
### For compiled-model (if required)
npu_plugin/plugin/src/properties.cpp > function registerCompiledModelProperties()
```cpp
TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::example_property, EXAMPLE_PROPERTY);
```
**Explanation:**
this macro will register (if all conditions are met) a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()  
**(!!) ONLY** if it has been previously explicitly set at compile time

## Step 5. Python bindings
In order for the property to be property exposed in python API, add python wrapper for the new property in pyOpenvino  
openvino/src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // sumbodule npu  
```cpp
    wrap_property_RW(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 6. Update documentation
Document the new property in the appropaite sections (+ additional information, if required) in:  
```bash
openvino/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
openvino/src/plugins/intel_npu/README.md 
```

<br><br>

# How to add a new public (metric backed) property
As described in the first paragraph, Metrics do not have an entry in our internal Config, they are static immutable data which just needs to be returned at get_property calls.  
This means we do not need to define and Option nor register an Option for it.  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
openvino/src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RO> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Only
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 } 

## Step 2. Define and register a callback function for the new (metric) Property
You need to register the new property and define a callback function for it, in properties.cpp.  
For plugin: npu_plugin/plugin/src/properties.cpp > function registerPluginProperties()  
For compiled-model: npu_plugin/plugin/src/properties.cpp > function registerCompiledModelProperties()  
```cpp
    REGISTER_SIMPLE_METRIC(ov::intel_npu::example_property, true, _metrics->GetDriverVersion());
```
**Explanation**
this macro will register a property with the name **ov::intel_npu::example_property (NPU_EXAMPLE_PROPERTY)**, which will be public and included in supported_properties (second parameter)  
which will execute _metrics->GetDriverVersion() function as it's get_property callback

## Step 3. Python bindings
In order for the property to be property exposed in python API, add python wrapper for the new property in pyOpenvino  
openvino/src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // sumbodule npu  
```cpp
    wrap_property_RO(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 4. Update documentation
Document the new property in the appropaite sections (+ additional information, if required) in:  
```bash
openvino/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
openvino/src/plugins/intel_npu/README.md 
```

<br><br>

# Compiled-model properties
By internal convention, what needs to be included in compiled-model properties gets decided based on the following statements:  
- every option which has an effect on model compilation (meaning options with mode=OptionMode::CompileTime or OptionMode::Both) need to be included
- options (with some specific exceptions) should be publicly advertised in compiled-model's supported_properties **ONLY** if they have been explicitly set prior to model compilation.  
- compiled-model properties (with a few specific exceptions) are all READ-ONLY, for the reason that the model has already been compiled.
This is to ensure that we only expose settings we are sure were taken into account by compiler.  

To facilitate registering properties in compiled-model **ONLY** if they have been explicitly set prior to model compilation, properties.cpp offers a helper macro:
#### TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(OPT_NAME, OPT_TYPE)
This macro will register the provided property (OPT_NAME) to the provided option (OPT_TYPE) only if a value for it exists in the config. (default values do not live in config, but directly read from OptionsDesc)  
Example:  
```cpp
    TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::compilation_mode, COMPILATION_MODE);
```

<br><br>

# Special cases
## SC.1 Adding a new property which requires custom functions
If the new property requires a custom callback function, only Step 4. changes.  
Instead of using TRY_REGISTER_SIMPLE_PROPERTY macro, you can choose from the following helper macros:  

#### TRY_REGISTER_VARPUB_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY)  
This can be used when callback is standard, but visibility (public/private) is custom.  
Instead of using automaticly the value from optionsBase, one can define a custom function to determine  
whether the property will be public (included in supported_properties) or private and provide as PROP_VISIBILITY parameter. 
#### TRY_REGISTER_CUSTOMFUNC_PROPERTY(OPT_NAME, OPT_TYPE, PROP_RETFUNC) 
This macro can be used whenever a custom callback function is required for this property,  
provided as a lambda function as PROP_RETFUNC parameter.  
(Standard callback function just returns the value of the config)  
#### TRY_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)
This macro bypasses all automatic descriptor fetching, availability checks, and compatibility verifications  
and gives you the possibility to register a completely custom property with custom visibility and custom callback function. 

## SC.2 Adding a new (metric-backed) property which requires customization
As oppposed to option-backed properties, for metric-backed ones we only have 1 extra helper macro,  
apart from REGISTER_SIMPLE_METRIC (which lets you define name, visibility and return value as a single function call):  

#### REGISTER_CUSTOM_METRIC(PROP_NAME, PROP_VISIBILITY, PROP_RETFUNC)
This macro offers the possibility to define a full custom callback function, as a lambda function propided as parameter PROP_RETFUNC,  
instead of just a single value function call as REGISTER_SIMPLE_METRIC.  
Example:  
```cpp
    REGISTER_CUSTOM_METRIC(ov::device::full_name,
                           !_metrics->GetAvailableDevicesNames().empty(),
                           [&](const Config& config) {
                               const auto specifiedDeviceName = get_specified_device_name(config);
                               return _metrics->GetFullDeviceName(specifiedDeviceName);
                           });
```

## SC.3 Filtering out options at registration phase
There is a possibility to get options filtered out right at the options registration phase too, if desired by a specific cornercase.  
If the option is not registered, it will be virtually inexistent in the whole stack at runtime, so there will be no added overhead by all the availability checks and verifications the properties manager does.  
To achieve this, we can condition the registration of the option in:  
plugin.cpp > (function) void Plugin::init_options()  
Example:  
```cpp
    if (_backend) {
        if (_backend->isCommandQueueExtSupported()) {
            REGISTER_OPTION(TURBO);
            REGISTER_OPTION(WORKLOAD_TYPE);
        }
    }
```
The above code will register TURBO and WORKLOAD_TYPE options only if command queue extension is supported by the backend (and of course, first if a backend exists)  
Having skipped the option registration (in case command queue extension is not supported), the option will not exist in the stack, will not be checked for compiler support, will not be enabled/disabled dinamically and **can not be enabled dynamically** anymore at runtime.

<br><br>

# Removing a public property
Removing a public property can be done by removing everything added in section “How do add a new public property” step-by-step.  