# Subgraphs Dumper tool

The tool is intended to analyse some arbitrary scope of the models in a formats supported by Inference Engine Readers
to extract and serialize unique patterns from all of the input models. Uniqueness and matching criteria are defined by implementation of
`Matcher` interface class declared in ./include/matchers/base_matcher.hpp and should be registered in 
`MatchersManager`declared in ./include/matchers/matchers_manager.hpp by adding to `m_registry` map.

## Building

To build the tool need to run following commands   
```
cmake -DENABLE_FUNCTIONAL_TESTS=ON
make -j subgraphsDumper
```
Outcome of a build is a `subgrpahsDumper` binary located in building artifacts folder.

## Running
The tool takes two command line parameters:    
* `--input_folders` - Required. Comma separated paths to the input folders with IRs
* `--local_cache` - Optional. Comma separated paths to the local cache folders with IRs.
* `--output_folder` - Required. Path to the output folders where to serialize IRs
* `--path_regex` - Optional. regular expression to be applied in input folders recursive discovery
* `--constants_size_threshold` - Optional. Maximum size of constant in megabytes to be serialized.
                                 If constant size exceeds specified number it will be replaced
                                 with parameter and meta information about original data range will be saved
* `--eliminate_dynamism` - Optional. If specified dynamic shapes will be eliminated from model 
                           and replaced by propagated upper bound values (if possible)       

E.g.    
```subgraphsDumper --input_folders /folder/with/models,/another/folder/with/models --output_folder /output/folder```

## Extraction algorithm
*NOTE: current implementation presumes only single operation matching rules, to be extended to handle wider patterns.*

1. Recursively searching for all of rhe models in provided input folders
2. Reading first model and iterating over the nodes in the ngraph function model's representation 
   (Parameters, Results and Constants are ignored)
3. Comparing current operation with all of the operations in internal cache by running all of the matchers registered in 
`MatchersManager`. Operation is cloned and added to the cache if it is not matched by any of matchers, otherwise will be ignored.
   Cloning rules may vary depending on operation type and defined in `./src/op_cloner.cpp`
4. Proceeding with a next model without resetting internal operations cache.
5. Serializing all cached subgraphs to the output folder in IR format.