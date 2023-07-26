# Subgraphs Dumper tool

The tool is intended to analyse some arbitrary scope of the models in a formats supported by OpenVINO frontends
to extract and serialize unique operations and patterns from the input models. Uniqueness and matching criteria are defined by implementation of
* `Matcher` interface class which defines the rules to dump operatons to the cache
* `Extractor` interface class which defines the rules to extract subgraphs from the models.

> NOTE:
> Please check the following architecture diagram to get detailed information.

## How to build

To build the tool need to run following commands   
```
cmake -DENABLE_FUNCTIONAL_TESTS=ON .
make subgraphsDumper
```
Outcome of a build is a `subgrpahsDumper` binary located in building artifacts folder.

## Running
The tool takes two command line parameters:    
* `--input_folders` - Required. Comma separated paths to the input folders with IRs. The separator is `,`
* `--output_folder` - Required. Path to the output folders where to serialize IRs
* `--local_cache` - Optional. Comma separated paths to the local cache folders with IRs. The separator is `,`
* `--path_regex` - Optional. regular expression to be applied in input folders recursive discovery
* `--extract_body` - Optional. Allow to extract operation bodies to operation cache.
* `--cache_type` - Optional. Allow to extract Operations, Subgraphs or both types. The default value is '' - OP and GRAPH.

E.g.    
```subgraphsDumper --input_folders /dir_0/to/models,/dir_1/to/models --output_folder /path/to/dir```

## Extraction algorithm
1. Recursively searching for all of the models in provided input folders using enabled OV frontends.
2. Reading models and iterating over the nodes in the OV model to extract operations and defined patterns
   (Parameters, Results and Constants are ignored)
3. Clone the entity by replacement of input nodes by Parameters/Constant.
4. Comparing cloned entity with entities extracted before in internal cache by running all of the matchers registered in Matcher/Extractor managers rules.
5. Serializing all cached subgraphs to the output folder in OV IR format.

## Validation
The tool is validated by combination of Unit and Functional tests using based OV approach. To run tests using the following commands:
```
cmake -DENABLE_FUNCTIONAL_TESTS=ON .
make subgraphsDumperTests
./subgraphsDumperTests --gtest_filter=*
```

## Architecture diagram
![SubgraphsDumper Architecture Diagram](./img/arch.png)