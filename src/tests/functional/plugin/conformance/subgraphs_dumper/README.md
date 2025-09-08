# Subgraphs Dumper Tool

The tool is designed to analyse any arbitrary scope of the models in a formats supported by OpenVINO frontends
to extract and serialize unique operations and patterns from the input models. The criteria for
uniqueness and matching are defined by implementation of twon interface classes:
* `Matcher` defines the rules for dumping operatons to the cache.
* `Extractor` defines the rules for extracting subgraphs from the models.

### Single operation matchers:
* `single op` matches all input & output element types and shape ranks, operation type and attributes.
* `convolution` is based on `single op` matcher but compares a weights of convolution.

### Subgraphs extractor types:
* `fused names` extractor dumps any subgraphs were changed by a transformation pipeline.
* `repeat pattern` extractor finds repeatitions in original model.
* `read value & assign` extracts stateful graphs.

> NOTE:
> Please check the following architecture [diagram](../../../../../../docs/articles_en/assets/images/subgraphs_dumper_arch_diaram.png) to get detailed information.

## Build

To build the tool, run the following commands:
```
cmake -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_TESTS=ON .
make --jobs=$(nproc --all) ov_subgraphs_dumper
```
The outcome of a build is a `ov_subgraphs_dumper` binary located in the building artifacts folder.

## Run
The tool takes only one required command-line parameter:
* `--input_folders` - Required. Comma separated paths to the input folders with models in Intermediate Representation format (IRs). The separator is `,`.
* `--output_folder` - Optinal. Path to the output folders where the IRs will be serialized. Default value is "output".
* `--local_cache` - Optional. Comma-separated paths to the local cache folders with IRs. The separator is `,`.
* `--path_regex` - Optional. Regular expression to be applied in input folders for recursive discovery.
* `--extract_body` - Optional. Allows extracting operation bodies to the operation cache.
* `--cache_type` - Optional. Allows extracting Operations, Subgraphs, or both types. The default value is `OP` and `GRAPH`.

Example running command:
```ov_subgraphs_dumper --input_folders /dir_0/to/models,/dir_1/to/models --output_folder /path/to/dir```

## Extraction Algorithm
1. Recursively search for all models in the provided input folders using enabled OpenVINO frontends.
2. Read models and iterate over the nodes in the OpenVINO model to extract operations and defined patterns. Parameters, Results, and Constants are ignored.
3. Clone the entity by replacing input nodes with Parameters/Constant.
4. Compare the cloned entity with previously extracted entities stored in the internal cache by running all matchers registered in the Matcher/Extractor manager's rules.
5. Serialize all cached subgraphs to the output folder in OV IR format.

## Validation
The tool is validated by a combination of Unit and Functional tests, using the based on OV approach. To run tests, execute using the following commands:
```
cmake -DENABLE_FUNCTIONAL_TESTS=ON .
make ov_subgraphs_dumper_tests
./ov_subgraphs_dumper_tests --gtest_filter=*
```

## Architecture Diagram
![SubgraphsDumper Architecture Diagram](../../../../../../docs/articles_en/assets/images/subgraphs_dumper_arch_diaram.png)
