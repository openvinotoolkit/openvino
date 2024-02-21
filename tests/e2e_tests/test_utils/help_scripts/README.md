# Scripts to help getting special information about  E2E tests

This folder contains supporting scripts for collecting any information about E2E tests.

 - `get_test_name.py` -  Script to collect all names of tests. In additional script can collect names of tests which using only skipped MO arguments.<br/>
As result of script is `result.txt` file which will save near `get_test_name.py`
 - `get_reference_res_from_html.py` - Script to collect reference values from HTML report. <br/> Can be used to collect metrics `e2e_tests/plugins/first_time_inference_tests/.automation/references_negative_ones/test_references_config.yml`.

### How to use it
1. `get_test_name.py`<br/>
To run script need to use following keys `--modules` and `--skip_mo_args`(this key is optional)<br/> 
For get all names of tests:<br/>
> Note: script must be run from e2e folder
``` bash
pytest get_tests_name.py --modules=<path_to_tests>
 ```
<br/>For get all names of tests using skipped MO arguments:<br/>
``` bash
pytest get_tests_name.py --modules=<path_to_tests> --skip_mo_args=<list_of_mo_args>
 ```
Example: 
```
pytest get_tests_name.py --modules=C:\frameworks.ai.openvino.tests\e2e_tests\pipelines\production\caffe
or
pytest get_tests_name.py --modules=C:\frameworks.ai.openvino.tests\e2e_tests\pipelines\production\caffe --skip_mo_args=input,input_shape
```
2. `get_reference_res_from_html.py`<br/>
To run script need to use following keys `--html` and `--ref_yml_path`<br/>
``` bash
python get_reference_res_from_html.py --html=<path_to_html> --ref_yml_path=<path_to_ref_yml_file>
 ```