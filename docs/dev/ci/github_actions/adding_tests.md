# How to add Tests to the OpenVINO GitHub Actions CI

The OpenVINO repository has [many workflows](./../../../../.github/workflows), their general and structural overview is available [here](./overview.md).  

The workflows have many jobs dedicated to building and testing of OpenVINO. This document describes the topic of adding 
tests to these workflows or adding an entirely new workflow.

## Add Tests to the Already Existing Workflow

### Add Tests to the Existing Test Suite

If the new tests could be executed as a part of the already existing test suite, e.g., new OVC Python API tests, 
there is no need to change the workflows, the added tests would be executed automatically in the corresponding step. 

Review the [workflows](./../../../../.github/workflows) and their jobs to know which tests are already enabled. 
Additionally, review the component's tests and how they are executed.

### Create a Step in a Job

If there is no step in the jobs that has the needed test suite, a new step could be added to the job. 
The steps are the commands that are executed one by one and united under one job. 
Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/about-workflows) for more.

An example step from [`job_python_unit_tests.yml`](./../../../../.github/workflows/job_python_unit_tests.yml):
```yaml
...
steps:
...
  - name: OVC unit tests
    if: fromJSON(inputs.affected-components).MO.test
    run: python3 -m pytest -s ${INSTALL_TEST_DIR}/ovc/unit_tests --junitxml=${INSTALL_TEST_DIR}/TEST-OpenVinoConversion.xml
...
```
It has:
* a `name`: `OVC unit tests`
* `if` condition: `fromJSON(inputs.affected-components).MO.test`
  * This step is executed only if the condition evaluates to `true`
  * This is a part of the Smart CI system implemented for the OpenVINO workflow. Read [here](./smart_ci.md) about the system and how to use it
* the `run` section with commands to be executed

To add a new step with new tests, navigate to the needed job and use the above template (or any other step in the job) for the new step. 
Refer to [this document](./reusable_workflows.md) to learn more about the workflow and job organisation.

### Create a New Job

If the new tests do not fit in any of the jobs in all the workflows, it is possible to create a dedicated job for them. 
An example dedicated job for a single set of tests from [`linux.yml`](./../../../../.github/workflows/linux.yml):
```yaml
NVIDIA_Plugin:
  name: NVIDIA plugin
  needs: [ Build, Smart_CI ]
  timeout-minutes: 15
  defaults:
    run:
      shell: bash
  runs-on: aks-linux-16-cores-32gb
  container:
    image: openvinogithubactions.azurecr.io/dockerhub/nvidia/cuda:11.8.0-runtime-ubuntu20.04
    volumes:
      - /mount:/mount
    options: -e SCCACHE_AZURE_BLOB_CONTAINER -e SCCACHE_AZURE_CONNECTION_STRING
  env:
    CMAKE_BUILD_TYPE: 'Release'
    CMAKE_GENERATOR: 'Ninja Multi-Config'
    CMAKE_CUDA_COMPILER_LAUNCHER: sccache
    CMAKE_CXX_COMPILER_LAUNCHER: sccache
    CMAKE_C_COMPILER_LAUNCHER: sccache
    INSTALL_DIR: /__w/openvino/openvino/install
    OPENVINO_DEVELOPER_PACKAGE: /__w/openvino/openvino/install/developer_package
    OPENVINO_REPO: /__w/openvino/openvino/openvino
    OPENVINO_CONTRIB_REPO: /__w/openvino/openvino/openvino_contrib
    NVIDIA_BUILD_DIR: /__w/openvino/openvino/nvidia_plugin_build
    DEBIAN_FRONTEND: 'noninteractive'
    SCCACHE_AZURE_KEY_PREFIX: ubuntu20_x86_64_Release
  if: fromJSON(needs.smart_ci.outputs.affected_components).NVIDIA
  
  steps:
  ...
```

Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#about-yaml-syntax-for-workflows) for a complete syntax reference.

A job:
* needs a name, provided by the `name` key
* needs a runner to execute `steps` on, provided by the `runs-on` key 
  * Refer to [this document](./runners.md) to learn more about the available runners and how to choose one
* might use Docker to execute `steps` in. The Docker configuration is provided by the `container` key 
  * Refer to [this document](./docker_images.md) to learn more about the available Docker images and how to choose one
* might use caches to speed up build and/or tests
  * Different types of caches are available. Refer to [this document](./caches.md) to learn more about the available caches and how to use one
* might use the Smart CI system to get executed conditionally with the `if` key
  * Refer to [this document](./smart_ci.md) for the Smart CI overview and usage
* a series of commands to execute, provided by the `steps` key
  * Refer to [this section](#create-a-step-in-the-job) to learn more about `steps`
* might use the build artefacts from the `Build` job
  * They could be downloaded using the `actions/download-artifact`, read more about the workflows' structure [here](./overview.md#structure-of-the-workflows)

If the job could be used in several workflows, it could be transformed into a reusable workflow. 
Read more about the reusable workflows [here](./reusable_workflows.md).

## Create a Dedicated Workflow

To introduce a new workflow, add a new `<name>.yml` file to the [`.github/workflows`](./../../../../.github) folder. 
Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/about-workflows#create-an-example-workflow) for a complete syntax reference and browse the existing workflows in [`.github/workflows`](./../../../../.github).

Refer to the [structural overview of the existing workflows](./overview.md#structure-of-the-workflows), their structure could be used as a template for a new one.

The dedicated workflow example is [`fedora.yml`](./../../../../.github/workflows/fedora.yml). It has:
* `Smart_CI`, `Build`, `RPM_Packages`, `Overall_Status` jobs
  * `Smart_CI` - the [Smart CI system](./smart_ci.md)
  * `Build` - pre-requisites installation, building of OpenVINO with certain CMake configuration, packaging and uploading of the artefacts
  * `RPM_Packages` - pre-requisites installation, downloading of the artefacts and tests
  * `Overall_Status` - the job for collecting the other jobs' statuses
* the uploading and downloading of the build artefacts between jobs using `actions/upload-artifact` and `actions/download-artifact`
* the usage of the [Smart CI system](./smart_ci.md)
* the usage of the [self-hosted runners](./runners.md) and [Docker images](./docker_images.md)
* the usage of [caches](./caches.md)

## Test Times and Usage

Be mindful about time and runners usage when adding new steps, jobs and workflows.

### Adding a Step

When adding a step in a job, consider checking the times of the other steps in the job, 
it is best if the step's execution time does not lengthen the execution time of the job too much and is in-line with the execution times of other steps.

If the step takes a lot of time, it might be better to [extract it into a separate job](#adding-a-job) so that it runs in parallel with other jobs. 
Additionally, when creating a job with this step, it would be possible to [pick a more powerful runner](./runners.md) to shorten the execution time.

### Adding a Job

When adding a job, consider checking the times of the other jobs in a workflow, it is best if the new job's execution time 
does not exceed the time of the longest job in the workflow.

If the job takes a lot of time, it might be possible to run it not on the pre-commit basis but on a post-commit/nightly/weekly basis. 
Refer to [this document](./overview.md#workflows-triggers-and-schedule) to learn more about triggers and schedules. 
Additionally, it could be possible to [pick a more powerful runner](./runners.md) to shorten the execution time.

### Adding a Workflow

When adding a workflow, consider checking the times of the other workflows, it is best if the new workflow's execution time 
does not exceed the time of the longest workflow.

If the workflow takes a lot of time, it might be possible to run it not on the pre-commit basis but on a post-commit/nightly/weekly basis. 
Refer to [this document](./overview.md#workflows-triggers-and-schedule) to learn more about triggers and schedules. 
Additionally, make sure [the right runners](./runners.md) are picked for each job so that the execution times are optimal.
