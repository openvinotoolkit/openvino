# How to add Tests to the OpenVINO GitHub Actions CI

The OpenVINO repository has [many workflows](./../../../../.github/workflows), which contain
jobs for building and testing OpenVINO. Their general and structural overview is
available in the [OpenVINO GitHub Actions CI page](./overview.md).

This document explains how to add tests to existing workflows and create new workflows.

## Add Tests to Existing Workflow

### Add Tests to Existing Test Suite

If the new tests could be executed as part of the existing test suite, for example,
new OVC Python API tests, there is no need to change the workflows.
The added tests will automatically run in the corresponding step.

Review [workflows](./../../../../.github/workflows) and their jobs to identify which tests
are already enabled. Additionally, examine the component tests and how they are executed.

### Create a Step in a Job

If a job does not include a step with the required test suite, a new step can be added to the job.
Steps are the commands executed consecutively and grouped together under a single job.
Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/about-workflows) for more information.

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
The step includes:
* a `name`: `OVC unit tests`.
* an `if` condition: `fromJSON(inputs.affected-components).MO.test`
  * This step is executed only if the condition is `true`.
  * This is a part of the Smart CI system implemented for the OpenVINO workflow. Read the [Smart CI Overview](./smart_ci.md) to learn about the system and its usage.
* a `run` section with commands to be executed.

To add a new step with new tests, navigate to the job and use the above template (or any existing
step in the job) for the new step. Refer to the [Overview of the Reusable Workflows](./reusable_workflows.md) to learn
more about workflows and job organization.

### Create a New Job

If the new tests do not align with any existing job across all workflows, it is possible to
create a dedicated job for them.

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
* requires a name, provided by the `name` key
* requires a runner to execute `steps` on, provided by the `runs-on` key
  * Refer to the [Overview of Runners](./runners.md) to learn more about available runners and how to choose one
* might use Docker to execute `steps` in, configured by the `container` key
  * Refer to the [Overview of Docker Images](./docker_images.md) to learn more about the available Docker images and how to choose one
* might use caches to speed up build and/or tests
  * Different types of caches are available. Refer to the [Overview of Caches](./caches.md) to learn more about available caches and how to use them
* might use the Smart CI system for conditional execution with the `if` key
  * Refer to the [Smart CI Overview](./smart_ci.md) for more information
* requires a series of commands to execute, provided by the `steps` key
  * Refer to the [creating steps in a job section](#create-a-step-in-a-job) to learn more about `steps`
* might use build artefacts from the `Build` job
  * The artefacts can be downloaded using the `actions/download-artifact`, read more about the workflows' structure in the [Overview of the OpenVINO GitHub Actions CI](./overview.md#structure-of-the-workflows)

If the job can be used in several workflows, it can be transformed into a reusable workflow.
Learn more in the [Overview of Reusable Workflows](./reusable_workflows.md).

## Create a Workflow

To create a new workflow, add a new `<name>.yml` file to the [`.github/workflows`](./../../../../.github) folder.
Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/about-workflows#create-an-example-workflow) for a complete syntax reference, and browse the existing workflows in [`.github/workflows`](./../../../../.github).

You can refer to the [structural overview of the existing workflows](./overview.md#structure-of-the-workflows) as a template for a new workflow.

The [`fedora.yml`](./../../../../.github/workflows/fedora.yml) workflow example includes:
* `Smart_CI`, `Build`, `RPM_Packages`, `Overall_Status` jobs:
  * `Smart_CI` - the [Smart CI system](./smart_ci.md).
  * `Build` - prerequisites installation, building OpenVINO with specified CMake configuration, packaging and uploading artefacts.
  * `RPM_Packages` - prerequisites installation, downloading artefacts and tests.
  * `Overall_Status` - the job for collecting statuses of other jobs
* uploading and downloading the build artefacts between jobs using `actions/upload-artifact` and `actions/download-artifact`
* usage of the [Smart CI system](./smart_ci.md)
* usage of the [self-hosted runners](./runners.md) and [Docker images](./docker_images.md)
* usage of [caches](./caches.md)

## Test Time and Usage

Be mindful about the time and runners usage when adding new steps, jobs and workflows.

### Adding a Step

When adding a step in a job, check its execution time compared to other jobs. Try to
keep it in line with these jobs to avoid extenfing the overall job duration too much.

If the step takes too long, consider [extracting it to a separate job](#adding-a-job)
so that it can run in parallel with other jobs. Additionally, for jobs with long steps,
there is an option to [pick a more powerful runner](./runners.md) to shorten the execution time.

### Adding a Job

When adding a job, check the execution time of existing jobs in the workflow. The new job's
execution time should not exceed the time of the longest job in the workflow.

If the job is time-consuming, consider running it on a post-commit/nightly/weekly basis,
instead of pre-commit.
Refer to the [Overview of the OpenVINO GitHub Actions CI](./overview.md#workflows-triggers-and-schedule) to learn more about triggers and schedules.
Additionally, [using a more powerful runner](./runners.md) can help shorten the execution time.

### Adding a Workflow

When adding a new workflow, check the execution times of existing workflows. The new workflow's
execution time should not exceed the time of the longest workflow.

If the workflow time-consuming, consider running it on a post-commit/nightly/weekly basis,
instead of pre-commit.
Refer to the [Overview of the OpenVINO GitHub Actions CI](./overview.md#workflows-triggers-and-schedule)
to learn more about triggers and schedules.
Additionally, make sure [the right runners](./runners.md) are picked for each job to optimize
execution time.
