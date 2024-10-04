# Overview of the OpenVINO GitHub Actions CI

Welcome to the OpenVINO Developer guide on the GitHub Actions infrastructure.
This document will give you an overview of the setup used in OpenVINO and point you at more
detailed instructions where necessary.

## Table of Contents

* [Workflows overview](#workflows)
  * [Triggers and schedules](#workflows-triggers-and-schedule)
  * [Required workflows](#required-workflows)
  * [Workflow structure](#structure-of-the-workflows)
  * [Workflow and job organisation](#workflows-and-jobs-organisation)
* [Finding results, artifacts and logs](#finding-results-artifacts-and-logs)
* [Custom actions overview](#custom-actions)
* [Machines overview](#machines)
* [Docker images overview](#docker-images)
* [Caches overview](#caches)
* [How to add new tests](#adding-new-tests)
* [Optimizing workflow based on PR changes](#optimizing-workflow-based-on-PR-changes)

## Workflows

GitHub Actions workflows are configurable and automated processes that run one or multiple
consecutive jobs (for more details, refer to the
[official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/about-workflows)).
They include:

* a series of commands that you would usually execute in a terminal, one by one
* information about the environment in which the commands should be executed


You can find all workflows for this repository in the [workflows folder](../../../../.github/workflows).
The three main ones, providing most coverage for different operating systems, are:
* [Linux](../../../../.github/workflows/ubuntu_22.yml)
* [Windows](../../../../.github/workflows/windows.yml)
* [macOS](../../../../.github/workflows/mac.yml)

Additionally, several supporting workflows build and test OpenVINO for other operating systems and processor architectures:
* [Android ARM64](../../../../.github/workflows/android_arm64.yml)
* [Fedora](../../../../.github/workflows/fedora_29.yml)
* [Linux Conditional Compilation](../../../../.github/workflows/linux_conditional_compilation.yml)
* [Linux RISC-V](../../../../.github/workflows/linux_riscv.yml)
* [Windows Conditional Compilation](../../../../.github/workflows/windows_conditional_compilation.yml)

### Reusing GitHub Actions

The OpenVINO workflows use both official and community-built actions, such as `actions/checkout`
and `actions/upload-artifact`. Additionally, jobs featured in several workflows are extracted
into _reusable workflows_. You can learn more about [using and writing them](./reusable_workflows.md),
check how to [reuse workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows),
and see what and how to [obtain additional actions](https://github.com/marketplace?type=actions).

### Workflows' Triggers and Schedule

Workflows run whenever they are triggered by predefined [events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows).
These triggers **are not** mutually exclusive and multiple can be used by one workflow.
The OpenVINO repository has three, and as you may see in the example below, they are all
included in the [Linux workflow](../../../../.github/workflows/ubuntu_22.yml). They are:

* `on: schedule` - schedule trigger
  * This trigger runs the workflow on a specified interval (e.g., nightly).
  * In the example below: `'0 0 * * 3,6'` - learn more on [cron syntax](https://crontab.guru/)
* `on: pull_request` - pre-commit trigger
  * This trigger runs the workflow when a pull request (PR) is created targeting the `master` or `release`
    branch and every time the PR is updated with new commits.
  * In the example below, it additionally requires that the changed files conform to the path
    globs specified under the `paths` key.
* `on: push` - post-commit trigger.
  * This trigger runs the workflow when a commit is pushed to the `master` or `release` branch
    (e.g., when a PR is merged).
  * In the example below, it additionally requires that the changed files conform to the path
    globs specified under the `paths` key.

The triggers for each workflow can be found at the beginning of a workflow file, in the `on`
key. You should also learn how to use
[paths](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore).




```yaml
on:
  schedule:
    # at 00:00 on Wednesday and Saturday
    - cron: '0 0 * * 3,6'
  pull_request:
    paths:
      - '**'
      - '!**/docs/**'
      - '!docs/**'
      - 'docs/snippets/**'
      - '!**/**.md'
      - '!**.md'
  push:
    paths:
      - '**'
      - '!docs/**'
      - '!**/docs/**'
      - 'docs/snippets/**'
      - '!**/**.md'
      - '!**.md'
    branches:
      - master
      - 'releases/**'
```

---
**NOTE**

The workflows listed above are **required** for OpenVINO contributions. If they fail the PR
cannot be merged. It is always a good idea to check their
[results](#finding-results-artifacts-and-logs) while working within the OpenVINO repository.

---


### Workflow Structure

The workflow structures for Linux, Windows, and macOS are mostly the same:

1. Clone the OpenVINO repository and required resources
2. Install build dependencies
3. Build OpenVINO from source
4. Pack and upload the artifacts (the built OpenVINO and tests)
5. Download and use the artifacts in the parallel jobs with different tests
6. Collect the test results and upload them as artifacts

**NOTE**: some workflows may use the same structure, while others may lack the last 3 steps,
with tests coming right after the `Build` step.

Overview of the [Linux workflow](../../../../.github/workflows/ubuntu_22.yml).
There are several jobs present:

```yaml
jobs:
  Build: ...
  Debian_Packages: ...
  Samples: ...
  Conformance: ...
  ONNX_Runtime: ...
  CXX_Unit_Tests: ...
  Python_Unit_Tests: ...
  CPU_Functional_Tests: ...
  TensorFlow_Hub_Models_Tests: ...
  PyTorch_Models_Tests: ...
  NVIDIA_Plugin: ...
```

The `Build` job executes the first 4 steps:
* clones OpenVINO
* installs dependencies
* builds from source with `cmake`
* packs and uploads the artifacts using `actions/upload-artifact`

The other jobs are responsible for running different tests using the built artifacts. They:
* download and unpack the artifacts using `actions/download-artifact`
* install the needed dependencies
* run tests
* collect test results
* upload test results as [pipeline artifacts](#artifacts)

#### Single Job Overview

Each job has several keys that describe its environment. Consider checking a comprehensive
[syntax overview](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions).

This section describes the specifics of the OpenVINO CI environment.

Overview of the [Linux workflow's](../../../../.github/workflows/ubuntu_22.yml) `Python_Unit_Tests` job:
```yaml
  Python_Unit_Tests:
    name: Python unit tests
    needs: Build
    timeout-minutes: 40
    defaults:
      run:
        shell: bash
    runs-on: aks-linux-4-cores-16gb
    container:
      image: openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04
      volumes:
        - /mount/caches:/mount/caches
    env:
      OPENVINO_REPO: /__w/openvino/openvino/openvino
      INSTALL_DIR: /__w/openvino/openvino/install
      INSTALL_TEST_DIR: /__w/openvino/openvino/install/tests
      LAYER_TESTS_INSTALL_DIR: /__w/openvino/openvino/install/tests/layer_tests

    steps: ...
```

* All the test jobs have the `needs: Build` which means that they wait for the `Build` job to
  finish as they require artifacts from it.
* The machine that is used for a job is specified using the `runs-on` key.
  * In this case `aks-linux-4-cores-16gb` is used. [Read more](#machines) on what machines are
    available and how to choose one for a job.
* Some jobs could run inside a Docker container. The image could be specified using the `image`
  key under the `container` key.
  * In this case, `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` is used.
    [Read more](#docker-images) on what images are available and when to use one.
* Some jobs may benefit from caching, for example, Python dependencies or `cmake` build artifacts.
  * [Read more](#caches) on how to utilize cache for a job.
* A job must define `steps` - a series of commands to execute in the predefined environment.
  * All the steps are executed in the shell specified by the `shell` key under `defaults: run:`
    unless a shell is specified directly in a step.

## Finding Results, Artifacts, and Logs

### Results

To understand which jobs have successfully passed, which are running and which have failed,
check the following:

**For Pull Requests:**
  * Open a Pull Request and scroll to the bottom of the page. You will see a list of jobs,
    both finished and still running for the most recent commit:

    ![check_results](../../assets/CI_check_results.png)

**For scheduled runs:**
  * Navigate to the [OpenVINO Repository Actions](https://github.com/openvinotoolkit/openvino/actions)
  * Select the required workflow from the list on the left
  * Filter the runs by clicking on `Event` and selecting `schedule`
    * You can additionally filter the results per branch, actor and result

### Artifacts

Artifacts, that is files produced by the workflow, are available only for the completed pipelines,
both successful or failed. To find artifacts for a pipeline, follow these steps:

1. Open a Pull Request and scroll to the list of jobs, as described above.
2. Click `Details`, to the right of the selected job, to see more information about it.
3. Click `Summary`, above the list of jobs on the left side of the window:

   ![jobs_list](../../assets/CI_completed_job_list.png)

4. Scroll to the bottom of the page
5. You will find the artifacts produced by **all the jobs in this pipeline**:

   ![pipeline_artifacts](../../assets/CI_pipeline_artifacts.png)

6. Click on the artifact name to download it.


### Logs

To find logs for a pipeline:
1. Open a Pull Request and scroll to the list of jobs, as described above.
2. Click `Details`, to the right of the selected job, to see more information about it.
3. Click on a step to see its logs

## Custom Actions

Several actions are written specifically for the needs of the OpenVINO workflows. Read more
about the available custom actions and what they do in the [custom actions document](./custom_actions.md).

Check the [Reusing GitHub Actions](#reusing-github-actions) section for more information.

## Machines

The machines that execute the commands from the workflows are called _runners_ in GitHub Actions.
Two types of runners are available for the OpenVINO organization:

* [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners) - runners provided and managed by GitHub
* [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) - runners created and managed by the OpenVINO CI team and linked to the OpenVINO repositories

Workflows utilize appropriate runners based on their jobs' needs. Learn more about the
available runners and how to choose one in the [OpenVINO Runner Overview](./runners.md).

## Docker Images

You can run jobs in Docker containers.
Refer to [the documentation for syntax overview](https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container).

Workflows utilize appropriate Docker images based on their jobs' needs. Learn more about the
available images and how to choose one in the [OpenVINO Docker Image Overview](./docker_images.md).

## Caches

Three types of caches are available:
* [GitHub Actions cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
* Shared drive cache
* Remote build cache via [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs)

workflows utilize appropriate caches based on their jobs' needs. Learn more about the
available caches and how to use one in the [OpenVINO Cache Overview](./caches.md).

## Adding New Tests

If you would like to add new tests, refer to the [How to add Tests](./adding_tests.md) document.

## Optimizing workflows based on PR changes

To optimize pre-commit workflow by running only the jobs that are actually required to validate
changes in a pull request, you can use the Smart CI feature - [learn more about it](./smart_ci.md).

## See also

* [GitHub Actions official documentation](https://docs.github.com/en/actions)
