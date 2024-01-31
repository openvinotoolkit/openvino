# Overview of the Reusable Workflows used in the OpenVINO GitHub Actions CI

To reduce duplication and increase maintainability, the common jobs from different workflows are extracted into _reusable workflows_.

You can find more information about reusing actions and workflows [here](https://github.com/marketplace?type=actions) and [here](https://docs.github.com/en/actions/using-workflows/reusing-workflows).

The reusable workflows are referenced as `jobs` in several validation workflows. They are structured and behave like [normal jobs](./overview.md#single-job-overview) with their own environment, runner and steps to execute.

This document describes the setup used in the OpenVINO GitHub Actions.

## Workflows Organisation

You can find all the workflows for this repository [here](../../../../.github/workflows).

Two categories of the workflows are present: 
* the workflow files starting with the OS name: e.g. [`linux.yml`](./../../../../.github/workflows/linux.yml), [`windows_conditional_compilation.yml`](./../../../../.github/workflows/windows_conditional_compilation.yml)
* the workflow files starting with the word `job`: e.g., [`job_cxx_unit_tests.yml`](./../../../../.github/workflows/job_cxx_unit_tests.yml), [`job_samples_tests.yml`](./../../../../.github/workflows/job_samples_tests.yml)

The former are the validation workflows incorporating building and testing of the corresponding OS, architecture and set of tools. Read more [here](./overview.md#structure-of-the-workflows) about these workflows.

The latter are the _reusable workflows_ that are used as jobs in several other workflows.

For example, the [`job_python_unit_tests.yml`](./../../../../.github/workflows/job_python_unit_tests.yml) reusable workflow is used in the [`linux.yml`](./../../../../.github/workflows/linux.yml), [`linux_arm64.yml`](./../../../../.github/workflows/linux_arm64.yml), 
[`mac.yml`](./../../../../.github/workflows/mac.yml) and [`mac_arm64.yml`](./../../../../.github/workflows/mac_arm64.yml) workflows as a `Python_Unit_Tests` job:
```yaml
  Python_Unit_Tests:
    name: Python unit tests
    needs: [ Build, Smart_CI ]
    uses: ./.github/workflows/job_python_unit_tests.yml
    with:
      runner: 'aks-linux-4-cores-16gb'
      container: '{"image": "openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04", "volumes": ["/mount:/mount"]}'
      affected-components: ${{ needs.smart_ci.outputs.affected_components }}
```

Refer to the next section for the usage reference.

## Using Reusable Workflows

Refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/reusing-workflows#calling-a-reusable-workflow) for a complete reference.

To use a reusable workflow, it should be referenced as a `job`. The [`job_python_unit_tests.yml`](./../../../../.github/workflows/job_python_unit_tests.yml) reusable workflow example 
in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow:
```yaml
  Python_Unit_Tests:
    name: Python unit tests
    needs: [ Build, Smart_CI ]
    uses: ./.github/workflows/job_python_unit_tests.yml
    with:
      runner: 'aks-linux-4-cores-16gb'
      container: '{"image": "openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04", "volumes": ["/mount:/mount"]}'
      affected-components: ${{ needs.smart_ci.outputs.affected_components }}
```
where:
* `name` - the display name of the job
* `needs` - the job's needs, i.e., the jobs that should be completed before this one starts
* `uses` - the path to the reusable workflow
* `with` - the input keys that will be passed to the reusable workflow. Refer to the workflow file to learn more about its inputs, refer to the [official GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/reusing-workflows#using-inputs-and-secrets-in-a-reusable-workflow) for a syntax reference

## Adding Reusable Workflows

If you would like to add new similar stages to several workflows, consider creating a reusable workflow to reduce duplication.

The reusable workflows in the OpenVINO GitHub Actions CI usually have:
* the filename starting with `job_`, e.g., [`job_cxx_unit_tests.yml`](./../../../../.github/workflows/job_cxx_unit_tests.yml)
* the `runner` input - the runner name that will be used to execute the steps in a job, learn more about the available runners and how to use them [here](./runners.md)
* the `container` input - JSON to be converted to the value of the "container" configuration for the job, learn more about using Docker in the workflows [here](./docker_images.md)
* *Optional* the `affected-components` input - components that are affected by changes in the commit defined by the Smart CI Action, learn more about the Smart CI system [here](./smart_ci.md)

*Note*: Per the [GitHub documentation](https://docs.github.com/en/actions/using-workflows/about-workflows#about-workflows), all workflows should be placed under [`./.github/workflows`](./../../../../.github/workflows).

As the reusable workflows are structured and behave like jobs, refer to [this document](./adding_tests.md) to learn more about creating a job and
use the information about the reusable workflows to make a job into a reusable workflow.
