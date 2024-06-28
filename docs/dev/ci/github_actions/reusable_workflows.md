# Reusable Workflows

To reduce duplication and increase maintainability, common jobs from different workflows are
extracted into _reusable workflows_.

You can find more information about reusing [actions](https://github.com/marketplace?type=actions)
and [workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows) in GitHub docs.

The reusable workflows are referenced as `jobs` in several validation workflows.
They are structured and function like [regular jobs](./overview.md#single-job-overview)
with their own environment, runner, and steps to execute.

This document describes the setup used in the OpenVINO GitHub Actions.

## Workflows Organisation

You can find all workflows for this repository in the [workflows folder](../../../../.github/workflows).

There are two categories of workflows:
* files starting with the OS name, for example: [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml), [`windows_conditional_compilation.yml`](./../../../../.github/workflows/windows_conditional_compilation.yml). These are validation workflows that include building and testing of the corresponding OS,
architecture and set of tools. Read more on the [workflows page](./overview.md#structure-of-the-workflows).
* files starting with the word `job`, for example: [`job_cxx_unit_tests.yml`](./../../../../.github/workflows/job_cxx_unit_tests.yml), [`job_samples_tests.yml`](./../../../../.github/workflows/job_samples_tests.yml). These workflows are  _reusable workflows_ used as jobs in several other workflows.

For example, the [`job_python_unit_tests.yml`](./../../../../.github/workflows/job_python_unit_tests.yml) reusable workflow is used in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml), [`linux_arm64.yml`](./../../../../.github/workflows/linux_arm64.yml),
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

## Using Reusable Workflows

A reusable workflow should be referenced as a `job`.
The [`job_python_unit_tests.yml`](./../../../../.github/workflows/job_python_unit_tests.yml)
reusable workflow example in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow:
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
* `name` - the display name of the job;
* `needs` - the job's dependencies: the jobs that must be completed before this one starts;
* `uses` - the path to the reusable workflow;
* `with` - the input keys passed to the reusable workflow. Refer to the workflow file to learn more about its inputs. Refer to the [GitHub Actions documentation](https://docs.github.com/en/actions/using-workflows/reusing-workflows#using-inputs-and-secrets-in-a-reusable-workflow) for a syntax reference.

Refer to the [GitHub Actions documentation on reusable workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows#calling-a-reusable-workflow) for a complete reference.

## Adding Reusable Workflows

To reduce duplication while adding similar stages to several workflows, create a reusable workflow.

In the OpenVINO GitHub Actions CI, reusable workflows typically have:
* the filename starting with `job_`, for example, [`job_cxx_unit_tests.yml`](./../../../../.github/workflows/job_cxx_unit_tests.yml)
* the `runner` input, specifying the runner name used to execute the steps in a job. Learn more about [available runners and how to use them](./runners.md)
* the `container` input represented as a JSON, which is converted to the value of the "container" configuration for the job. Learn more about [using Docker in the workflows](./docker_images.md)
* *Optional* the `affected-components` input, indicating components affected by changes in the commit defined by the Smart CI Action. Learn more about the [Smart CI system](./smart_ci.md)

>**NOTE**: All workflows should be placed under [`./.github/workflows`](./../../../../.github/workflows) according to the [GitHub documentation](https://docs.github.com/en/actions/using-workflows/about-workflows#about-workflows).

Since reusable workflows are structured and behave like jobs, you can refer to the [adding tests page](./adding_tests.md) to learn more about creating a job and
use the information about the reusable workflows to transform a job into a reusable workflow.
