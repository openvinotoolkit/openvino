#!groovy
properties([
    parameters([
        booleanParam(defaultValue: true,
                     description: 'Cancel the rest of parallel stages if one of them fails and return status immediately',
                     name: 'failFast')
    ])
])

dldtPipelineEntrypoint(this)
