#!groovy

properties([
    parameters([
        booleanParam(defaultValue: true,
                     description: 'Cancel the rest of parallel stages if one of them fails and return status immediately',
                     name: 'failFast'),
        string(defaultValue: '',
               description: 'Pipeline shared library version (branch/tag/commit). Determined automatically if empty',
               name: 'library_version')
    ])
])
loadOpenVinoLibrary {
    entrypoint(this)
}
