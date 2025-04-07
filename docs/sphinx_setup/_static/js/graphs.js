// =================== GENERAL OUTPUT CONFIG =========================

class Filter {
    // param: GraphData[], networkModels[]
    static FilterByNetworkModel(graphDataArr, networkModels) {
        const optionMap = new Map();
        networkModels.map((model) => graphDataArr.filter((graphData => graphData.Model === model)))
            .flat(1)
            .forEach(item => optionMap.set(item.Platform, item));
        return Array.from(optionMap.values());
    }
    // param: GraphData[], ieType
    static ByIeTypes(graphDataArr, ieTypes) {
        const optionMap = new Map();
        graphDataArr
            .filter(graphData => ieTypes.includes(graphData.PlatformType))
            .forEach(item => optionMap.set(item.Platform, item));
        return Array.from(optionMap.values());
    }
    // param: GraphData[], ieType, networkModels
    static ByTypesAndModels(graphDataArr, ieTypes, models) {
        return Array.from(
            graphDataArr
                .filter(({ PlatformType, Model }) => ieTypes.includes(PlatformType) && models.includes(Model))
                .reduce((map, item) => map.set(item.Platform, item), new Map())
                .values()
        );
    }
    // param: GraphData[], clientPlatforms
    static ByIeKpis(graphDataArr, clientPlatforms) {
        return Array.from(
            graphDataArr.reduce((kpiSet, data) => {
                if (clientPlatforms.some(platformName => data.Platform.includes(platformName))) {
                    Object.keys(data.Parameters).forEach(key => kpiSet.add(key));
                }
                return kpiSet;
            }, new Set())
        );
    }
    // param: GraphData[]
    static getParameters(graphDataArr) {
        var parameters = []
        graphDataArr.filter((data) => {
            for (var key in data.Parameters) {
                if (!parameters.includes(Graph.capitalizeFirstLetter(key))) parameters.push(Graph.capitalizeFirstLetter(key))
            }
        })
        return parameters;
    }
    // param: GraphData[]
    static getIeTypes(graphDataArr) {
        var kpis = []
        graphDataArr.filter((data) => {
            for (var key in data.Parameters) {
                if (!kpis.includes(Graph.capitalizeFirstLetter(key))) kpis.push(Graph.capitalizeFirstLetter(key))
            }
        })
        return kpis;
    }
    // param: GraphData[], clientPlatforms[]
    static BySortPlatforms(graphDataArr, platformsArr) {
        return graphDataArr
            .filter((data) => platformsArr.includes(data.Platform))
            .sort((a, b) => a.Platform.localeCompare(b.Platform));
        //sort is necessary
    }
}

class Modal {
    static getPrecisionsLabels(graphDataArr) {
        const kpisSet = new Set();
        graphDataArr.forEach(data => {
            Object.values(data.Parameters).forEach(param => {
                param.Precisions.forEach(precision => {
                    Object.keys(precision).forEach(key => {
                        kpisSet.add(key.toUpperCase());
                    });
                });
            });
        });
        return Array.from(kpisSet);
    }

    static getPrecisions(appConfig, labels) {
        return labels.map((label) => {
            var prec = appConfig.PrecisionsMap[label];
            if (prec !== undefined) {
                return prec;
            }
            else {
                return "no name";
            }
        });
    }
}

class Graph {
    // functions to get unique keys 
    static getNetworkModels(graphDataArr) {
        return Array.from(new Set(graphDataArr.map(obj => obj.Model)))
            .sort((a, b) => a.localeCompare(b));
    }
    static getIeTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.PlatformType)))
            .sort((a, b) => a.localeCompare(b));
    }

    // param: GraphData[]
    static getPlatformNames(graphDataArr) {
        return graphDataArr.map((data) => data.Platform)
            .sort((a, b) => a.localeCompare(b));
    }

    // param: GraphData[], engine: string, precisions: list
    static getDatabyParameter(graphDataArr, engine, array) {
        if (!Array.isArray(array[engine])) {
            array[engine] = [];
        }
        array[engine].push(graphDataArr.Parameters[engine].Precisions);
        return array;
    }

    // this returns an object that is used to ender the chart
    static getGraphConfig(engine, precisions, appConfig) {
        return {
            chartTitle: 'Throughput vs Latency',
            iconClass: 'latency-icon',
            datasets: precisions.map((precision) => appConfig.PrecisionData[engine][precision]),
            unit: "None"
        };
    }

    // param: GraphData[], parameterName: string, precisions: list
    static getDatabyParameterOld(graphDataArr, parameterName, precisions) {
        var array = [];
        graphDataArr.forEach((item) => {
            if (item.Parameters[parameterName] !== undefined) {
                array.push(item.Parameters[parameterName].Precisions);
            }
            else {
                var obj = {};
                precisions.forEach((prec) => {
                    obj[prec] = 0;
                })
                array.push([obj])
            }
        })

        return array;
    }

    // this returns an object that is used to ender the chart
    static getGraphConfigOld(parameterName, item, precisions, appConfig) {
        return {
            chartTitle: Graph.capitalizeFirstLetter(parameterName),
            iconClass: parameterName + '-icon',
            unit: item.Parameters[parameterName]?.Unit,
            datasets: precisions.map((precision) => appConfig.PrecisionData[precision]),
        };
    }
    static capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
}

class ChartDisplay {
    constructor(mode, numberOfCharts) {
        this.mode = mode;
        this.numberOfChartsInRow = numberOfCharts;
    }
}

$(document).ready(function () {

    $('.ov-toolkit-benchmark-results').on('click', () => showModal("graph-data-ov.json", false));
    $('.ovms-toolkit-benchmark-results').on('click', () => showModal("graph-data-ovms.json", false));
    $('.ovms-toolkit-benchmark-llm-result').on('click', () => showModal("graph-data-ovms-genai.json", true));
    function clickBuildGraphs(graph, appConfig, networkModels, ieTypes, platforms, kpis, precisions, isLLM) {
        renderData(graph, appConfig, networkModels, ieTypes, platforms, kpis, precisions, isLLM);
        $('.modal-footer').show();
        $('#modal-display-graphs').show();
        $('.edit-settings-btn').off('click').on('click', (event) => {
            $('#modal-configure-graphs').show();
            $('#modal-display-graphs').hide();
            $('.modal-footer').hide();
            $('.chart-placeholder').empty();
        });

        $('.graph-chart-title-header').off('click').on('click', (event) => {
            var parent = event.target.parentElement;
            if ($(parent).children('.chart-wrap,.empty-chart-container').is(":visible")) {
                $(parent).children('.chart-wrap,.empty-chart-container').hide();
                $(parent).children('.chevron-right-btn').show();
                $(parent).children('.chevron-down-btn').hide();
            } else {
                $(parent).children('.chart-wrap,.empty-chart-container').show();
                $(parent).children('.chevron-down-btn').show();
                $(parent).children('.chevron-right-btn').hide();
            }
        });
    }

    function hideModal() {
        $('#graphModal').remove();
        $('body').css('overflow', 'auto');
    }

    function showModal(file, isLLM) {
        $('body').css('overflow', 'hidden');

        fetch('../_static/benchmarks_files/data/' + file)
            .then((response) => response.json())
            .then((jsonData) => {
                fetch('../_static/benchmarks_files/graph-config.json')
                    .then((configResponse) => configResponse.json())
                    .then((appConfig) => {
                        renderModal(jsonData, appConfig, isLLM)
                    })
            });
    }

    function getSelectedNetworkModels() {
        return $('.models-column input:checked, .platforms-column input:checked').not('[data-networkmodel="Select All"]').map(function () {
            return $(this).data('networkmodel');
        }).get();
    }

    function getSelectedIeTypes() {
        return $('.ietype-column input:checked').map(function () {
            return $(this).data('ietype');
        }).get();
    }

    function getSelectedClientPlatforms() {
        return $('.platforms-column input:checked').map(function () {
            return $(this).data('platform');
        }).get();
    }

    function getSelectedKpis() {
        return $('.kpi-column input:checked').map(function () {
            return $(this).data('kpi');
        }).get();
    }

    function getSelectedPrecisions() {
        return $('.precisions-column input:checked').map(function () {
            return $(this).data('precision');
        }).get();
    }

    function validateSelections() {
        if (getSelectedNetworkModels().length > 0
            && getSelectedIeTypes()
            && getSelectedClientPlatforms().length > 0
            && getSelectedKpis().length > 0) {
            if (getSelectedPrecisions().length > 0) {
                $('#build-graphs-btn').prop('disabled', false);
                return;
            }
            $('#build-graphs-btn').prop('disabled', true);
            return;
        }
        $('#build-graphs-btn').prop('disabled', true);
    }

    function renderModal(graph, appConfig, isLLM) {
        var modalPath = isLLM === true ? '../_static/html/modalLLM.html' : '../_static/html/modal.html'
        new Graph(graph);
        var networkModels = Graph.getNetworkModels(graph);
        var ieTypes = Graph.getIeTypes(graph);
        fetch(modalPath).then((response) => response.text()).then((text) => {

            // generate and configure modal container
            var modal = $('<div>');
            modal.attr('id', 'graphModal');
            modal.addClass('modal');
            var modalContent = $(text);
            modalContent.attr('id', 'graphModalContent');
            modalContent.addClass('modal-content');
            modal.append(modalContent);

            const models = networkModels.map((networkModel) => createCheckMark(networkModel, 'networkmodel'));
            modal.find('.models-column').append(models);

            const selectAllModelsButton = createCheckMark('', 'networkmodel', false, false);
            modal.find('.models-selectall').append(selectAllModelsButton);

            const selectAllPlatformsButton = createCheckMark('', 'platform', false, false);
            modal.find('.platforms-selectall').append(selectAllPlatformsButton);

            const precisions = Modal.getPrecisionsLabels(graph).map((precision) => createCheckMark(precision, 'precision', false, false));
            modal.find('.precisions-column').append(precisions);

            selectAllCheckboxes(precisions);
            disableAllCheckboxes(precisions);

            const selectAllTypesButton = createCheckMark('', 'ietype')
            modal.find('.ietype-selectall').append(selectAllTypesButton);

            const iefilter = ieTypes.map((ieType) => createCheckMark(ieType, 'ietype'));
            modal.find('.ietype-column').append(iefilter);

            modal.find('#modal-display-graphs').hide();
            modal.find('.ietype-column input').first().prop('checked', true);

            const kpiLabels = Filter.getParameters(graph).map((parameter) => createCheckMark(parameter, 'kpi', false, true));
            modal.find('.kpi-column').append(kpiLabels);

            $('body').prepend(modal);

            if (appConfig.DefaultSelections.platformTypes?.data?.includes('Select All')) {
                selectAllCheckboxes(iefilter);
            };
            preselectDefaultSettings(graph, modal, appConfig);
            renderClientPlatforms(graph, modal);

            $('#build-graphs-btn').on('click', () => {
                $('#modal-configure-graphs').hide();
                clickBuildGraphs(graph, appConfig, getSelectedNetworkModels(), getSelectedIeTypes(), getSelectedClientPlatforms(), getSelectedKpis(), Modal.getPrecisions(appConfig, getSelectedPrecisions()), isLLM);
            });
            $('.modal-close').on('click', hideModal);
            $('.close-btn').on('click', hideModal);

            modal.find('.ietype-selectall input').on('click', function () {
                if ($(this).prop('checked'))
                    selectAllCheckboxes(iefilter);
                else deSelectAllCheckboxes(iefilter);
            });

            modal.find('.models-selectall input').on('click', function () {
                if ($(this).prop('checked')) selectAllCheckboxes(models);
                else deSelectAllCheckboxes(models);

                renderClientPlatforms(graph, modal)
            });

            modal.find('.platforms-selectall input').on('click', function () {
                if ($(this).prop('checked'))
                    renderClientPlatforms(graph, modal)
                else {
                    var enabledPlatforms = modal.find('.platforms-column .checkmark-container');
                    deSelectCheckbox(enabledPlatforms);
                };

            });

            modal.find('.models-column input').on('click', function () {
                if (!$(this)[0].checked) {
                    deSelectCheckbox(selectAllModelsButton);
                }
            });


            modal.find('.ietype-column input').on('click', function () {
                if (!$(this)[0].checked) {
                    deSelectCheckbox(selectAllTypesButton);
                }
            });

            modal.find('.models-column input').on('click', () => renderClientPlatforms(graph, modal));
            modal.find('.ietype-column input').on('click', () => renderClientPlatforms(graph, modal));
            modal.find('.ietype-selectall input').on('click', () => renderClientPlatforms(graph, modal));
            modal.find('.platforms-column').on('click', () => enableParmeters(graph, getSelectedClientPlatforms()));

            modal.find('.kpi-column input').on('click', validatePrecisionSelection);
            modal.find('input').on('click', validateSelections);

            var modalFilters = document.getElementById("modal-filters");

            var showFiltersButton = document.getElementById("filters");
            showFiltersButton.onclick = function () {
                modalFilters.style.display = "block";
            }

            var closeFiltersButton = document.getElementsByClassName("close-filters")[0];
            closeFiltersButton.onclick = function () {
                modalFilters.style.display = "none";
            }

            window.onclick = function (event) {
                if (event.target == modalFilters) {
                    modalFilters.style.display = "none";
                }
            }
        });
    }

    function validatePrecisionSelection() {
        const precisions = $('.precisions-column').find('input')
        precisions.prop('disabled', false);
    }

    function preselectDefaultSettings(graph, modal, appConfig) {
        selectDefaultPlatformType(appConfig.DefaultSelections.platformTypes, graph, modal);
        clearAllSettings(appConfig.DefaultSelections);
        validateSelections();
        validatePrecisionSelection();
    }
    function selectDefaultPlatformType(platformTypes, graph, modal) {
        if (!platformTypes) return;
        const type = platformTypes.data[0];
        $(`input[data-ietype="${type}"]`).prop('checked', true);
        renderClientPlatforms(graph, modal);
    }

    function clearAllSettings(defaultSelections) {
        Object.keys(defaultSelections).forEach(setting => {
            const { name, data } = defaultSelections[setting];
            data.forEach(selection => {
                $(`input[data-${name}="${selection}"]`).prop('checked', true);
            });
        });
    }

    function filterClientPlatforms(graph, ietypes) {
        return Filter.ByIeTypes(graph, ietypes);
    }

    function filterPlatforms(graph, ietypes, models) {
        return Filter.ByTypesAndModels(graph, ietypes, models);
    }

    function renderClientPlatforms(graph, modal) {
        var fPlatforms = filterClientPlatforms(graph, getSelectedIeTypes());
        var platformNames = Graph.getPlatformNames(fPlatforms);
        $('.platforms-column .checkmark-container').remove();

        const clientPlatforms = platformNames.map((platform) => createCheckMark(platform, 'platform', true, false));

        var enabledPlatforms = filterPlatforms(graph, getSelectedIeTypes(), getSelectedNetworkModels());
        enableCheckBoxes(clientPlatforms, enabledPlatforms);
        modal.find('.platforms-column').append(clientPlatforms);

        enableParmeters(graph, getSelectedClientPlatforms());
        modal.find('.platforms-column input').on('click', validateSelections);
        validateSelections();
    }

    function enableParmeters(graph, clientPlatforms) {
        var allKpis = Filter.getParameters(graph);

        allKpis.forEach((kpi) => {
            $(`input[data-kpi="${Graph.capitalizeFirstLetter(kpi)}"]`).prop('disabled', true);
        })

        var kpis = Filter.ByIeKpis(graph, clientPlatforms);
        kpis.forEach((kpi) => {
            $(`input[data-kpi="${Graph.capitalizeFirstLetter(kpi)}"]`).prop('disabled', false);
        })
    }

    function createCheckMark(itemLabel, modelLabel, disabled, checked = false) {
        const item = $('<label class="checkmark-container">');
        item.text(itemLabel);
        const checkbox = $('<input type="checkbox"/>');
        checkbox.prop('disabled', disabled);
        checkbox.prop('checked', checked);
        const checkboxSpan = $('<span class="checkmark">');
        item.append(checkbox);
        item.append(checkboxSpan);
        checkbox.attr('data-' + modelLabel, itemLabel);
        return item;
    }

    // receives a jquery list of items and selects all input checkboxes
    function selectAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('checked', true);
        });
    }

    function enableCheckBoxes(items, enabledItems) {
        items.forEach((item) => {
            item.find(':input').prop('disabled', true);
            enabledItems.forEach((platform) => {
                var tmp = item.find(':input');
                if (tmp[0].dataset.platform === platform.Platform) {
                    item.find(':input').prop('checked', true);
                    item.find(':input').prop('disabled', false);
                }
            })
        })
    }

    function disableAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('disabled', true);
        })
    }

    function deSelectAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('checked', false);
        });
    }
    function deSelectCheckbox(item) {
        item.find(':input').prop('checked', false);
    }

    // =================== HTMLLEGEND =========================

    const getOrCreateLegendList = (chart, id) => {
        const legendContainer = document.getElementById(id);
        let listContainer = legendContainer.querySelector('ul');

        if (!listContainer) {
            listContainer = document.createElement('ul');
            listContainer.style.display = 'flex';
            listContainer.style.flexDirection = 'row';
            listContainer.style.margin = 0;
            listContainer.style.padding = 0;
            listContainer.style.paddingLeft = '0px';
            listContainer.style.float = "right";

            legendContainer.appendChild(listContainer);
        }

        return listContainer;
    };

    const htmlLegendPlugin = {
        id: 'htmlLegend',
        afterUpdate(chart, args, options) {
            charts = [...new Set([...charts, ...[chart]])];
            const ul = getOrCreateLegendList(chart, chart.options.plugins.htmlLegend.containerID);
            // Remove old legend items
            while (ul.firstChild) {
                ul.firstChild.remove();
            }

            const items = chart.options.plugins.legend.labels.generateLabels(chart);
            items.forEach(item => {
                const li = document.createElement('li');
                li.style.alignItems = 'center';
                li.style.display = 'block';
                li.style.flexDirection = 'column';
                li.style.marginLeft = '6px';
                li.style.cursor = "pointer";
                li.style.fontSize = '0.6rem';
                li.style.textDecoration = item.hidden ? 'line-through' : '';
                li.onclick = () => {
                    charts.forEach((chartItem) => {
                        chartItem.setDatasetVisibility(item.datasetIndex, !chartItem.isDatasetVisible(item.datasetIndex));
                        chartItem.update();
                      })
                };
                
                const boxSpan = document.createElement('span');
                boxSpan.style.background = item.fillStyle;
                boxSpan.style.borderColor = item.strokeStyle;
                boxSpan.style.display = 'inline-block';
                boxSpan.style.height = '10px';
                boxSpan.style.marginRight = '4px';
                boxSpan.style.width = '30px';

                const textSpan = document.createElement('span');
                textSpan.style.bottom = '1px'
                textSpan.style.position = 'relative'
                textSpan.style.fontSize = '0.6rem';
                textSpan.style.textDecoration = item.hidden ? 'line-through' : '';

                const text = document.createTextNode(item.text);
                textSpan.appendChild(text);

                li.appendChild(boxSpan);
                li.appendChild(textSpan);
                ul.appendChild(li);
            });
        }
    };

    function getChartOptionsByEngines(allowedAxisIDs) {
        const axisConfigs = {
            x: {
                title: { display: true, text: 'Request Rate' }
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: { display: true, text: 'Throughput [tok/s]' },
                grid: { drawOnChartArea: true }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: { display: true, text: 'TPOT Mean Latency' },
                grid: { drawOnChartArea: true }
            }
        };

        return {
            responsive: true,
            scales: Object.keys(axisConfigs)
                .filter(key => allowedAxisIDs.includes(key))
                .reduce((obj, key) => {
                    obj[key] = axisConfigs[key];
                    return obj;
                }, {}),
            plugins: {
                legend: { display: false },
                htmlLegend: { containerID: 'modal-footer' }
            }
        };
    }
    function getChartOptions(title) {
        return {
            responsive: true,
            indexAxis: 'y',
            maintainAspectRatio: false,
            title: {
                display: false,
                text: title
            },
            scales: {
                x: {
                    ticks: {
                        beginAtZero: true
                    }
                },
                y: {
                    ticks: {
                        display: false,
                        beginAtZero: true
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                htmlLegend: {
                    containerID: 'modal-footer',
                }
            }
        }
    }

    function renderData(graph, appConfig, networkModels, ieTypes, platforms, kpis, precisions, isLLM) {
        $('.chart-placeholder').empty();
        $('.modal-footer').empty();
        const display = new ChartDisplay(getChartsDisplayMode(kpis.length), kpis.length);

        networkModels.forEach((networkModel) => {
            var chartName = networkModel;
            var chartSlug = chartName.replace(')', '').replace(' (', '-');
            var chartContainer = $('<div>');
            var chevronDown = '<span class="chevron-down-btn"></span>';
            var chevronRight = '<span style="display:none" class="chevron-right-btn"></span>';
            $(chevronRight).hide();

            var chartContainerHeader = $(chevronDown + chevronRight + '<span class="graph-chart-title">' + networkModel + '</span>');
            chartContainerHeader.addClass('graph-chart-title-header');
            chartContainer.prepend(chartContainerHeader);
            chartContainer.attr('id', 'ov-chart-container-' + chartSlug);

            chartContainer.addClass('chart-container');

            var filteredNetworkModels = Filter.FilterByNetworkModel(graph, [networkModel]);
            var filteredIeTypes = Filter.ByIeTypes(filteredNetworkModels, ieTypes);
            var filteredGraphData = Filter.BySortPlatforms(filteredIeTypes, platforms);
            var filterdPlatforms = platforms.filter(platform =>
                filteredGraphData.some(filteredGraph => platform === filteredGraph.Platform)
              );
            $('.chart-placeholder').append(chartContainer);
            if (filteredGraphData.length > 0) {
                if (isLLM === true) {
                    var graphConfigs = setGraphConfigsByEngines(filteredGraphData, appConfig, kpis, precisions);
                    createChartWithNewDataByEngines(filterdPlatforms, graphConfigs, chartContainer, display);
                }
                else {
                    var graphConfigs = setGraphConfigs(filteredGraphData, appConfig, kpis, precisions);
                    createChartWithNewData(filterdPlatforms, graphConfigs, appConfig, chartContainer, display);
                }

            } else {
                createEmptyChartContainer(chartContainer);
            }
        })

        $(window).off('resize');
        $(window).resize(() => resetChartsDisplay(display));
    };

    function createEmptyChartContainer(chartContainer) {
        chartContainer.append($('<div>').addClass('empty-chart-container').text('No data for this configuration.'));
    }

    // this function should take the final data set and turn it into graphs
    // params: GraphData, unused, chartContainer
    function createChartWithNewDataByEngines(labels, graphConfigs, chartContainer, display) {
        var chartWrap = $('<div>');
        chartWrap.addClass('chart-wrap');
        chartContainer.append(chartWrap);
        var labelsContainer = $('<div>');
        labelsContainer.addClass('chart-labels-container');
        chartWrap.append(labelsContainer);

        var chartGraphsContainer = $('<div>');
        chartGraphsContainer.addClass('chart-graphs-container');
        chartWrap.append(chartGraphsContainer);

        graphConfigs.forEach((graphConfig, index) => {
            const id = getRandomNumber();
            if (graphConfig.unit === undefined) {
                graphConfig.unit = 'No unit.';
            }

            var graphItem = $(`<div id=${id}>`);
            graphItem.addClass('graph-item');
            var columnHeaderContainer = $('<div>');
            columnHeaderContainer.addClass('chart-column-title');
            var columnIcon = $('<div class="icon">');
            columnIcon.addClass(graphConfig.iconClass);
            columnHeaderContainer.append(columnIcon);
            var columnHeader = $('<div class="chart-header">');
            columnHeader.append($('<div class="title">' + graphConfig.chartTitle + '</div>'));
            columnHeaderContainer.append(columnHeader);
            chartGraphsContainer.append(graphItem);
            var graphClass = $('<div>');
            graphClass.addClass('graph-row');

            graphItem.append(columnHeaderContainer);
            graphItem.append(graphClass);
            processMetricByEngines(labels, graphConfig, graphClass, 'graph-row-column', id);
            window.setTimeout(() => {
                var labelsItem = $('<div>');
                setInitialItemsVisibility(labelsItem, index, display.mode);
            });
        });
        setChartsDisplayDirection(display.mode);
        adjustHeaderIcons(display.mode);
    }
    function createChartWithNewData(labels, graphConfigs, appConfig, chartContainer, display) {

        var chartWrap = $('<div>');
        chartWrap.addClass('chart-wrap');
        chartContainer.append(chartWrap);

        var labelsContainer = $('<div>');
        labelsContainer.addClass('chart-labels-container');
        chartWrap.append(labelsContainer);

        var chartGraphsContainer = $('<div>');
        chartGraphsContainer.addClass('chart-graphs-container');
        chartWrap.append(chartGraphsContainer);

        graphConfigs.forEach((graphConfig, index) => {
            const id = getRandomNumber();
            if (graphConfig.unit === undefined) {
                graphConfig.unit = 'No unit.';
            }

            var graphItem = $(`<div id=${id}>`);
            graphItem.addClass('graph-item');
            var columnHeaderContainer = $('<div>');
            columnHeaderContainer.addClass('chart-column-title');
            var columnIcon = $('<div class="icon">');
            columnIcon.addClass(graphConfig.iconClass);
            columnHeaderContainer.append(columnIcon);
            var columnHeader = $('<div class="chart-header">');
            columnHeader.append($('<div class="title">' + graphConfig.chartTitle + '</div>'));
            columnHeader.append($('<div class="subtitle">' + graphConfig.unit + ' ' + appConfig.UnitDescription[graphConfig.unit] +'</div>'));
            columnHeaderContainer.append(columnHeader);
            chartGraphsContainer.append(graphItem);
            var graphClass = $('<div>');
            graphClass.addClass('graph-row');

            graphItem.append(columnHeaderContainer);
            graphItem.append(graphClass);
            processMetric(labels, graphConfig.datasets, graphConfig.chartTitle, graphClass, 'graph-row-column', id);

            window.setTimeout(() => {
                const topPadding = getLabelsTopPadding(display.mode);
                const labelsHeight = (labels.length * 55);
                const chartHeight = $(graphItem).outerHeight();
                const bottomPadding = (chartHeight - (topPadding + labelsHeight));
                var labelsItem = $('<div>');
                labelsItem.addClass('chart-labels-item');
                labels.forEach((label) => {
                    labelsItem.append($('<div class="title">' + label + '</div>'));
                });

                labelsItem.css('padding-top', topPadding + 'px');
                labelsItem.css('padding-bottom', bottomPadding + 'px');
                setInitialItemsVisibility(labelsItem, index, display.mode);
                labelsContainer.append(labelsItem);
            });
        })
    }
    function setGraphConfigsByEngines(model, appConfig, engines, precisions) {
        var graphConfigs = model.map((platform) => {
            var kpiData = [];
            engines.forEach((engine) => Graph.getDatabyParameter(platform, engine, kpiData));
            var graphConfig = [];
            engines.forEach((engine) => {
                const engineGraphConfig = { engine: engine, config: Graph.getGraphConfig(engine, precisions, JSON.parse(JSON.stringify(appConfig))) };
                precisions.forEach((precision, index) => {
                    engineGraphConfig.config.datasets[index].data = kpiData[engine][0].find(el => el[precision])?.[precision];
                });
                graphConfig.push(engineGraphConfig);
                graphConfig.chartTitle = platform.Platform;
            });
            return graphConfig
        })
        return graphConfigs;
    }

    function setGraphConfigs(model, appConfig, parameters, precisions) {
        var graphConfigs = parameters.map((parameter) => {
            var groupUnit = model[0];
            var kpiData = Graph.getDatabyParameterOld(model, appConfig.ParametersMap[parameter], precisions);
            var config = Graph.getGraphConfigOld(appConfig.ParametersMap[parameter], groupUnit, precisions, JSON.parse(JSON.stringify(appConfig)));
            precisions.forEach((precision, index) => {
                config.datasets[index].data = kpiData.map(tData => tData[0][precision]);
            });
            return config;
        });
        return graphConfigs
    }
    function processMetric(labels, datasets, chartTitle, container, widthClass, id) {
        var heightRatio = (30 + (labels.length * 55));
        var chart = $('<div>');
        const containerId = `legend-container-${id}`;
        const legend = $(`<div id="${containerId}">`);
        legend.addClass('graph-legend-container');
        chart.addClass('chart');
        chart.addClass(widthClass);
        chart.height(heightRatio);
        var canvas = $('<canvas>');
        chart.append(canvas);
        container.append(chart);
        container.append(legend);
        var context = canvas.get(0).getContext('2d');
        context.canvas.height = heightRatio;
        window.setTimeout(() => {
            new Chart(context, {
                type: 'bar',
                data: getChartData(labels, datasets),
                options: getChartOptions(chartTitle),
                plugins: [htmlLegendPlugin]
            });
        });
    }

    function getChartData(labels, datasets) {
        return {
            labels: labels,
            datasets: datasets.map((item) => {
                return {
                    label: item.label,
                    data: item.data,
                    backgroundColor: item.color,
                    borderColor: 'rgba(170,170,170,0)',
                    barThickness: 10
                }
            })
        }
    }
    var charts = [];
    function processMetricByEngines(labels, datasets, container, widthClass, id) {
        var heightRatio = (30 + (labels.length * 55));
        var chart = $('<div>');
        const containerId = `legend-container-${id}`;
        const legend = $(`<div id="${containerId}">`);
        legend.addClass('graph-legend-container');
        chart.addClass('chart');
        chart.addClass(widthClass);
        chart.height(heightRatio);
        var canvas = $('<canvas>');
        chart.append(canvas);
        container.append(chart);
        container.append(legend);
        var context = canvas.get(0).getContext('2d');
        context.canvas.height = heightRatio;
        window.setTimeout(() => {
            var labels = [];
            for (const key in datasets[0].config.datasets[0].data) {
                labels.push(key);
            }

            var graphDatas = [];
            datasets.forEach((engineSet) => {
                engineSet.config.datasets.forEach((precision) => {
                    var precData = [];
                    for (const key in precision.data) {
                        precData.push(precision.data[key]);
                    }
                    graphDatas.push({
                        label: engineSet.engine + ' ' + precision.label,
                        data: precData,
                        borderColor: precision.color,
                        backgroundColor: precision.color,
                        yAxisID: precision.label === "Throughput" ? 'y' : 'y1',
                        fill: false
                    })
                })
            })

            let allowedAxisIDs = ['x'];
            const newAxisIDs = [...new Set(graphDatas.map(item => item.yAxisID))];

            newAxisIDs.forEach(id => {
                if (!allowedAxisIDs.includes(id)) {
                    allowedAxisIDs.push(id);
                }
            });

            new Chart(context, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: graphDatas
                },
                options: getChartOptionsByEngines(allowedAxisIDs),
                plugins: [htmlLegendPlugin]
            });

        });
    }

    function getRandomNumber() {
        return Math.floor(Math.random() * 100000);
    }

    function resetChartsDisplay(currentDisplay) {
        const newDisplayMode = getChartsDisplayMode(currentDisplay.numberOfChartsInRow);
        if (currentDisplay.mode != newDisplayMode) {
            currentDisplay.mode = newDisplayMode;
            setChartsDisplayDirection(currentDisplay.mode);
            adjustLabels(currentDisplay.mode);
            adjustHeaderIcons(currentDisplay.mode);
        }
    }

    function adjustLabels(displayMode) {
        const firstLabels = $('.chart-labels-container').find('.chart-labels-item:first-child');
        const labels = $('.chart-labels-container').find('.chart-labels-item');
        labels.css('padding-top', getLabelsTopPadding(displayMode));
        if (displayMode == 'column') {
            labels.show();
        }
        else {
            labels.hide()
            firstLabels.show();
        }
    }

    function adjustHeaderIcons(displayMode) {
        const icons = $('.graph-item').find('.chart-column-title');
        if (displayMode == 'rowCompact')
            icons.css('flex-direction', 'column')
        else
            icons.css('flex-direction', 'row')
    }

    function getLabelsTopPadding(displayMode) {
        return (displayMode == 'rowCompact') ? 105.91 : 83.912;
    }

    function setChartsDisplayDirection(displayMode) {
        const container = $('.chart-placeholder').find('.chart-graphs-container');
        if (displayMode == 'column') {
            container.css('flex-direction', 'column');
        }
        else {
            container.css('flex-direction', 'row');
        }
    }

    function setInitialItemsVisibility(item, count, displayMode) {
        if (count == 0 || displayMode == 'column') item.show();
        else item.hide();
    }

    function getChartsDisplayMode(numberOfCharts) {
        switch (numberOfCharts) {
            case 4:
                return window.matchMedia('(max-width: 721px)').matches ? 'column'
                    : window.matchMedia('(max-width: 830px)').matches ? 'rowCompact'
                        : 'row';
            case 3:
                return window.matchMedia('(max-width: 569px)').matches ? 'column'
                    : window.matchMedia('(max-width: 649px)').matches ? 'rowCompact'
                        : 'row';
            case 2:
                return window.matchMedia('(max-width: 500px)').matches ? 'column'
                    : 'row';
            default:
                return 'row';
        }
    }
});