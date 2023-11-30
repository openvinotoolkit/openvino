// =================== GENERAL OUTPUT CONFIG =========================

const chartDisclaimers = {
    Value: 'Value: Performance/(No_of_sockets * Price_of_CPU_dGPU), where prices are in USD as of November 2023.',
    Efficiency: 'Efficiency: Performance/(No_of_sockets * TDP_of_CPU_dGPU), where total power dissipation (TDP) is in Watt as of November 2023.'
}

const OVdefaultSelections = {
    platformTypes: {name: 'ietype', data: ['core']},
    platforms: {name: 'platform',
        data: [
            'Intel® Core™ i5-10500TE ',
            'Intel® Core™ i7-1185G7 CPU',
            'Intel® Core™ i9-10900TE ',
        ]
    },
    platformFilters: {name: 'coretype', data: ['CPU']},
    models: {name: 'networkmodel',
        data: [
            'bert-base-cased',
            'yolo_v3_tiny',
            'yolo_v8n',
            'resnet-50',
        ]
    },
    parameters: {name: 'kpi', data: ['Throughput']},
    pracision: {name: 'precision', data: ['INT8', 'FP32']}
}

const OVMSdefaultSelections = {
    platforms: {name: 'platform',
        data: [
            'Intel®  Xeon® 8260M CPU-only',
            'Intel®  Xeon® Gold 6238M CPU-only'
        ]
    },
    models: {name: 'networkmodel',
        data: [
            'bert-small-uncased-whole-word-masking-squad-0002',
            'mobilenet-ssd ',
            'resnet-50',
            'yolo_v3_tiny'
        ]
    },
    parameters: {name: 'kpi', data: ['Throughput']},
    pracision: {name: 'precision', data: ['OV-INT8 (reference)', 'INT8']}
}

// ====================================================


class Filter {

    // param: GraphData[], networkModels[]
    static FilterByNetworkModel(graphDataArr, networkModels) {
        // This is a bit obtuse, collect all options from all models
        // Some of them might return dupes, so convert them to a map, and get unique objects based on names
        const optionMap = new Map();
        networkModels.map((model) => graphDataArr.filter((graphData => graphData.networkModel === model)))
          .flat(1)
          .forEach(item => optionMap.set(item.platformName, item));
        // convert the option map back to an array with just the values
        return Array.from(optionMap.values());
    }

    // param: GraphData[], ieType
    static FilterByIeType(graphDataArr, value) {
        return graphDataArr.filter((data) => data.ieType.includes(value));
    }

    // param: GraphData[], clientPlatforms[]
    static FilterByClientPlatforms(graphDataArr, platformsArr) {
        return graphDataArr.filter((data) => platformsArr.includes(data.platformName));
    }

    // param: GraphData[], coreTypes[]
    static FilterByCoreTypes(graphDataArr, coreTypes) {
        if (coreTypes) {
            return graphDataArr.filter((data) => coreTypes.includes(data.ieType));
        }
        return graphDataArr;
    }

    // param: GraphData[] (of one networkModel), key (throughput, latency, efficiency, value)
    static getKpiData(graphDataArr, key) {
        return graphDataArr.map((data) => {
            return data[key];
        });
    }
}


class ExcelDataTransformer {

    static transform(csvdata, version) {
        const entries = csvdata.filter((entry) => {
            return !entry.includes('begin_rec') && !entry.includes('end_rec');
        });
        // do other purging and data massaging here

        // else generate
        return entries.map((entry) => {
            if (version == 'ovms')
                return new GraphData(new OVMSExcelData(entry));
            return new GraphData(new ExcelData(entry));
        });
    }
}


class ExcelData {
    constructor(csvdataline) {
        if (!csvdataline) {
            return;
        }
        this.networkModel = csvdataline[0].toLowerCase();
        this.release = csvdataline[1];
        this.ieType = csvdataline[2];
        this.platformName = csvdataline[3];
        this.throughputInt4 = csvdataline[22];
        this.throughputInt8 = csvdataline[4];
        this.throughputFP16 = csvdataline[5];
        this.throughputFP32 = csvdataline[6];
        this.value = csvdataline[7];
        this.efficiency = csvdataline[8];
        this.price = csvdataline[9];
        this.tdp = csvdataline[10];
        this.sockets = csvdataline[11];
        this.pricePerSocket = csvdataline[12];
        this.tdpPerSocket = csvdataline[13];
        this.latency = csvdataline[14];
        this.latency16 = csvdataline[19];
        this.latency32 = csvdataline[20];
        this.latency4 = csvdataline[21];
        this.throughputUnit = csvdataline[15];
        this.valueUnit = csvdataline[16];
        this.efficiencyUnit = csvdataline[17];
        this.latencyUnit = csvdataline[18];
    }
}


class OVMSExcelData extends ExcelData {
    constructor(csvdataline) {
        super(csvdataline);
        this.throughputOVMSInt8 = csvdataline[5];
        this.throughputInt8 = csvdataline[4];
        this.throughputOVMSFP32 = csvdataline[7];
        this.throughputFP32 = csvdataline[6];
        this.throughputUnit = csvdataline[8]
    }
}


class GraphData {
    constructor(excelData) {
        if (!excelData) {
            return;
        }
        this.networkModel = excelData.networkModel;
        this.release = excelData.release;
        this.ieType = excelData.ieType;
        this.platformName = excelData.platformName;
        this.kpi = new KPI(
            {
                'ovmsint8': excelData.throughputOVMSInt8,
                'ovmsfp32': excelData.throughputOVMSFP32,
                'int4': excelData.throughputInt4,
                'int8': excelData.throughputInt8,
                'fp16': excelData.throughputFP16,
                'fp32': excelData.throughputFP32
            },
            excelData.value,
            excelData.efficiency,
            {
                'ovmsint8': excelData.throughputOVMSInt8,
                'ovmsfp32': excelData.throughputOVMSFP32,
                'int4': excelData.latency4,
                'int8': excelData.latency,
                'fp16': excelData.latency16,
                'fp32': excelData.latency32
            },);
        
        this.price = excelData.price;
        this.tdp = excelData.tdp;
        this.sockets = excelData.sockets;
        this.pricePerSocket = excelData.pricePerSocket;
        this.tdpPerSocket = excelData.tdpPerSocket;
        this.latency = excelData.latency;
        this.throughputUnit = excelData.throughputUnit;
        this.valueUnit = excelData.valueUnit;
        this.efficiencyUnit = excelData.efficiencyUnit;
        this.latencyUnit = excelData.latencyUnit;
    }
}


class KPI {
    constructor(precisions, value, efficiency, latencies) {
        this.throughput = precisions;
        this.value = value;
        this.efficiency = efficiency;
        this.latency = latencies;
    }
}


class Modal {
    static getIeTypeLabel(ietype) {
        switch (ietype) {
            case 'core':
                return 'Client Platforms (Intel® Core™)';
            case 'xeon':
                return 'Server Platforms (Intel® Xeon®)';
            case 'atom':
                return 'Mobile Platforms (Intel® Atom™)';
            case 'accel':
                return 'Accelerator Platforms';
            default:
                return '';
        }
    }
    static getCoreTypesLabels() {
        return ['CPU', 'iGPU', 'CPU+iGPU'];
    }
    static getKpisLabels(version) {
        if (version == 'ovms')
            return ['Throughput'];
        return ['Throughput', 'Latency', 'Value', 'Efficiency'];
    }
    static getPrecisionsLabels(version) {
        if (version == 'ovms')
            return ['OV-INT8 (reference)', 'INT8', 'OV-FP32 (reference)', 'FP32'];
        return ['INT4', 'INT8', 'FP16', 'FP32'];
    }
    static getCoreTypes(labels) {
        return labels.map((label) => {
            switch (label) {
                case 'CPU':
                    return 'core';
                case 'iGPU':
                    return 'core-iGPU';
                case 'CPU+iGPU':
                    return 'core-CPU+iGPU';
                default:
                    return '';
            }
        });
    }
    static getPrecisions(labels) {
        return labels.map((label) => {
            switch (label) {
                case 'OV-INT8 (reference)':
                    return 'ovmsint8';
                case 'OV-FP32 (reference)':
                    return 'ovmsfp32';
                case 'INT4':
                    return 'int4';
                case 'INT8':
                    return 'int8';
                case 'FP16':
                    return 'fp16';
                case 'FP32':
                    return 'fp32';
                default:
                    return '';
            }
        });
    }
    static getUnitDescription(unit) {
            console.log(unit)
            switch (unit) {
                case 'msec.':
                    return '(lower is better)';
                case 'msec/token':
                    return '(lower is better)';
                case 'Generating time, sec.':
                    return '(lower is better)';
                case 'msec/token/TDP':
                    return '(lower is better)';
                case 'FPS':
                    return '(higher is better)';
                case 'FPS/$':
                    return '(higher is better)';
                case 'FPS/TDP':
                    return '(higher is better)';
                default:
                    return '';
            }
    }
}


class Graph {
    constructor(data) {
        this.data = data;
    }
    data = new GraphData();

    // functions to get unique keys 
    static getNetworkModels(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.networkModel)));
    }
    static getIeTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.ieType)));
    }
    static getPlatforms(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.platformName)));
    }
    static getCoreTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.ieType)));
    }

    // param: GraphData[]
    static getPlatformNames(graphDataArr) {
        return graphDataArr.map((data) => data.platformName);
    }

    // param: GraphData[], kpi: string
    static getDatabyKPI(graphDataArr, kpi) {
        switch (kpi) {
            case 'throughput':
                return graphDataArr.map((data) => data.kpi.throughput);
            case 'latency':
                return graphDataArr.map((data) => data.kpi.latency);
            case 'efficiency':
                return graphDataArr.map((data) => data.kpi.efficiency);
            case 'value':
                return graphDataArr.map((data) => data.kpi.value);
            default:
                return [];
        }
    }
    

    // this returns an object that is used to ender the chart
    static getGraphConfig(kpi, units, precisions) {
        switch (kpi) {
            case 'throughput':
                return {
                    chartTitle: 'Throughput',
                    iconClass: 'throughput-icon',
                    unit: units.throughputUnit,
                    datasets: precisions.map((precision) => this.getPrecisionThroughputConfig(precision, units.throughputUnit)),
                };
            case 'latency':
                return {
                    chartTitle: 'Latency',
                    iconClass: 'latency-icon',
                    unit: units.latencyUnit,
                    datasets: precisions.map((precision) => this.getPrecisionLatencyConfig(precision, units.latencyUnit)),
                };
            case 'value':
                return {
                    chartTitle: 'Value',
                    iconClass: 'value-icon',
                    unit: units.valueUnit,
                    datasets: [{ data: null, color: '#8BAE46', label: `INT8` }],
                };
            case 'efficiency':
                return {
                    chartTitle: 'Efficiency',
                    iconClass: 'efficiency-icon',
                    unit: units.efficiencyUnit,
                    datasets: [{ data: null, color: '#E96115', label: `INT8` }],
                };
            default:
                return {};
        }
    }

    static getPrecisionThroughputConfig(precision, unit) {
        switch (precision) {
            case 'ovmsint8':
                return { data: null, color: '#FF8F51', label: `${unit} (OV Ref. INT8)` };
            case 'ovmsfp32':
                return { data: null, color: '#B24501', label: `${unit} (OV Ref. FP32)` };
            case 'int4':
                return { data: null, color: '#5bd0f0', label: `INT4` };
            case 'int8':
                return { data: null, color: '#00C7FD', label: `INT8` };
            case 'fp16':
                return { data: null, color: '#009fca', label: `FP16` };
            case 'fp32':
                return { data: null, color: '#007797', label: `FP32` };
            default:
                return {};
        }
    }

    static getPrecisionLatencyConfig(precision, unit) {
        switch (precision) {
            case 'ovmsint8':
                return { data: null, color: '#FF8F51', label: `${unit} (OV Ref. INT8)` };
            case 'ovmsfp32':
                return { data: null, color: '#B24501', label: `${unit} (OV Ref. FP32)` };
            case 'int4':
                return { data: null, color: '#c197d1', label: `INT4` };
            case 'int8':
                return { data: null, color: '#b274ca', label: `INT8` };
            case 'fp16':
                return { data: null, color: '#8424a9', label: `FP16` };
            case 'fp32':
                return { data: null, color: '#5b037d', label: `FP32` };
            default:
                return {};
        }
    }

    static getGraphPlatformText(platform) {
        switch (platform) {
            case 'atom':
                return 'Mobile Platforms';
            case 'core':
                return 'Client Platforms';
            case 'xeon':
                return 'Server Platforms';
            case 'accel':
                return 'Accelerated Platforms';
            default:
                return '';
        }
    }
}


class ChartDisplay {
    constructor(mode, numberOfCharts) {
        this.mode = mode;
        this.numberOfChartsInRow = numberOfCharts;
    }
}


$(document).ready(function () {

    $('.ov-toolkit-benchmark-results').on('click', () => showModal('ov'));
    $('.ovms-toolkit-benchmark-results').on('click', () => showModal('ovms'));

    function clickBuildGraphs(graph, networkModels, ietype, platforms, kpis, precisions) {
        renderData(graph, networkModels, ietype, platforms, kpis, precisions);
        $('.modal-footer').show();
        $('#modal-display-graphs').show();
        $('.edit-settings-btn').on('click', (event) => {
            $('#modal-configure-graphs').show();
            $('#modal-display-graphs').hide();
            $('.modal-footer').hide();
            $('.chart-placeholder').empty();
        });

        $('.graph-chart-title-header').on('click', (event) => {
            var parent = event.target.parentElement;

            if ($(parent).children('.chart-wrap,.empty-chart-container').is(":visible")) {
                $(parent).children('.chart-wrap,.empty-chart-container').hide();
                $(parent).children('.chevron-right-btn').show();
                $(parent).children('.chevron-down-btn').hide();
                $
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
    
    function showModal(version) {
        $('body').css('overflow', 'hidden');

        let dataPath = '_static/benchmarks_files/OV-benchmark-data.csv';
        if (version == 'ovms')
            dataPath = '_static/benchmarks_files/OVMS-benchmark-data.csv';
        Papa.parse(dataPath, {
            download: true,
            complete: (result) => renderModal(result, version)
        });
    }

    function getSelectedNetworkModels() {
        return $('.models-column-one input:checked, .models-column-two input:checked').not('[data-networkmodel="Select All"]').map(function () {
            return $(this).data('networkmodel');
        }).get();
    }
    function getSelectedIeType() {
        return $('.ietype-column input:checked').map(function () {
            return $(this).data('ietype');
        }).get().pop();
    }
    function getSelectedCoreTypes() {
        return $('.client-platform-column .selected').map(function () {
            return $(this).data('coretype');
        }).get();
    }
    function getSelectedClientPlatforms() {
        return $('.client-platform-column input:checked').map(function () {
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
        && getSelectedIeType()
        && getSelectedClientPlatforms().length > 0
        && getSelectedKpis().length > 0) {
            if (getSelectedKpis().includes('Throughput')) {
                if (getSelectedPrecisions().length > 0) {
                    $('#build-graphs-btn').prop('disabled', false);
                    return;
                }
                $('#build-graphs-btn').prop('disabled', true);
                return;
            }
            $('#build-graphs-btn').prop('disabled', false);
            return;
        }
        $('#build-graphs-btn').prop('disabled', true);
    }

    function renderModal(result, version) {
        // remove header from csv line
        result.data.shift();
        var graph = new Graph(ExcelDataTransformer.transform(result.data, version));

        var networkModels = Graph.getNetworkModels(graph.data);
        var ieTypes = Graph.getIeTypes(graph.data);

        fetch('_static/html/modal.html').then((response) => response.text()).then((text) => {

            // generate and configure modal container
            var modal = $('<div>');
            modal.attr('id', 'graphModal');
            modal.addClass('modal');
            // generate and configure modal content from html import
            var modalContent = $(text);
            modalContent.attr('id', 'graphModalContent');
            modalContent.addClass('modal-content');
            modal.append(modalContent);

            const models = networkModels.map((networkModel) => createCheckMark(networkModel, 'networkmodel'));
            const selectAllModelsButton = createCheckMark('Select All', 'networkmodel')
            modal.find('.models-column-one').append(selectAllModelsButton).append(models.slice(0, models.length / 2));
            modal.find('.models-column-two').append(models.slice(models.length / 2));

            const precisions = Modal.getPrecisionsLabels(version).map((precision) => createCheckMark(precision, 'precision'));
            modal.find('.precisions-column').append(precisions);
            selectAllCheckboxes(precisions);
            disableAllCheckboxes(precisions);

            const types = ieTypes.map((ieType) => {
                var labelText = Modal.getIeTypeLabel(ieType);
                if (labelText) {
                    const item = $('<label class="checkmark-container">');
                    const checkboxSpan = $('<span class="checkmark radiobutton">');
                    item.text(Modal.getIeTypeLabel(ieType));
                    const radio = $('<input type="radio" name="ietype"/>');
                    item.append(radio);
                    item.append(checkboxSpan);
                    radio.attr('data-ietype', ieType);
                    return item;
                }
            });
            modal.find('#modal-display-graphs').hide();

            modal.find('.ietype-column').append(types);
            modal.find('.ietype-column input').first().prop('checked', true);

            const kpiLabels = Modal.getKpisLabels(version).map((kpi) => createCheckMark(kpi, 'kpi'));
            modal.find('.kpi-column').append(kpiLabels);

            $('body').prepend(modal);

            renderClientPlatforms(graph.data, modal, version, true);
            preselectDefaultSettings(graph.data, modal, version);

            $('.clear-all-btn').on('click', clearAll);
            $('#build-graphs-btn').on('click', () => {
                $('#modal-configure-graphs').hide();
                clickBuildGraphs(graph, getSelectedNetworkModels(), getSelectedIeType(), getSelectedClientPlatforms(), getSelectedKpis(), Modal.getPrecisions(getSelectedPrecisions()));
            });
            $('.modal-close').on('click', hideModal);
            $('.close-btn').on('click', hideModal);
            modal.find('.models-column-one input[data-networkmodel="Select All"]').on('click', function() {
                if ($(this).prop('checked'))
                    selectAllCheckboxes(models);
                else deSelectAllCheckboxes(models);
            });
            modal.find('.ietype-column input').on('click', () => renderClientPlatforms(graph.data, modal, version, true));
            modal.find('.kpi-column input').on('click', validateThroughputSelection);
            modal.find('input').on('click', validateSelections);
        });
    }
    
    function validateThroughputSelection() {
        const precisions = $('.precisions-column').find('input')
        if (getSelectedKpis().includes('Throughput') || getSelectedKpis().includes('Latency')) {
            precisions.prop('disabled', false);
        }
        else {
            precisions.prop('disabled', true);
        }
    }

    function clearAll() {
        $('.modal-content-grid-container input:checkbox').each((index, object) => $(object).prop('checked', false));
        // Uncomment if you want the Clear All button to reset the Platform Type column as well
        // modal.find('.ietype-column input').first().prop('checked', true);
        validateThroughputSelection();
        validateSelections();
    }

    function preselectDefaultSettings(data, modal, version) {
        const defaultSelections = (version == 'ov') ? OVdefaultSelections : OVMSdefaultSelections;
        if (defaultSelections.platformTypes) {
            const type = defaultSelections.platformTypes.data[0]
            $(`input[data-ietype="${type}"]`).prop('checked', true);
            renderClientPlatforms(data, modal, version);
        }
        if (defaultSelections.platformFilters) {
            const filters = modal.find('.selectable-box-container').children('.selectable-box');
            filters.removeClass('selected');
            defaultSelections.platformFilters.data.forEach(selection => {
                filters.filter(`[data-${defaultSelections.platformFilters.name}="${selection}"]`).addClass('selected');
            });
            renderClientPlatforms(data, modal, version);
        }
        clearAll();
        for (setting in defaultSelections) {
            let name = defaultSelections[setting].name;
            defaultSelections[setting].data.forEach(selection => {
                $(`input[data-${name}="${selection}"]`).prop('checked', true);
            });
        }
        validateThroughputSelection();
        validateSelections();
    }

    function showCoreSelectorTypes(coreTypes, graphDataArr,  modal) {
        if ($('.client-platform-column').find('.selectable-box-container').length) {
            $('.client-platform-column').find('.selectable-box-container').show();
            return;
        }
        var container = $('<div>');
        container.addClass('selectable-box-container');
        coreTypes.forEach((type) => {
            var box = $('<div>' + type + '</div>');
            box.attr('data-coretype', type);
            box.addClass('selectable-box selected');
            container.append(box);
        });
        $('.client-platform-column').prepend(container);
        $('.client-platform-column .selectable-box').on('click', function () {
            if ($(this).hasClass('selected')) {
                $(this).removeClass('selected');
            } else {
                $(this).addClass('selected');
            }
            var fPlatforms = filterClientPlatforms(graphDataArr, getSelectedNetworkModels(), getSelectedIeType(), Modal.getCoreTypes(getSelectedCoreTypes()));
            renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms), true);
            validateSelections();
        });
    }

    function hideCoreSelectorTypes() {
        $('.client-platform-column').find('.selectable-box-container').hide();
    }

    function filterClientPlatforms(data, networkModels, ietype, coreTypes) {
        // No longer filtering on the network type, if at some point we want the network type as a filter, uncomment this
        // var first = Filter.FilterByNetworkModel(data, networkModels);
        var second = Filter.FilterByIeType(data, ietype);
        if (ietype === 'core') {
          second = Filter.FilterByCoreTypes(second, coreTypes);
        }
        const optionMap = new Map();
        second.forEach(item => optionMap.set(item.platformName, item));
        return Array.from(optionMap.values());
    }

    function renderClientPlatforms(data, modal, version, preselectEveryItem) {
        if (getSelectedIeType() === 'core') {
            showCoreSelectorTypes(Modal.getCoreTypesLabels(), data, modal);
            if (version === 'ovms')
                hideCoreSelectorTypes();
        }
        else {
            hideCoreSelectorTypes();
        }
        var fPlatforms = filterClientPlatforms(data, getSelectedNetworkModels(), getSelectedIeType(), Modal.getCoreTypes(getSelectedCoreTypes()));
        renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms), preselectEveryItem);
    }

    function renderClientPlatformsItems(modal, platformNames, preselectEveryItem) {
        $('.client-platform-column .checkmark-container').remove();
        const clientPlatforms = platformNames.map((platform) => createCheckMark(platform, 'platform'));
        if (preselectEveryItem)
            selectAllCheckboxes(clientPlatforms);
        modal.find('.client-platform-column').append(clientPlatforms);
        modal.find('.client-platform-column input').on('click', validateSelections);
    }

    function createCheckMark(itemLabel, modelLabel) {
        const item = $('<label class="checkmark-container">');
        item.text(itemLabel);
        const checkbox = $('<input type="checkbox"/>');
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

    function enableAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('disabled', false);
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

        legendContainer.appendChild(listContainer);
      }

      return listContainer;
    };

    const htmlLegendPlugin = {
      id: 'htmlLegend',
      afterUpdate(chart, args, options) {

        const ul = getOrCreateLegendList(chart, chart.options.plugins.htmlLegend.containerID);
        
        // Remove old legend items
        while (ul.firstChild) {
          ul.firstChild.remove();
        }

        // Reuse the built-in legendItems generator
        const items = chart.legend.legendItems;
        items.forEach(item => {
          const li = document.createElement('li');
          li.style.alignItems = 'center';
          li.style.display = 'block';
          li.style.flexDirection = 'column';
          li.style.marginLeft = '10px';

          li.onclick = () => {
            const {type} = chart.config;
            if (type === 'pie' || type === 'doughnut') {
              // Pie and doughnut charts only have a single dataset and visibility is per item
              chart.toggleDataVisibility(item.index);
            } else {
              chart.setDatasetVisibility(item.datasetIndex, !chart.isDatasetVisible(item.datasetIndex));
            }
            chart.update();
          };

          // Color box
          const boxSpan = document.createElement('span');
          boxSpan.style.background = item.fillStyle;
          boxSpan.style.borderColor = item.strokeStyle;
          boxSpan.style.borderWidth = item.lineWidth + 'px';
          boxSpan.style.display = 'inline-block';
          boxSpan.style.height = '12px';
          boxSpan.style.marginRight = '4px';
          boxSpan.style.width = '30px';

          // Text
          const textContainer = document.createElement('p');
          textContainer.style.color = item.fontColor;
          textContainer.style.margin = 0;
          textContainer.style.padding = 0;
          textContainer.style.fontSize = '0.8rem';
          textContainer.style.textDecoration = item.hidden ? 'line-through' : '';

          const text = document.createTextNode(item.text);
          textContainer.appendChild(text);

          li.appendChild(boxSpan);
          li.appendChild(textContainer);
          ul.appendChild(li);
        });
      }
    };

    // ====================================================

    function getChartOptions(title, containerId) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            legend: {display: false},
            title: {
                display: false,
                text: title
            },
            scales: {
                xAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }],
                yAxes: [{
                    ticks: {
                        display: false, //this will remove only the label
                        beginAtZero: true
                    }
                }]
            },
            plugins: {
                htmlLegend: {
                // ID of the container to put the legend in
                    containerID: containerId,
                }
            }
        }
    }

    // params: string[], Datasets[]
    function getChartDataNew(labels, datasets) {
        return {
            labels: labels,
            datasets: datasets.map((item) => {
                return {
                    label: item.label,
                    data: item.data,
                    backgroundColor: item.color,
                    borderColor: 'rgba(170,170,170,0)',
                    barThickness: 12
                }
            })
        }
    }

    function renderData(graph, networkModels, ietype, platforms, kpis, precisions) {

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

            var filteredNetworkModels = Filter.FilterByNetworkModel(graph.data, [networkModel]);
            var filteredIeTypes = Filter.FilterByIeType(filteredNetworkModels, ietype);
            var filteredGraphData = Filter.FilterByClientPlatforms(filteredIeTypes, platforms);

            $('.chart-placeholder').append(chartContainer);
            if (filteredGraphData.length > 0) {
                createChartWithNewData(filteredGraphData, chartContainer, kpis, ietype, precisions, display);
            } else {
              createEmptyChartContainer(chartContainer);
            }
        })
       
        if(kpis.includes('Value') || kpis.includes('Efficiency')){
            $('.modal-footer').append($('<div class="modal-line-divider"></div>'))
        }
        $('.modal-footer').append($('<div class="modal-footer-content"><div class="modal-disclaimer-box"></div></div>'))
        for (let kpi of kpis) {
            if (chartDisclaimers[kpi])
                $('.modal-disclaimer-box').append($('<p>').text(chartDisclaimers[kpi]))
        }

        $(window).off('resize');
        $(window).resize(() => resetChartsDisplay(display));
    };

    function createEmptyChartContainer(chartContainer) {
      chartContainer.append($('<div>').addClass('empty-chart-container').text('No data for this configuration.'));
    }

    // this function should take the final data set and turn it into graphs
    // params: GraphData, unused, chartContainer
    function createChartWithNewData(model, chartContainer, kpis, ietype, precisions, display) {
        var chartWrap = $('<div>');
        chartWrap.addClass('chart-wrap');
        chartContainer.append(chartWrap);
        var labels = Graph.getPlatformNames(model);
        var graphConfigs = kpis.map((str) => {
            var kpi = str.toLowerCase();
            var groupUnit = model[0];
            if (kpi === 'throughput') {
                var throughputData = Graph.getDatabyKPI(model, kpi);
                var config = Graph.getGraphConfig(kpi, groupUnit, precisions);
                precisions.forEach((prec, index) => {
                    config.datasets[index].data = throughputData.map(tData => tData[prec]);
                });
                return config;
            }
            else if(kpi === 'latency'){
                var latencyData = Graph.getDatabyKPI(model, kpi);
                var config = Graph.getGraphConfig(kpi, groupUnit, precisions);
                precisions.forEach((prec, index) => {
                    config.datasets[index].data = latencyData.map(tData => tData[prec]);
                });
                return config;
            }
            var config = Graph.getGraphConfig(kpi, groupUnit);
            config.datasets[0].data = Graph.getDatabyKPI(model, kpi);
            return config;
        });
        // get the client platform labels and create labels for all the graphs
        var labelsContainer = $('<div>');
        labelsContainer.addClass('chart-labels-container');
        chartWrap.append(labelsContainer);

        // get the kpi title's and create headers for the graphs 
        var chartGraphsContainer = $('<div>');
        chartGraphsContainer.addClass('chart-graphs-container');
        chartWrap.append(chartGraphsContainer);

        graphConfigs.forEach((graphConfig, index) => {
            const id = getRandomNumber();
            var graphItem = $(`<div id=${id}>`);
            graphItem.addClass('graph-item');
            var columnHeaderContainer = $('<div>');
            columnHeaderContainer.addClass('chart-column-title');
            var columnIcon = $('<div class="icon">');
            columnIcon.addClass(graphConfig.iconClass);
            columnHeaderContainer.append(columnIcon);
            var columnHeader = $('<div class="chart-header">');
            columnHeader.append($('<div class="title">' + graphConfig.chartTitle + '</div>'));
            columnHeader.append($('<div class="title">' + Graph.getGraphPlatformText(ietype) + '</div>'));
            columnHeader.append($('<div class="subtitle">' + graphConfig.unit + ' ' + Modal.getUnitDescription(graphConfig.unit) + '</div>'));

            columnHeaderContainer.append(columnHeader);
            chartGraphsContainer.append(graphItem);
            var graphClass = $('<div>');
            graphClass.addClass('graph-row');
            
            graphItem.append(columnHeaderContainer);
            graphItem.append(graphClass);
            processMetricNew(labels, graphConfig.datasets, graphConfig.chartTitle, graphClass, 'graph-row-column', id);
            
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
        });
        setChartsDisplayDirection(display.mode);
        adjustHeaderIcons(display.mode);
    }

    function processMetricNew(labels, datasets, chartTitle, container, widthClass, id) {
        // ratio for consistent chart label height
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
            type: 'horizontalBar',
            data: getChartDataNew(labels, datasets),
            options: getChartOptions(chartTitle, containerId),
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
