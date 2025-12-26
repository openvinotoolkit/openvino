from jinja2 import select_autoescape, Template
from conan.api.output import cli_out_write

build_order_html = r"""
<html lang="en">
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.0/cytoscape.min.js"
        integrity="sha512-zHc90yHSbkgx0bvVpDK/nVgxANlE+yKN/jKy91tZ4P/vId8AL7HyjSpZqHmEujWDWNwxYXcfaLdYWjAULl35MQ=="
        crossorigin="anonymous"
        referrerpolicy="no-referrer"></script>
    </head>

    <body>
        <style>
            body {
                font: 14px helvetica neue, helvetica, arial, sans-serif;
                display: flex;
                margin: 0;
                padding: 0;
                height: 100vh;
            }

            .sidebar {
                background: #f9f9f9;
                border-right: 1px solid #ccc;
                padding: 20px;
                box-sizing: border-box;
                overflow-y: auto;
                font-size: 14px;
                width: 440px;
            }

            .content {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            #cy {
                flex-grow: 1;
            }

            #node-info {
                margin-top: 20px;
                background: #f9f9f9;
                padding: 12px;
                font-family: monospace;
                white-space: pre-wrap;
                font-size: 12px;
                word-wrap: break-word;
            }

            .legend {
                margin-top: 20px;
                font-size: 14px;
            }

            .legend-item {
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }

            .legend-color {
                width: 20px;
                height: 20px;
                margin-right: 10px;
            }
        </style>

        <div class="sidebar">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ffff37;"></div>
                    <span>All packages need to be built</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff9b28;"></div>
                    <span>Some packages need to be built</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #70c7e6;"></div>
                    <span>Cache</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #79eb8a;"></div>
                    <span>Download</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="border: 1px solid black; width: 20px; height: 20px;"></div>
                    <span>Requirements in the <i>host</i> context</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="border: 1px solid black; border-radius: 50%; width: 20px; height: 20px;"></div>
                    <span>Requirements in the <i>build</i> context</span>
                </div>
            </div>
            <div id="node-info">
                <p>Click on a node to see details.</p>
            </div>
        </div>
        <div class="content">
            <div id="cy"></div>
        </div>

        <script type="text/javascript">
            document.addEventListener("DOMContentLoaded", function() {
                var buildOrderData = {{ build_order | tojson }};

                var elements = [];
                var edges = [];

                var yOffset = 50;
                var posX = 0;
                var posY = 0;
                var columns = 4;

                buildOrderData.forEach((step, stepIndex) => {
                    var stepId = 'step' + stepIndex;
                    elements.push({
                        data: { id: stepId, label: 'Step ' + (stepIndex + 1) },
                        position: { x: posX, y: posY }
                    });

                    step.forEach((lib, libIndex) => {
                        var libId = stepId + '_lib' + libIndex;
                        var libLabel = lib.ref.split('#')[0];

                        // Determine the type of the library
                        var libData = Array.isArray(lib.packages) ? lib.packages[0][0] : lib;
                        var libType = libData.binary || libData.build || libData.cache;

                        var isAllBuild = true;
                        var isSomeBuild = false;
                        var shape = 'rectangle';
                        var borderColor = '#00695C';

                        if (Array.isArray(lib.packages)) {
                            lib.packages.forEach(pkgArray => {
                                pkgArray.forEach(pkg => {
                                    if (pkg.binary === "Build") {
                                        isSomeBuild = true;
                                    } else {
                                        isAllBuild = false;
                                    }
                                    if (pkg.context === "build") {
                                        shape = 'ellipse';
                                        borderColor = '#0000FF';  // Different border color for build context
                                    }
                                });
                            });
                        }

                        var nodeColor;
                        if (isAllBuild) {
                            nodeColor = "#ffff37"; // Light orange for all build
                        } else if (isSomeBuild) {
                            nodeColor = "#ff9b28"; // Yellow for some build
                        } else if (libType === "Cache") {
                            nodeColor = "#70c7e6"; // Light green
                        } else if (libType === "Download") {
                            nodeColor = "#79eb8a"; // Light blue
                        } else {
                            nodeColor = "#FFFFFF"; // Default color
                        }

                        if (libIndex % columns === 0) {
                            posX = 0;
                            posY += yOffset; // move to the next row
                        }

                        elements.push({
                            data: { id: libId, parent: stepId, label: libLabel, info: lib, color: nodeColor, shape: shape, borderColor: borderColor },
                            position: { x: posX + libLabel.length / 2.0 * 12, y: posY }
                        });
                        posX += libLabel.length * 12;
                    });

                    if (stepIndex > 0) {
                        var prevStepId = 'step' + (stepIndex - 1);
                        edges.push({ data: { id: prevStepId + '_to_' + stepId, source: prevStepId, target: stepId } });
                    }

                    posY += yOffset * 2;
                    posX = 0;
                });

                var cy = cytoscape({
                    container: document.getElementById('cy'),
                    boxSelectionEnabled: false,
                    style: [
                        {
                            selector: 'node[color][shape][borderColor]',
                            style: {
                                'shape': 'data(shape)',
                                'content': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'background-color': 'data(color)',
                                'border-color': 'data(borderColor)',
                                'border-width': 1,
                                'width': function(ele) { return ele.data('label').length * 10.5; },
                                'padding': '5px',
                                'font-family': 'monospace',
                                'font-size': '16px'
                            }
                        },
                        {
                            selector: ':parent',
                            style: {
                                'text-valign': 'top',
                                'text-halign': 'center',
                                'shape': 'round-rectangle',
                                'background-opacity': 0.1,
                                'border-color': '#004D40',
                                'border-width': 2,
                                'padding': 10,
                                'font-family': 'monospace',
                                'font-size': '16px'
                            }
                        },
                        {
                            selector: 'edge',
                            style: {
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'line-color': '#004D40',
                                'target-arrow-color': '#004D40',
                                'width': 2
                            }
                        }
                    ],
                    elements: {
                        nodes: elements,
                        edges: edges
                    },
                    layout: {
                        name: 'preset',
                        padding: 5,
                        fit: true
                    }
                });

                // Add click event listener to nodes
                cy.on('tap', 'node', function(evt){
                    var node = evt.target;
                    var info = node.data('info');
                    var infoHtml = '';
                    for (var key in info) {
                        if (info.hasOwnProperty(key)) {
                            infoHtml += '<p><strong>' + key + ':</strong> ' + JSON.stringify(info[key], null, 2) + '</p>';
                        }
                    }
                    document.getElementById('node-info').innerHTML = infoHtml;
                });
            });
        </script>
    </body>
</html>
"""


def _render_build_order(build_order, template):
    from conan import __version__
    context = {'build_order': build_order, 'version': __version__}
    return template.render(context)


def format_build_order_html(result):
    build_order = result["build_order"]
    build_order = build_order["order"] if isinstance(build_order, dict) else build_order
    template = Template(build_order_html, autoescape=select_autoescape(['html', 'xml']))
    cli_out_write(_render_build_order(build_order, template))
