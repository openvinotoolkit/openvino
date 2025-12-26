import json

from jinja2 import select_autoescape, Template

from conan.api.output import cli_out_write, Color


severity_order = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1
}


def text_vuln_formatter(result):

    severity_colors = {
        "Critical": Color.BRIGHT_RED,
        "High": Color.RED,
        "Medium": Color.BRIGHT_YELLOW,
        "Low": Color.BRIGHT_CYAN
    }

    def wrap_and_indent(txt, limit=80, indent=2):
        txt = txt.replace("\n", " ").strip()
        if len(txt) <= limit:
            return " " * indent + txt
        lines = []
        while len(txt) > limit:
            split_index = txt.rfind(" ", 0, limit)
            if split_index == -1:
                split_index = limit
            lines.append(" " * indent + txt[:split_index].strip())
            txt = txt[split_index:].strip()
        lines.append(" " * indent + txt)
        return "\n".join(lines)

    total_vulns = 0
    summary_lines = []

    for ref, pkg_info in result["data"].items():
        edges = pkg_info.get("vulnerabilities", {}).get("edges", [])
        count = len(edges)

        border_line = "*" * (len(ref) + 4)
        cli_out_write("\n" + border_line, fg=Color.BRIGHT_WHITE)
        cli_out_write(f"* {ref} *", fg=Color.BRIGHT_WHITE)
        cli_out_write(border_line, fg=Color.BRIGHT_WHITE)

        if "error" in pkg_info:
            details = pkg_info["error"].get("details", "")
            cli_out_write(f"\n{details}\n", fg=Color.BRIGHT_YELLOW)
            continue

        if not count:
            cli_out_write("\nNo vulnerabilities found.\n", fg=Color.BRIGHT_GREEN)
            continue

        total_vulns += count
        summary_lines.append(
            f"{ref} {count} {'vulnerability' if count == 1 else 'vulnerabilities'} found")
        cli_out_write(f"\n{count} {'vulnerability' if count == 1 else 'vulnerabilities'} found:\n",
                      fg=Color.BRIGHT_YELLOW)

        sorted_vulns = sorted(edges,
                              key=lambda v: -severity_order.get(v["node"].get("severity", "Medium"),
                                                                2))

        for vuln in sorted_vulns:
            node = vuln["node"]
            name = node["name"]
            sev = node.get("severity", "Medium")
            sev_color = severity_colors.get(sev, Color.BRIGHT_YELLOW)
            score = node.get("cvss", {}).get("preferredBaseScore")
            score_txt = f", CVSS: {score}" if score else ""
            desc = node.get("description", "")
            desc = (desc[:240] + "...") if len(desc) > 240 else desc
            desc_wrapped = wrap_and_indent(desc)
            isWithdrawn = node.get("withdrawn", False)
            publishedAt = node.get("publishedAt")

            cli_out_write(f"- {name}", fg=Color.BRIGHT_WHITE, endline="")
            if isWithdrawn:
                cli_out_write(" [WITHDRAWN]", fg=Color.BRIGHT_CYAN, endline="")
            cli_out_write(f" (Severity: {sev}{score_txt})", fg=sev_color)
            advisories = node.get("advisories", {})
            jfrog_advisories = [adv for adv in advisories
                                if adv.get("name", "").startswith("JFSA-")]
            for adv in jfrog_advisories:
                if adv.get("shortDescription"):
                    cli_out_write(f"  Summary provided by JFrog Research ({adv['name']})",
                                  fg=Color.BRIGHT_GREEN)
                    cli_out_write(wrap_and_indent(f"Short description: {adv['shortDescription']}",
                                                  indent=4))
                    if adv.get("severity"):
                        cli_out_write(f"    Severity: ", endline="")
                        cli_out_write(adv['severity'], fg=severity_colors.get(adv['severity']))
                        reasons = adv.get("impactReasons", [])
                        if reasons:
                            cli_out_write(f"    Impact reasons:")
                            for reason in reasons:
                                cli_out_write(wrap_and_indent(f"* {reason['name']}", indent=8),
                                              fg=Color.GREEN if reason['isPositive'] else Color.RED)
                    if result["provider_url"]:
                        expected_url = (result["provider_url"].rstrip("/")
                                        + f"/ui/catalog/vulnerabilities/details/{adv['name']}")
                        cli_out_write(f"    Url: {expected_url}")
                    cli_out_write("")

            cli_out_write("\n" + desc_wrapped)

            if publishedAt:
                cli_out_write(f"  Published at: ", endline="", fg=Color.BRIGHT_BLUE)
                cli_out_write(publishedAt)

            references = node.get("references")
            if references:
                cli_out_write(f"  url: ", endline="", fg=Color.BRIGHT_BLUE)
                cli_out_write(references[0])

            vulnerablePackages = node.get("vulnerablePackages")
            if vulnerablePackages:
                fixVersions = [fix['version']
                               for fix_edge in vulnerablePackages.get("edges", [])
                               for fix in fix_edge['node'].get("fixVersions", [])]
                if fixVersions:
                    cli_out_write(f"  fixed in version(s): ", endline="", fg=Color.BRIGHT_BLUE)
                    cli_out_write(', '.join(fixVersions))
            cli_out_write("")

    color_for_total = Color.BRIGHT_RED if total_vulns else Color.BRIGHT_GREEN

    cli_out_write(f"Total vulnerabilities found: {total_vulns}\n", fg=color_for_total)

    if total_vulns > 0:
        cli_out_write("\nSummary:\n", fg=Color.BRIGHT_WHITE)
        for line in summary_lines:
            cli_out_write(f"- {line}", fg=Color.BRIGHT_WHITE)

        cli_out_write("\nIf you are using packages from Conan Center, some vulnerabilities may have already been mitigated "
                      "through patches applied in the recipe.\nTo verify if a patch has been applied, check the recipe in Conan Center.\n",
                      fg=Color.BRIGHT_YELLOW)

    if total_vulns > 0 or not "error" in result:
        cli_out_write("\nVulnerability information provided by JFrog Catalog. Check "
                      "https://audit.conan.io/jfrogcuration for more information.\n",
                      fg=Color.BRIGHT_GREEN)
        cli_out_write("You can send questions and report issues about "
                      "the returned vulnerabilities to conan-research@jfrog.com.\n",
                      fg=Color.BRIGHT_GREEN)


def json_vuln_formatter(result):
    cli_out_write(json.dumps(result, indent=4))


def _render_vulns(vulns, template):
    from conan import __version__
    template = Template(template, autoescape=select_autoescape(['html', 'xml']))
    return template.render(vulns=vulns, version=__version__)


vuln_html = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>Conan Audit Vulnerabilities Report</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/2.3.4/css/dataTables.dataTables.min.css">
  <style>
    body { margin: 0; padding: 0; font-family: Arial, sans-serif; background: #333; color: #ffffff; }
    .container { width: 95%; margin: 40px auto; padding: 20px; background: #222; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-radius: 8px; }
    h1 { text-align: center; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; table-layout: fixed; padding-top: 10px;}
    col[data-dt-column="0"] { width: 10%; }
    col[data-dt-column="1"] { width: 10%; }
    col[data-dt-column="2"] { width: auto; }
    thead { background: #333; color: #fff; }
    thead th { padding: 12px; text-align: left; }
    tbody tr { border-bottom: 1px solid #ddd; }
    tbody tr:hover { background: #f0f0f0; }
    td { padding: 10px; vertical-align: top; white-space: normal; word-wrap: break-word; overflow-wrap: break-word; word-break: break-word;}
    .severity-badge { padding: 2px 4px; border-radius: 4px; color: #fff; font-weight: bold; display: inline-block; }
    .severity-Critical { background: #d9534f; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(217,83,79,0.7); } 70% { box-shadow: 0 0 0 12px rgba(217,83,79,0); } 100% { box-shadow: 0 0 0 0 rgba(217,83,79,0); } }
    .severity-High { background: #f0ad4e; }
    .severity-Medium { background: #f7ecb5; color: #333; }
    .severity-Low { background: #5cb85c; }
    .footer { text-align: center; color: #666; margin-bottom: 10px; }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .jfrog-research-summary { padding: 10px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #555; }
    .jfrog-research-details { margin-top: 10px; }
  </style>
  <script
    src="https://code.jquery.com/jquery-3.7.1.min.js"
    integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
    crossorigin="anonymous"></script>
  <script src="https://cdn.datatables.net/2.3.4/js/dataTables.min.js"></script>
  <script>
    $(document).ready(function(){
      $('#vuln_table').DataTable({
        "columnDefs": [
          { "orderable": true, "targets": [0, 1] },
          { "orderable": false, "targets": [2] }
        ],
        "order": [[1, "desc"]],
        "autoWidth": false,
      });
    });
  </script>
</head>
<body>
  <div class="container">
    <h1>Conan Audit Vulnerabilities Report</h1>
    <table id="vuln_table" class="stripe">
      <colgroup>
        <col class="pkg-col">
        <col class="info-col">
        <col class="desc-col">
      </colgroup>
      <thead>
        <tr>
          <th>Package</th>
          <th>Info</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
      {% for vuln in vulns %}
        {% set parts = vuln.severity.split(' - ') %}
        {% set severity_id = parts[0] %}
        {% set severity_label = parts[1] if parts|length > 1 else parts[0] %}
        <tr>
          <td>
            {{ vuln.package }}
          </td>
          <td>
            <span style="display: none">{{ vuln.score }}</span>
            {% if vuln.withdrawn %}
                <span style="color: #00ced1; font-weight: bold;">[WITHDRAWN]</span><br>
            {% endif %}
            {{ vuln.vuln_id }}
            <br>
            {% if vuln.severity not in ['N/A', ''] %}
              <span class="severity-badge severity-{{ severity_label }}">{{ severity_label }}</span>
            {% else %}
              {{ vuln.severity }}
            {% endif %}
            {{ vuln.score }}
          </td>
          <td>
            {% for research in vuln.advisories %}
                {% if research.shortDescription %}
                <div class="jfrog-research-summary">
                    <strong>Summary provided by JFrog Research <span style="color: green">({{ research.name }})</span></strong>
                    <div class="jfrog-research-details">
                        <b>Short description:</b> {{ research.shortDescription }}<br>
                        {% if research.severity %}
                            <b>Impact severity:</b> <span class="severity-badge severity-{{ research.severity }}">{{ research.severity }}</span><br>
                            {% if research.impactReasons %}
                                <b>Impact reasons:</b>
                                <ul>
                                {% for reason in research.impactReasons %}
                                    <li style="color: {{ 'inherit' if reason.isPositive else 'red' }};">{{ reason.name }}</li>
                                {% endfor %}
                                </ul>
                            {% endif %}
                        {% endif %}
                        {% if vuln.provider_url %}
                            {% set expected_url = vuln.provider_url.rstrip('/') + '/ui/catalog/vulnerabilities/details/' + research.name %}
                            <b>More info available in:</b> <a href="{{ expected_url }}" target="_blank">{{ expected_url }}</a><br>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
            {% endfor %}
            <strong>Description:</strong>
            <br>
            {{ vuln.description }}
            {% if vuln.publishedAt %}
                <br>
                <br>
                <strong>Published at:</strong> {{ vuln.publishedAt }}
            {% endif %}
            {% if vuln.fixVersions %}
                <div class="fix-versions-section">
                    <br>
                    <strong>Fixed in version(s):</strong>
                    <br>
                    {% for version in vuln.fixVersions %}
                        <span class="severity-badge severity-Medium">{{ version }}</span>
                    {% endfor %}
                </div>
            {% endif %}
            {% if vuln.references %}
              <br><strong>References:</strong>
              <ul>
                {% for ref in vuln.references %}
                  <li><a href="{{ ref }}" target="_blank">{{ ref }}</a></li>
                {% endfor %}
              </ul>
            {% endif %}
            {% if vuln.aliases %}
              <br><strong>Aliases:</strong> {{ ', '.join(vuln.aliases) }}
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    <div class="footer">
      <p>Vulnerability information provided by JFrog Advanced Security. Please check <a href="https://jfrog.com/advanced-security/" target="_blank">https://jfrog.com/advanced-security/</a> for more information.</p>
      <p>You can send questions and report issues about the returned vulnerabilities to <a href="mailto:conan-research@jfrog.com">conan-research@jfrog.com</a>.</p>
      <p>Conan version: {{ version }}</p>
    </div>
  </div>
</body>
</html>
"""


def html_vuln_formatter(result):
    vulns = []
    for ref, pkg_info in result["data"].items():
        edges = pkg_info.get("vulnerabilities", {}).get("edges", [])
        if not edges:
            description = "No vulnerabilities found." if not "error" in pkg_info else pkg_info["error"].get("details", "")
            vulns.append({
                "package": ref,
                "vuln_id": "-",
                "aliases": [],
                "severity": "N/A",
                "score": "-",
                "description": description,
                "references": [],
                "withdrawn": False,
                "advisories": [],
                "provider_url": result.get("provider_url"),
                "fixVersions": [],
                "publishedAt": None
            })
        else:
            sorted_vulns = sorted(edges, key=lambda v: -severity_order.get(v["node"].get("severity", "Medium"), 2))
            for vuln in sorted_vulns:
                node = vuln["node"]
                name = node.get("name")
                sev = node.get("severity", "Medium")
                sev = f"{severity_order.get(sev, 2)} - {sev}"
                score = node.get("cvss", {}).get("preferredBaseScore")
                score_txt = f"CVSS: {score}" if score else "-"
                aliases = node.get("aliases", [])
                references = node.get("references", [])
                desc = node.get("description", "")
                withdrawn = node.get("withdrawn", False)
                advisories = node.get("advisories", [])
                jfrogAdvisories = [adv for adv in advisories
                                   if adv.get("name", "").startswith("JFSA-")]
                fixVersions = [fix['version']
                               for fix_edge in node.get("vulnerablePackages", {}).get("edges", [])
                               for fix in fix_edge['node'].get("fixVersions", [])]
                vulns.append({
                    "package": ref,
                    "vuln_id": name,
                    "aliases": aliases,
                    "severity": sev,
                    "score": score_txt,
                    "description": desc,
                    "references": references,
                    "withdrawn": withdrawn,
                    "advisories": jfrogAdvisories,
                    "provider_url": result.get("provider_url"),
                    "fixVersions": fixVersions,
                    "publishedAt": node.get("publishedAt")
                })

    cli_out_write(_render_vulns(vulns, vuln_html))
