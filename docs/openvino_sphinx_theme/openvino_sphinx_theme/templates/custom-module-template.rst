{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% block attributes %}
{% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for attr in attributes %}
      {{ attr }}
   {% endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for func in functions %}
      {% if not func.startswith('_') %}
         {{ func }}
      {% endif %}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
   {% for cl in classes %}
      {{ cl }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for mod in modules %}
   {{ mod }}
{%- endfor %}
{% endif %}
{% endblock %}
