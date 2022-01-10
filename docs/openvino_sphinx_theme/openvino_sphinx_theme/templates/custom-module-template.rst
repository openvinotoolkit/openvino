{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
  
{% block attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for attr in attributes %}
      {{ attr }}
   {% endfor %}
{% endblock %}

{% block functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   {% for func in functions %}
      {{ func }}
   {%- endfor %}
{% endblock %}

{% block classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
   {% for cl in classes %}
      {{ cl }}
   {%- endfor %}
{% endblock %}

{% block exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
{% endblock %}

{% block modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for mod in modules %}
   {{ mod }}
{%- endfor %}
{% endblock %}
