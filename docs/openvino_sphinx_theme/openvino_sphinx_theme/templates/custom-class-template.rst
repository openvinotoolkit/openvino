{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for met in methods %}
      ~{{ name }}.{{ met }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for att in attributes %}
      ~{{ name }}.{{ att }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
