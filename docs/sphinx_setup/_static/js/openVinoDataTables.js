$(document).ready(function () {
    var table = $('table.modeldata').DataTable({
      "autoWidth": false,
    });

    document.querySelectorAll('input.toggle-vis').forEach((el) => {
      if (el.checked) {
        table.columns([el.getAttribute('data-column')]).visible(false, true);
      }

      el.addEventListener('click', function (e) {
        let columnIdx = e.target.getAttribute('data-column');
        let column = table.column(columnIdx);
        column.visible(!column.visible());
      });
    });
  });