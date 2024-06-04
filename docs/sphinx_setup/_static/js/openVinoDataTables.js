$(document).ready(function () {
  var table = $('table.modeldata').DataTable({
      "autoWidth": false,
      "scrollY": "550px"
  });

  document.querySelectorAll('input.toggle-vis').forEach((el) => {
      el.addEventListener('click', function (e) {
  
        let columnIdx = e.target.getAttribute('data-column');         
        let column = table.column(columnIdx);
        column.visible(!column.visible());
      });
  });
});