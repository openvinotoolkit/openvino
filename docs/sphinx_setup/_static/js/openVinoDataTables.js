$(document).ready(function () {
    var table = $('table.modeldata').DataTable({
        "autoWidth": false,
        "scrollY": "400px"
    });
  
    document.querySelectorAll('input.toggle-vis').forEach((el) => {
        el.addEventListener('click', function (e) {
        //   e.preventDefault();
  
          let columnIdx = e.target.getAttribute('data-column');         
          let column = table.column(columnIdx);
          column.visible(!column.visible());
        });
    });
  });