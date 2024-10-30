$(document).ready(function () {
  var pageTitle = document.title;
  var columnDefs;
  if(pageTitle.includes('Most Efficient Large Language Models for AI PC'))
  {
    columnDefs=  [
      { "visible": false, "targets": [1, 2, 3, 4, 5] }
    ]
  }
  else
  {
    columnDefs=[]
  }
  
  var table = $('table.modeldata').DataTable({
    responsive: true,
    "autoWidth": false,
    stateSave: true,
    language: {
      buttons: {
        colvisRestore: "Show all columns"
      }
    },
    lengthMenu: [
      [10, 25, 50, -1],
      ['10 rows', '25 rows', '50 rows', 'Show all rows']
    ],
    "columnDefs": columnDefs,
    layout: {
      topStart: {
        buttons: [
          'pageLength',
          {
            extend: 'colvis',
            postfixButtons: ['colvisRestore'],
          },
          {
            extend: 'print',
            text: 'Print pdf',
            exportOptions: {
              columns: ':visible'
            }
          }
        ]
      }
    }
  });
});