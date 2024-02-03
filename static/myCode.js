function myFunction() {

    const comment = document.getElementById("comment").value;

    const dict_values = {comment} // Variable javascript a diccionario
    const s = JSON.stringify(dict_values); // Javascript value a JSON string
    console.log(s); // Mostrar JSON en consola
    window.alert(s)
    $.ajax({
        url:"/test",
        type:"POST",
        contentType: "application/json",
        data: JSON.stringify(s)});
}