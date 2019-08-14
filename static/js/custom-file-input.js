/*
	By Osvaldas Valutis, www.osvaldas.info
	Available for use under the MIT License
*/

'use strict';

function postData(files){
    var formData = new FormData();
    formData.append("files", files[0]);
    $.ajax({
        url:'http://127.0.0.1:10001/yuyi/api/chest/v2', /*接口域名地址*/
        type:'post',
        data: formData,
        contentType: false,
        processData: false,
        success:function(res){
            if(res["result"]=="ok"){
				$("#dicom_result").attr('src', 'data:image/png;base64,' + res.figure);
            }else if(res["result"]=="err"){
                alert('失败');
            }else{
                console.log(res);
			}
			
        }
    })
}

;( function ( document, window, index )
{
	var inputs = document.querySelectorAll( '.inputfile' );
	Array.prototype.forEach.call( inputs, function( input )
	{
		var label	 = input.nextElementSibling,
			labelVal = label.innerHTML;

		input.addEventListener( 'change', function( e )
		{
			var fileName = '';
			if( this.files && this.files.length == 1 ){
				fileName = e.target.value.split( '\\' ).pop();
			}
			else{
				alert('Please select one file to upload.');
				return;
			}

			if( fileName )
				label.querySelector( 'span' ).innerHTML = fileName;
			else
				label.innerHTML = labelVal;
				
			postData(this.files)
		});

		// Firefox bug fix
		input.addEventListener( 'focus', function(){ input.classList.add( 'has-focus' ); });
		input.addEventListener( 'blur', function(){ input.classList.remove( 'has-focus' ); });
	});
}( document, window, 0 ));