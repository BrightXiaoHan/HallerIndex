/*
	By Osvaldas Valutis, www.osvaldas.info
	Available for use under the MIT License
*/

'use strict';

function postData(files){
	var formData = new FormData();
	$.each(files,function(i,file){
		formData.append('file['+i+']', file);
	});
    $.ajax({
        url:'http://183.214.140.56:10001/yuyi/api/chest/v2', /*接口域名地址*/
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
				fileName = this.files.length + "files selected."
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