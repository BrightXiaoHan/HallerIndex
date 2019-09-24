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
      	$(".box").attr("hidden","hidden").siblings().removeAttr("hidden");
				$("#dicom_result").attr('src', 'data:image/png;base64,' + res.figure);
      }else if(res["result"]=="err"){
      	alert('失败');
      }else{
        console.log(res);
			}
    }
  })
}

;( function ( document, window, index ){
	var inputs = document.querySelectorAll( '.inputfile' );
	Array.prototype.forEach.call( inputs, function( input )
	{
		var label	 = input.nextElementSibling,
			labelVal = label.innerHTML;

		input.addEventListener( 'change', function( e ){
			
			var fileName = '';
			if( this.files && this.files.length == 1 ){
				fileName = "提示：当前上传的文件为："+e.target.value.split( '\\' ).pop();
			} else{
				fileName = "提示：当前批量上传的文件为："+e.target.value.split( '\\' ).pop()+"...等， 共"+this.files.length+"个文件";
			}

			if( fileName ){
				$("#file-tip").text(fileName)
			}

			if(this.files && this.files.length>0){
				postData(this.files);
			}
		});

		// Firefox bug fix
		input.addEventListener( 'focus', function(){ input.classList.add( 'has-focus' ); });
		input.addEventListener( 'blur', function(){ input.classList.remove( 'has-focus' ); });
	});

	//点击返回按钮
	$("#file-revert").on('click',function(){
		$(".box2").attr("hidden","hidden").siblings().removeAttr("hidden");
		$("#dicom_result").attr("src","");
	})
}( document, window, 0 ));