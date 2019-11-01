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
	//侦查附件上传情况 ,这个方法大概0.05-0.1秒执行一次
	function OnProgRess(event) {
		var event = event || window.event;
		//console.log(event);  //事件对象
		console.log("已经上传：" + event.loaded); //已经上传大小情况(已上传大小，上传完毕后就 等于 附件总大小)
		//console.log(event.total);  //附件总大小(固定不变)
		var loaded = Math.floor(100 * (event.loaded / event.total)); //已经上传的百分比  
		$("#speed").html(loaded + "%").css("width", loaded + "%");
	};
	var xhr = $.ajaxSettings.xhr(); //创建并返回XMLHttpRequest对象的回调函数(jQuery中$.ajax中的方法)	
  $.ajax({
    url:'http://183.214.140.56:10001/yuyi/api/chest/v2', /*接口域名地址*/
    type:'post',
    data: formData,
    contentType: false,
    processData: false,
	xhr: function() {
		if(OnProgRess && xhr.upload) {
			xhr.upload.addEventListener("progress", OnProgRess, false);
			return xhr;
		}
	}, 	
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
		$("#speed").text("0%").css("width", "0%");;
		$(".box2").attr("hidden","hidden").siblings().removeAttr("hidden");
		$("#dicom_result").attr("src","");
	})
}( document, window, 0 ));