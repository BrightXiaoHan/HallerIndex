import Image
 
def fig2img ( fig ):
    """将pyplot figure转化为PIL 图像
    
    Args:
        fig ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring("RGBA", (w, h), buf.tostring())
