import numpy
import base64
from PIL import Image
from io import BytesIO
 
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2img(fig):
    """将pyplot figure转化为PIL 图像
    
    Args:
        fig ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

def image_to_base64(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def erode(img):

    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 获得结构元素
    # 第一个参数：结构元素形状，这里是矩形
    # 第二个参数：结构元素大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 执行腐蚀
    dst = cv2.erode(binary, kernel)
    return dst

def dilate(img):
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 获得结构元素
    # 第一个参数：结构元素形状，这里是矩形
    # 第二个参数：结构元素大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 执行膨胀
    dst = cv2.dilate(binary, kernel)
    return dst