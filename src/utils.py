import io
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

def wrap_dicom_buffer(buffer, name="tmp_name"):
    """包装网络传输的二进制文件流，供dicom包读取
    
    Args:
        buffer (byte): 二进制文件流。如果是文件路径则直接返回
    """
    if isinstance(buffer, bytes):
        reader = io.BufferedReader(io.BytesIO(buffer))
        reader.raw.name = name
        return reader
    else:
        return buffer


def concatenate_images(images, mode="horizontal"):
    """拼接图像
    
    Args:
        images (list): 待拼接的图像列表。PIL.Image
        mode (str, optional): 拼接模式。"horizontal": 横向拼接， "vertical": 纵向拼接。 Defaults to "horizontal".
    
    Returns:
        PIL.Image: 拼接结果
    """
    widths, heights = zip(*(i.size for i in images))

    if mode == "horizontal":
        total_width = sum(widths)
        max_height = max(heights)
    elif mode == "vertical":
        total_width = max(widths)
        max_height = sum(heights)
    else:
        raise AttributeError("Parameter mode must be one of 'horizontal' and 'vertical'.")
    
    new_im = Image.new('RGB', (total_width, max_height))

    offset = 0
    for im in images:
        if mode == "horizontal":
            new_im.paste(im, (offset,0))
            offset += im.size[0]
        elif mode == "vertical":
            new_im.paste(im, (0, offset))
            offset += im.size[1]
        else:
            raise AttributeError("Parameter mode must be one of 'horizontal' and 'vertical'.")

    return new_im

