import io
import os
import tornado
import json

import numpy as np

from tornado.web import RequestHandler, Application
from src import diagnosis_v1, diagnosis_v2, depression_degree
from src.utils import image_to_base64

here = os.path.dirname(os.path.abspath(__file__))

class _BaseDiagnosisiHandler(RequestHandler):

    def post(self):
        ret = {'result': 'ok'}
        files = self.request.files.values()  # 提取表单中‘name’为‘file’的文件元数据

        all_files_content = []

        for file_meta in files:
            if file_meta:
                all_files_content.append(file_meta[0]['body'])
        
        data, figure = self.on_process_file(all_files_content)
        ret["data"] = data
        ret["figure"] = image_to_base64(figure).decode("ascii")
        
        self.set_header("Content-Type","application/json")
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST')

        self.write(json.dumps(ret))

    def on_process_file(self, file_content):
        raise NotImplementedError


# # v1 is Deprecation
# class DiagnosisHandlerV1(_BaseDiagnosisiHandler):

#     def on_process_file(self, file_content):
#         reader = io.BufferedReader(io.BytesIO(file_content))
#         reader.raw.name = "tmp_name"
#         (h1, h2), figure = diagnosis_v1(reader)

#         data = {
#             "H1": h1,
#             "H2": h2
#         }
#         return data, figure


class DiagnosisHandlerV2(_BaseDiagnosisiHandler):

    def on_process_file(self, files):

        degrees = []

        for file_content in files:
            reader = io.BufferedReader(io.BytesIO(file_content))
            reader.raw.name = "tmp_name"
            degrees.append(depression_degree(reader))

        degrees = np.array(degrees)
        index = np.argmax(degrees)
        f = files[index]

        reader = io.BufferedReader(io.BytesIO(f))
        reader.raw.name = "tmp_name"

        haller, figure = diagnosis_v2(reader)

        data = {
            "haller": haller
        }
        return data, figure

class IndexHandler(RequestHandler):

    def get(self):
        msg = "胸部指数自动诊断"
        self.render(os.path.join(here, "static/index.html"), info=msg)


if __name__ == "__main__":
    application = Application([
            (r"/yuyi/api/chest/v2", DiagnosisHandlerV2),
            (r"/yuyi/api/chest/index.html", IndexHandler)
        ],
        template_path=os.path.join(here, "static"),
        static_path=os.path.join(here, "static"),
    )
    application.listen(10001)
    tornado.ioloop.IOLoop.instance().start()