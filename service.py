import io
import tornado
import json
from tornado.web import RequestHandler, Application
from src import diagnosis_v1, diagnosis_v2
from src.utils import image_to_base64

class _BaseDiagnosisiHandler(RequestHandler):

    def post(self):
        ret = {'result': 'ok'}
        file_meta = self.request.files.get('files', None)[0]  # 提取表单中‘name’为‘file’的文件元数据

        if not file_meta:
            ret['result'] = 'err'
        else:
            file_content = file_meta['body']
        
        data, figure = self.on_process_file(file_content)
        ret["data"] = data
        ret["figure"] = image_to_base64(figure).decode("ascii")
        
        self.set_header("Content-Type","application/json")
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST')

        self.write(json.dumps(ret))

    def on_process_file(self, file_content):
        raise NotImplementedError


class DiagnosisHandlerV1(_BaseDiagnosisiHandler):

    def on_process_file(self, file_content):
        reader = io.BufferedReader(io.BytesIO(file_content))
        reader.raw.name = "tmp_name"
        (h1, h2), figure = diagnosis_v1(reader)

        data = {
            "H1": h1,
            "H2": h2
        }
        return data, figure


class DiagnosisHandlerV2(_BaseDiagnosisiHandler):

    def on_process_file(self, file_content):
        reader = io.BufferedReader(io.BytesIO(file_content))
        reader.raw.name = "tmp_name"
        haller, figure = diagnosis_v2(reader)

        data = {
            "haller": haller
        }
        return data, figure


if __name__ == "__main__":
    application = Application([
        (r"/yuyi/api/chest/v1", DiagnosisHandlerV1),
        (r"/yuyi/api/chest/v2", DiagnosisHandlerV2)
    ])
    application.listen(10001)
    tornado.ioloop.IOLoop.instance().start()