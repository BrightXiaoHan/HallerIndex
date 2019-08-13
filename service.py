import io
import tornado
from tornado.web import RequestHandler, Application
from src import diagnosis_v1, diagnosis_v2

class _BaseDiagnosisiHandler(RequestHandler):

    def post(self):
        ret = {'result': 'OK'}
        upload_path = os.path.dirname(__file__) # 文件的暂存路径
        file_metas = self.request.files.get('file', None)  # 提取表单中‘name’为‘file’的文件元数据

        if not file_metas:
            ret['result'] = 'Invalid Files.'
        else:
            for meta in file_metas:
                file_content = meta['body']
                break

        self.set_header ('Content-Type', 'application/octet-stream')
        self.set_header ('Content-Disposition', 'attachment; filename='+filename)

        
        data = self.on_process_file(file_content)
        self.write(data)

        self.finish()

    def on_process_file(self, file_content):
        raise NotImplementedError


class DiagnosisHandlerV1(_BaseDiagnosisiHandler):

    def on_process_file(self, file_content):
        reader = io.BufferedReader(io.BytesIO(file_content))


class DiagnosisHandlerV2(_BaseDiagnosisiHandler):

    def on_process_file(self, file_content):
        pass


if __name__ == "__main__":
    application = Application([
        (r"/yuyi/api/chest/v1", DiagnosisHandlerV1),
        (r"/yuyi/api/chest/v2", DiagnosisHandlerV2)
    ])
    application.listen(3333)
    tornado.ioloop.IOLoop.instance().start()