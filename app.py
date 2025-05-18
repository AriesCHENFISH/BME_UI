from flask import Flask, render_template
import uuid
import traceback
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from models import db  # 从 models 中引入 db 实例和模型类

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://flaskuser:123456@47.122.30.152/breast_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.route('/home')
def home():
    return render_template('home.html')  

@app.route('/')
def start():
    return render_template('start.html')



from flask import Flask, request, jsonify
from Bmode_Dualtask_For_UI.Core.refer_bmode import refer_bmode
from werkzeug.utils import secure_filename
import os
import shutil
from ceus_dual_task.dual_tasks_code.refer_ceus import refer_ceus
import sys
import threading
import queue
from flask import Response, stream_with_context
from werkzeug.datastructures import FileStorage
from io import BytesIO

# 创建全局日志缓冲区
log_queue = queue.Queue()

# 自定义输出类，将print写入队列
class StreamToQueue:
    def write(self, message):
        if message.strip():  # 去掉空行
            log_queue.put(message)

    def flush(self):
        pass

# 重定向标准输出
sys.stdout = StreamToQueue()

def analyze_from_files(bmode_file, ceus_files):
    bmode_result = refer_bmode(bmode_file)

    # 保存 ceus 文件临时目录
    temp_dir = os.path.join('temp_ceus')
    os.makedirs(temp_dir, exist_ok=True)
    for i, file in enumerate(ceus_files):
        filename = secure_filename(f"{i:04d}.png")
        file.save(os.path.join(temp_dir, filename))

    ceus_result = refer_ceus(temp_dir)

    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "bmode": bmode_result,
        "ceus": ceus_result
    }

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            try:
                message = log_queue.get(timeout=5)
                yield f"data: {message}\n\n"
            except queue.Empty:
                continue
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/analyze_bmode', methods=['POST'])
def analyze_bmode():
    print("开始分析")
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    result = refer_bmode(image_file)
    
    return jsonify(result)

import uuid
import traceback
from flask import current_app

@app.route('/analyze_all', methods=['POST'])
def analyze_all():
    try:
        if 'bmode' not in request.files or 'ceus[]' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        # 处理 bmode 图像
        bmode_file = request.files['bmode']
        bmode_result = refer_bmode(bmode_file)

        # 生成唯一临时目录路径
        temp_id = str(uuid.uuid4())
        temp_dir = os.path.join('temp_ceus', temp_id)
        os.makedirs(temp_dir, exist_ok=True)

        # 保存 ceus 图像序列
        ceus_files = request.files.getlist('ceus[]')
        for i, file in enumerate(ceus_files):
            filename = secure_filename(f"{i:04d}.png")
            file.save(os.path.join(temp_dir, filename))

        # 分析
        ceus_result = refer_ceus(temp_dir)

        return jsonify({
            "bmode": bmode_result,
            "ceus": ceus_result
        })
    
    except Exception as e:
        current_app.logger.error(f"分析过程中出现异常: {e}")
        traceback.print_exc()
        return jsonify({"error": "分析失败", "detail": str(e)}), 500

    finally:
        # 无论成功或失败都清理临时目录
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            current_app.logger.warning(f"清理临时目录失败: {cleanup_error}")



from flask import request, jsonify
from models import db, AnalysisResult
from datetime import datetime

@app.route('/save_result', methods=['POST'])
def save_result():
    data = request.get_json()
    new_result = AnalysisResult(
        patient_id=data.get("patient_id"),
        result=data.get("result"),
        report_path=data.get("report_path"),
        image_path=data.get("image_path"),
        analysis_time=datetime.utcnow()
    )
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"status": "success"})


@app.route('/get_history', methods=['POST'])
def get_history():
    patient_id = request.json.get("patient_id")
    results = AnalysisResult.query.filter_by(patient_id=patient_id).order_by(AnalysisResult.analysis_time.desc()).all()
    data = [{
        "analysis_time": r.analysis_time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": r.result,
        "report_path": r.report_path,
        "image_path": r.image_path
    } for r in results]
    return jsonify(data)


# @app.route('/analyze_all', methods=['POST'])
# def analyze_all():
#     if 'bmode' not in request.files or 'ceus[]' not in request.files:
#         return jsonify({'error': 'Missing files'}), 400

#     # 处理 bmode
#     bmode_file = request.files['bmode']
#     bmode_result = refer_bmode(bmode_file)

#     # 处理 ceus 序列（是多个文件）
#     ceus_files = request.files.getlist('ceus[]')
#     temp_dir = os.path.join('temp_ceus')
#     os.makedirs(temp_dir, exist_ok=True)
#     for i, file in enumerate(ceus_files):
#         filename = secure_filename(f"{i:04d}.png")
#         file.save(os.path.join(temp_dir, filename))

#     ceus_result = refer_ceus(temp_dir)

#     # 清理
#     shutil.rmtree(temp_dir, ignore_errors=True)

#     return jsonify({
#         "bmode": bmode_result,
#         "ceus": ceus_result
#     })


from models import db, PatientInfo  # 确保导入你的模型

@app.route('/api/patient_info', methods=['POST'])
def get_patient_info():
    data = request.json
    patient_id = data.get('patient_id')
    id_card = data.get('id_card')

    patient = PatientInfo.query.filter_by(patient_id=patient_id, id_card=id_card).first()

    if patient:
        return jsonify({
            "success": True,
            "data": {
                "id": patient.patient_id,
                "name": patient.name,
                "idCard": patient.id_card,
                "gender": patient.gender,
                "age": patient.age,
                "phone": patient.phone,
                "email": patient.email
            }
        })
    else:
        return jsonify({"success": False, "message": "信息不存在/身份证号错误"}), 404






# @app.route('/analyze_all', methods=['POST'])
# def analyze_all():
#     if 'bmode' not in request.files or 'ceus[]' not in request.files:
#         return jsonify({'error': 'Missing files'}), 400

#     bmode_file = request.files['bmode']
#     ceus_files = request.files.getlist('ceus[]')

#     result = analyze_from_files(bmode_file, ceus_files)
#     return jsonify(result)



# import base64

# @app.route('/auto_load', methods=['POST'])
# def auto_load():
#     patient_id = request.form.get('patient_id')
#     base_path = f"D:/BME_ui/for_test/{patient_id}"

#     bmode_path = os.path.join(base_path, "bmode.png")
#     ceus_dir = os.path.join(base_path, "60frames")

#     if not os.path.exists(bmode_path) or not os.path.isdir(ceus_dir):
#         return jsonify({"error": "路径不存在"}), 404

#     # 读取 B-mode 图像并转为 base64
#     with open(bmode_path, 'rb') as f:
#         bmode_base64 = base64.b64encode(f.read()).decode('utf-8')

#     # 读取 CEUS 第一帧图像并转为 base64
#     ceus_preview_base64 = ""
#     for filename in sorted(os.listdir(ceus_dir)):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             with open(os.path.join(ceus_dir, filename), 'rb') as f:
#                 ceus_preview_base64 = base64.b64encode(f.read()).decode('utf-8')
#             break

#     return jsonify({
#         "bmode_preview": bmode_base64,
#         "ceus_preview": ceus_preview_base64
#     })


from flask import send_file

@app.route('/auto_load_file', methods=['POST'])
def auto_load_file():
    patient_id = request.form.get('patient_id')
    base_path = f"./for_test/{patient_id}"
    bmode_path = os.path.join(base_path, "bmode.png")
    ceus_dir = os.path.join(base_path, "60frames")

    if not os.path.exists(bmode_path) or not os.path.isdir(ceus_dir):
        return jsonify({"error": "路径不存在"}), 404

    # 获取所有 CEUS 图像的路径
    ceus_files = sorted([
        f for f in os.listdir(ceus_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    ceus_paths = [f"/get_image/{patient_id}/60frames/{filename}" for filename in ceus_files]

    return {
        "bmode_path": f"/get_image/{patient_id}/bmode.png",
        "ceus_paths": ceus_paths
    }


from flask import send_file, abort

@app.route('/get_image/<path:subpath>')
def get_image(subpath):
    file_path = os.path.join("./for_test", subpath)
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        mimetype = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.tif': 'image/tiff'
        }.get(ext, 'application/octet-stream')
        return send_file(file_path, mimetype=mimetype)
    return abort(404)

import re
import base64

def load_base64_image(data_url):
    match = re.match(r'data:image/(png|jpeg);base64,(.*)', data_url)
    if not match:
        raise ValueError("Invalid base64 image")
    image_data = base64.b64decode(match.group(2))
    return Image.open(BytesIO(image_data)).convert("RGB")
from PIL import Image, ImageDraw, ImageFont
import base64
import requests
from io import BytesIO

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    name = data.get('name', '未知')
    birthday = data.get('birthday', '未知')
    id = data.get("patientCard",'未知')
    time = data.get("time", '未知')
    result = data.get('result', '未知')
    bmode_url = data.get('bmodeMask')
    ceus_url = data.get('ceusMask')
    bmode_pre_url = data.get('bmodePre')
    ceus_pre_url = data.get('ceusPre')
    advice = data.get("doctorAdvice", "无")

    # 1. 加载模板
    template_path = "./static/image/report.jpg"
    report = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(report)

    # 2. 加载字体（你也可以用系统默认）
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 28)
        print("成功加载 Noto Sans CJK 字体")
    else:
        print("字体文件未找到，使用默认字体")
        font = ImageFont.load_default()

    # 3. 写入信息
    draw.text((120, 390), name, font=font, fill="black")
    draw.text((390, 390), birthday, font=font, fill="black")
    draw.text((640, 390), "女", font=font, fill="black")
    draw.text((150, 430), id, font=font, fill="black")
    draw.text((160, 480), time, font=font, fill="black")
    draw.text((890, 1460), result, font=font, fill="red")
    draw.text((120, 1750), f"{advice}", font=font, fill="black")

    # 4. 加载图像（通过 URL fetch）
    def load_image_from_url(url):
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert("RGB")

    bmode_img = load_image_from_url(bmode_url).resize((280, 280))
    ceus_img = load_image_from_url(ceus_url).resize((280, 280))
    bmode_pre_img = load_base64_image(bmode_pre_url).resize((280, 280))
    ceus_pre_img = load_base64_image(ceus_pre_url).resize((280, 280))

    # 5. 粘贴图像
    report.paste(ceus_img, (390, 760))
    report.paste(bmode_img, (390, 1200))
    report.paste(ceus_pre_img, (70,760))
    report.paste(bmode_pre_img,(70, 1200))

    # 6. 返回图像
    output = BytesIO()
    report.save(output, format="JPEG")
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="diagnosis_report.jpg", mimetype="image/jpeg")





if __name__ == '__main__':
    
    
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8001, debug=True)
    application = app
