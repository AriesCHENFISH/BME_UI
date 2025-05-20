from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PatientInfo(db.Model):
    __tablename__ = 'patient_info'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255))
    id_card = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    age = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))

    analysis_results = db.relationship('AnalysisResult', backref='patient', cascade="all, delete-orphan")


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(255), db.ForeignKey('patient_info.patient_id', ondelete='CASCADE', onupdate='CASCADE'))
    analysis_time = db.Column(db.DateTime, server_default=db.func.now())
    result = db.Column(db.Text)
    report_path = db.Column(db.String(255))
    name = db.Column(db.String(255))
    # image_path = db.Column(db.String(255))
