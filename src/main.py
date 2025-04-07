import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

class MotherApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 사용자 맞춤 AI 설문 버튼
        btn_main = QPushButton("사용자 맞춤 AI 설문", self)
        btn_main.clicked.connect(self.run_main)
        layout.addWidget(btn_main)
        
        # MINERVA 실행 버튼
        btn_main2 = QPushButton("MINERVA 실행", self)
        btn_main2.clicked.connect(self.run_main2)
        layout.addWidget(btn_main2)
        
        self.setLayout(layout)
        
        self.setGeometry(900, 900, 900, 500)
        self.setWindowTitle("Mother Application")
        self.show()
    
    def run_main(self):
        # 현재 스크립트와 같은 디렉토리에 있는 파일 경로 사용
        script_dir = os.path.dirname(os.path.abspath(__file__))
        user_survey_path = os.path.join(script_dir, "user_survey.py")
        
        # 파일 존재 여부 확인 후 실행
        if os.path.exists(user_survey_path):
            subprocess.Popen(["python", user_survey_path])
        else:
            print(f"Error: {user_survey_path} 파일을 찾을 수 없습니다.")
    
    def run_main2(self):
        # 현재 스크립트와 같은 디렉토리에 있는 파일 경로 사용
        script_dir = os.path.dirname(os.path.abspath(__file__))
        minerva_path = os.path.join(script_dir, "minerva.py")
        
        # 파일 존재 여부 확인 후 실행
        if os.path.exists(minerva_path):
            subprocess.Popen(["python", minerva_path])
        else:
            print(f"Error: {minerva_path} 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MotherApp()
    sys.exit(app.exec_())
