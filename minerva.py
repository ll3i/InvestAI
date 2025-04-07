import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import openai
from dotenv import load_dotenv

class OpenAIThread(QThread):
    response_received = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.client = openai.OpenAI(api_key=api_key)
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def run(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.messages
            )
            self.response_received.emit(response.choices[0].message.content)
        except Exception as e:
            self.response_received.emit(f"Error: {str(e)}")

class ChatbotGUI(QMainWindow):
    def __init__(self, api_key, personalized_prompts):
        super().__init__()
        self.api_key = api_key
        self.personalized_prompts = personalized_prompts
        self.openai_thread = OpenAIThread(api_key)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('투자 챗봇')
        self.setGeometry(100, 100, 800, 600)

        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 채팅 기록 표시 영역
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # 입력 영역
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        send_button = QPushButton('전송')
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)

        layout.addLayout(input_layout)

        # OpenAI 스레드 연결
        self.openai_thread.response_received.connect(self.display_response)

    def send_message(self):
        user_input = self.input_field.text()
        if user_input:
            self.display_message("사용자", user_input)
            self.input_field.clear()
            
            # 시스템 프롬프트 추가
            self.openai_thread.add_message("system", self.personalized_prompts)
            self.openai_thread.add_message("user", user_input)
            self.openai_thread.start()

    def display_message(self, sender, message):
        self.chat_display.append(f"{sender}: {message}\n")

    def display_response(self, response):
        self.display_message("챗봇", response)

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    personalized_prompts = """
    당신은 전문 투자 자문가입니다. 사용자의 투자 관련 질문에 대해 전문적이고 상세한 답변을 제공해주세요.
    답변은 다음을 포함해야 합니다:
    1. 현재 시장 상황 분석
    2. 관련 투자 전략 제안
    3. 위험 관리 방안
    4. 구체적인 실행 계획
    """

    app = QApplication(sys.argv)
    gui = ChatbotGUI(api_key, personalized_prompts)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 