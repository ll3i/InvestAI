# 06. AI/ML 모델 개요

## 6.1 AI 아키텍처 설계

### 6.1.1 멀티 에이전트 시스템 구조

Pixie의 핵심은 서로 다른 전문성을 가진 4개의 AI 에이전트가 협업하는 멀티 에이전트 시스템입니다. 이 구조는 단일 모델의 한계를 극복하고, 각 도메인에 특화된 분석을 통해 보다 정확하고 균형 잡힌 투자 조언을 제공합니다.

**시스템 아키텍처 다이어그램**
```
사용자 입력
    ↓
[AI-A: 초기 분석 에이전트]
    ├─ 사용자 의도 파악
    ├─ 투자 프로필 매칭
    └─ 기초 전략 수립
         ↓
[AI-A2: 쿼리 정제 에이전트]
    ├─ 모호성 해소
    ├─ 데이터 요구사항 정의
    └─ 분석 범위 설정
         ↓
[AI-B: 데이터 분석 에이전트]
    ├─ 실시간 데이터 분석
    ├─ 기술적/펀더멘탈 분석
    └─ 리스크 평가
         ↓
[최종 통합 에이전트]
    ├─ 결과 종합
    ├─ 일관성 검증
    └─ 최종 조언 생성
         ↓
사용자 응답
```

**에이전트 간 통신 프로토콜**
```python
class AgentCommunication:
    def __init__(self):
        self.message_queue = []
        self.context = {}
    
    def send_message(self, from_agent, to_agent, message, data):
        """에이전트 간 메시지 전송"""
        msg = {
            'from': from_agent,
            'to': to_agent,
            'timestamp': datetime.now(),
            'message': message,
            'data': data,
            'correlation_id': generate_uuid()
        }
        self.message_queue.append(msg)
        return msg['correlation_id']
    
    def receive_message(self, agent_id):
        """특정 에이전트의 메시지 수신"""
        messages = [m for m in self.message_queue if m['to'] == agent_id]
        return messages
```

### 6.1.2 대규모 언어 모델 (LLM) 통합

**기반 모델 선택 및 최적화**

1. **OpenAI GPT-4 Turbo**
   - 용도: 복잡한 자연어 이해 및 생성
   - 컨텍스트 윈도우: 128K 토큰
   - 특징: 최신 금융 지식, 다국어 지원
   - 파인튜닝: 한국 금융 시장 특화 데이터셋

2. **Claude 3 Opus**
   - 용도: 장문 분석 및 리포트 생성
   - 컨텍스트 윈도우: 200K 토큰
   - 특징: 높은 정확도, 환각 현상 최소화
   - 활용: 재무제표 분석, 투자 전략 수립

3. **Llama 3 70B (자체 호스팅)**
   - 용도: 비용 효율적인 대량 처리
   - 인프라: 자체 GPU 클러스터
   - 특징: 커스터마이징 가능, 데이터 보안
   - 최적화: 양자화(Quantization) 적용

**프롬프트 엔지니어링**
```python
class PromptTemplate:
    @staticmethod
    def create_investment_prompt(user_profile, market_context, query):
        prompt = f"""
        당신은 20년 경력의 투자 전문가입니다.
        
        [사용자 프로필]
        - 투자 경험: {user_profile['experience_level']}
        - 위험 성향: {user_profile['risk_tolerance']}
        - 투자 목표: {user_profile['investment_goal']}
        - 투자 기간: {user_profile['time_horizon']}
        
        [현재 시장 상황]
        - KOSPI: {market_context['kospi']} ({market_context['kospi_change']}%)
        - 주요 이슈: {market_context['major_issues']}
        - 투자 심리: {market_context['sentiment']}
        
        [사용자 질문]
        {query}
        
        위 정보를 바탕으로 개인화된 투자 조언을 제공해주세요.
        답변은 다음 구조를 따라주세요:
        1. 핵심 답변 (2-3문장)
        2. 상세 분석 (시장 상황 고려)
        3. 구체적 행동 방안
        4. 리스크 고려사항
        """
        return prompt
```

## 6.2 머신러닝 모델 구현

### 6.2.1 시계열 예측 모델

**LSTM 기반 주가 예측 모델**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class StockPricePredictionModel:
    def __init__(self, sequence_length=60, n_features=10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
    
    def _build_model(self):
        """LSTM 모델 구조 정의"""
        model = models.Sequential([
            # 첫 번째 LSTM 레이어
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.2),
            
            # 두 번째 LSTM 레이어
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            
            # 세 번째 LSTM 레이어
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense 레이어
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # 다음 날 종가 예측
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """모델 학습"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        return history
```

**Prophet 기반 시계열 분석**

```python
from prophet import Prophet
import pandas as pd

class ProphetForecaster:
    def __init__(self, seasonality_mode='multiplicative'):
        self.model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        # 한국 공휴일 추가
        self.model.add_country_holidays(country_name='KR')
        
        # 커스텀 이벤트 추가 (배당일, 실적 발표일 등)
        self._add_custom_events()
    
    def _add_custom_events(self):
        """주요 이벤트 추가"""
        # 분기 실적 발표 시즌
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            self.model.add_seasonality(
                name=f'{quarter}_earnings',
                period=365.25,
                fourier_order=5
            )
    
    def forecast(self, df, periods=30):
        """미래 가격 예측"""
        # 데이터 포맷 맞추기
        df_prophet = df[['date', 'close']].rename(
            columns={'date': 'ds', 'close': 'y'}
        )
        
        # 모델 학습
        self.model.fit(df_prophet)
        
        # 예측
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

### 6.2.2 포트폴리오 최적화 모델

**Modern Portfolio Theory 기반 최적화**

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class PortfolioOptimizer:
    def __init__(self, returns_data):
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.n_assets = len(self.mean_returns)
    
    def optimize_sharpe_ratio(self, risk_free_rate=0.03):
        """샤프 비율 최대화"""
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio
        
        # 제약 조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 가중치 합 = 1
        ]
        
        # 경계 조건 (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # 초기값
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # 최적화 실행
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def optimize_risk_parity(self):
        """리스크 패리티 포트폴리오"""
        weights = cp.Variable(self.n_assets)
        
        # 리스크 기여도 계산
        portfolio_variance = cp.quad_form(weights, self.cov_matrix)
        
        # 목적 함수: 리스크 기여도의 분산 최소화
        objective = cp.Minimize(portfolio_variance)
        
        # 제약 조건
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # 문제 해결
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return weights.value
```

### 6.2.3 감성 분석 모델

**BERT 기반 뉴스 감성 분석**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NewsSentimentAnalyzer:
    def __init__(self, model_name='klue/bert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # 긍정, 중립, 부정
        )
        
        # 파인튜닝된 가중치 로드
        self.load_fine_tuned_weights()
    
    def load_fine_tuned_weights(self):
        """금융 뉴스 데이터로 파인튜닝된 가중치 로드"""
        checkpoint = torch.load('models/financial_bert_kr.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def analyze_sentiment(self, text):
        """텍스트 감성 분석"""
        # 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 결과 해석
        sentiment_scores = predictions[0].cpu().numpy()
        sentiment_labels = ['부정', '중립', '긍정']
        
        result = {
            'sentiment': sentiment_labels[np.argmax(sentiment_scores)],
            'confidence': float(np.max(sentiment_scores)),
            'scores': {
                label: float(score) 
                for label, score in zip(sentiment_labels, sentiment_scores)
            }
        }
        
        return result
```

## 6.3 알고리즘 및 네트워크 구조

### 6.3.1 강화학습 기반 트레이딩 에이전트

**Deep Q-Network (DQN) 구현**

```python
import gym
import numpy as np
import torch
import torch.nn as nn

class TradingEnvironment(gym.Env):
    """트레이딩 환경 정의"""
    def __init__(self, data, initial_balance=10000000):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.current_balance = initial_balance
        self.shares_held = 0
        
        # 행동 공간: 0=Hold, 1=Buy, 2=Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # 상태 공간: [가격, 거래량, 기술적 지표들...]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
    
    def step(self, action):
        """환경 스텝 실행"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # 행동 실행
        if action == 1:  # Buy
            shares_to_buy = self.current_balance // current_price
            self.shares_held += shares_to_buy
            self.current_balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.current_balance += self.shares_held * current_price
            self.shares_held = 0
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 보상 계산
        total_value = self.current_balance + self.shares_held * current_price
        reward = (total_value - self.initial_balance) / self.initial_balance
        
        # 종료 조건
        done = self.current_step >= len(self.data) - 1
        
        # 다음 상태
        obs = self._get_observation()
        
        return obs, reward, done, {}
    
    def _get_observation(self):
        """현재 상태 벡터 생성"""
        # 가격, 거래량, RSI, MACD 등 기술적 지표 포함
        row = self.data.iloc[self.current_step]
        obs = np.array([
            row['close'], row['volume'], row['rsi'], row['macd'],
            row['bollinger_upper'], row['bollinger_lower'],
            # ... 추가 특징들
        ])
        return obs

class DQNAgent(nn.Module):
    """DQN 에이전트 네트워크"""
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

### 6.3.2 그래프 신경망 기반 종목 관계 분석

**Graph Neural Network (GNN) 구현**

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class StockRelationGNN(torch.nn.Module):
    """종목 간 관계를 학습하는 GNN"""
    def __init__(self, num_features, hidden_dim=64, num_classes=3):
        super(StockRelationGNN, self).__init__()
        
        # Graph Convolution 레이어
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)
        
        # 분류 레이어
        self.classifier = torch.nn.Linear(32, num_classes)
        
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x, edge_index, batch):
        """
        x: 노드 특징 (각 종목의 특징)
        edge_index: 엣지 정보 (종목 간 관계)
        batch: 배치 정보
        """
        # Graph Convolution
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # 글로벌 풀링
        x = global_mean_pool(x, batch)
        
        # 분류
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

def create_stock_graph(correlation_matrix, threshold=0.7):
    """상관관계 행렬로부터 그래프 생성"""
    edges = []
    n_stocks = len(correlation_matrix)
    
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            if abs(correlation_matrix[i][j]) > threshold:
                edges.append([i, j])
                edges.append([j, i])  # 양방향 엣지
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
```

## 6.4 모델 학습 및 최적화

### 6.4.1 학습 데이터 구성

**데이터셋 구축 전략**

1. **Historical Data (과거 10년)**
   - 일별 주가 데이터: 200종목 × 2,500일 = 500,000 레코드
   - 재무제표: 200종목 × 40분기 = 8,000 레코드
   - 뉴스 데이터: 일평균 200개 × 3,650일 = 730,000 기사

2. **Feature Engineering**
   ```python
   def create_features(df):
       """투자 예측을 위한 특징 생성"""
       # 가격 관련 특징
       df['returns'] = df['close'].pct_change()
       df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
       df['volatility'] = df['returns'].rolling(20).std()
       
       # 기술적 지표
       df['rsi'] = calculate_rsi(df['close'])
       df['macd'], df['macd_signal'] = calculate_macd(df['close'])
       df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
       
       # 거래량 지표
       df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
       df['obv'] = calculate_obv(df['close'], df['volume'])
       
       # 시장 상대 지표
       df['relative_strength'] = df['returns'] / market_returns
       df['beta'] = calculate_beta(df['returns'], market_returns)
       
       return df
   ```

3. **데이터 분할**
   - Training Set: 70% (2014-2020)
   - Validation Set: 15% (2021-2022)
   - Test Set: 15% (2023-2024)

### 6.4.2 하이퍼파라미터 튜닝

**Optuna를 이용한 자동 튜닝**

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """최적화 목적 함수"""
    # 하이퍼파라미터 제안
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True)
    }
    
    # 모델 학습
    model = XGBRegressor(**params)
    
    # 교차 검증
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='neg_mean_squared_error'
    )
    
    return -scores.mean()

# 최적화 실행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
```

### 6.4.3 모델 앙상블

**다중 모델 앙상블 전략**

```python
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'lstm': LSTMModel(),
            'xgboost': XGBRegressor(),
            'lightgbm': LGBMRegressor(),
            'prophet': ProphetForecaster(),
            'arima': ARIMAModel()
        }
        
        # 모델별 가중치 (메타 학습으로 결정)
        self.weights = {
            'lstm': 0.3,
            'xgboost': 0.25,
            'lightgbm': 0.2,
            'prophet': 0.15,
            'arima': 0.1
        }
    
    def predict(self, X):
        """앙상블 예측"""
        predictions = {}
        
        # 각 모델의 예측값 수집
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # 가중 평균
        ensemble_pred = sum(
            self.weights[name] * pred 
            for name, pred in predictions.items()
        )
        
        # 예측 신뢰도 계산
        confidence = self._calculate_confidence(predictions)
        
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': predictions
        }
    
    def _calculate_confidence(self, predictions):
        """예측 신뢰도 계산"""
        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)
        
        # 표준편차가 작을수록 높은 신뢰도
        confidence = 1 / (1 + std_dev)
        
        return confidence
```

## 6.5 모델 평가 및 검증

### 6.5.1 성능 평가 지표

**투자 성과 평가 메트릭**

```python
class PerformanceMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
        """샤프 비율 계산"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """최대 낙폭 계산"""
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns, max_drawdown):
        """칼마 비율 계산"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(returns):
        """승률 계산"""
        return (returns > 0).mean()
```

### 6.5.2 백테스팅 프레임워크

```python
class BacktestingEngine:
    def __init__(self, strategy, initial_capital=10000000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = []
    
    def run_backtest(self, data, start_date, end_date):
        """백테스팅 실행"""
        portfolio = Portfolio(self.initial_capital)
        
        for date in pd.date_range(start_date, end_date):
            # 당일 데이터
            daily_data = data[data['date'] == date]
            
            # 전략 신호 생성
            signals = self.strategy.generate_signals(daily_data, portfolio)
            
            # 주문 실행
            for signal in signals:
                if signal['action'] == 'BUY':
                    portfolio.buy(signal['ticker'], signal['quantity'], signal['price'])
                elif signal['action'] == 'SELL':
                    portfolio.sell(signal['ticker'], signal['quantity'], signal['price'])
            
            # 포트폴리오 평가
            portfolio.mark_to_market(daily_data)
            
            # 결과 기록
            self.results.append({
                'date': date,
                'total_value': portfolio.total_value,
                'returns': portfolio.daily_return,
                'positions': portfolio.positions.copy()
            })
        
        return self._analyze_results()
```

이러한 체계적인 AI/ML 모델 구조를 통해 Pixie는 최첨단 인공지능 기술을 활용하여 정확하고 신뢰할 수 있는 투자 조언을 제공합니다. 멀티 에이전트 시스템, 다양한 머신러닝 모델, 그리고 강화학습 기반 트레이딩 전략을 결합하여 사용자에게 최적의 투자 솔루션을 제시합니다.