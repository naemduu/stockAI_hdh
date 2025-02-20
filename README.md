# stockAI_hdh

## 개요
트랜스포머를 이용한 주식 포트폴리오 생성기

## 유의사항
yfinance 다운로드 오류 시:
```bash
pip install --upgrade yfinance
```

## 실행 방법
(1) 주식 데이터 업데이트
```bash
python download/download.py
```
(2) 모델 훈련
```bash
python train_part.py
```

## 업데이트 내역

### v1.0.0 (2025-02-20)
- 기본 트랜스포머 모델
- train과 test분리

