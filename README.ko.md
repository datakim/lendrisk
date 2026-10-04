# lendrisk

신용평가·사업자금융 분석에 사용하는 설치형 Python 라이브러리입니다.
매출·현금흐름 분석, MCA/RBF 상환 시뮬레이션, 스트레스 분석과 신용모형
평가지표를 제공합니다. 실행 중 데이터를 외부 서비스로 보내지 않습니다.

[English README](README.md) · [기획서](docs/project-plan.md) · [계산 규칙](docs/conventions.md)

## 설치

```bash
python -m pip install "git+https://github.com/datakim/lendrisk.git@v0.1.0a1"
```

현재 버전은 `0.1.0a1` 알파입니다. PyPI에는 아직 배포하지 않았습니다.
Python 3.10 이상을 사용하며 기본 의존성은 NumPy와 pandas입니다.

## 첫 사용 예제

```python
from lendrisk import RevenueAdvance, RevenueShock, compare_scenarios, make_merchant_cashflows

# 합성 매출 경로를 사용합니다. 미래 매출을 예측한 결과는 아닙니다.
path = make_merchant_cashflows(days=365, daily_revenue=100_000, seed=42)
product = RevenueAdvance(
    principal=3_000_000,
    factor_rate=1.12,  # 총 회수목표액 = 3,360,000
    holdback_rate=0.10,  # 일별 적격 매출의 10% 회수
)
result = product.simulate(path, opening_cash=500_000)
print(result.summary())
print(compare_scenarios(product, path, [RevenueShock("매출 20% 감소", 0.8)]))
```

매출 시계열은 `date`, `revenue` 열을 가진 pandas DataFrame으로 전달합니다.
하루에 한 행을 사용하고, 매출이 없는 날도 명시적으로 0을 입력합니다.
`operating_cost` 열을 추가하면 현금 잔액과 부족 구간을 계산합니다.
금액의 통화와 단위는 사용자가 일관되게 맞춥니다.

## 구현 범위

- 기준일 이전 자료만 사용하는 매출·현금흐름 피처와 가맹점별 피처 테이블
- 고정 수수료를 포함한 매출 연동 상환, 기간별 최소 상환액과 누적 상환 목표
- 원리금균등 월별 상환 스케줄
- 매출 감소, 비용의 고정·변동 비중을 반영한 시나리오 비교
- AUC/Gini/KS, Brier/log loss, PSI, PD × LGD × EAD
- 실제 날짜에 따른 XIRR/XNPV와 선택적 OptBinning 연결

회수되지 않은 잔액은 분석 기간 끝에 그대로 남깁니다. 시뮬레이션은 계약상
지급이 실행된다고 가정하며, 현금 부족을 부도 판정으로 바꾸지 않습니다.
XIRR은 ACT/365 기준의 수학적 수익률이고 법정 APR을 계산하는 기능은 아닙니다.
사전 학습한 신용모형이나 자동 승인 정책은 포함하지 않습니다.

개발용 설치와 테스트 방법은 [기여 가이드](CONTRIBUTING.md), 함수와 입력 규칙은
[API 문서](docs/api.md)에 정리했습니다.
