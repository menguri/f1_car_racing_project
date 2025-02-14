# CarRacing DQN & Constrained RL Project

이 프로젝트는 **CarRacing-v2** 환경에서 DQN 및 Constrained DQN 접근을 통해 에이전트를 학습하는 예제입니다.  
병렬 환경, Shrink & Perturb, 맵 리셋 등 다양한 기법을 적용했으며,  
다음 논문 아이디어와 연관되어 있습니다:

1. [Constrained Reinforcement Learning]  
2. [Sample Efficient Reinforcement Learning (ICML 2024)]

---

## 주요 아이디어

1. **일반 DQN과 Constrained DQN**  
   - 일반 DQN(`module.py`의 `DQN` 클래스)과, 속도 초과/트랙 이탈을 페널티로 처리하는 Constrained DQN(`ConstrainedDQN` 클래스)을 구현.  
   - Constrained DQN은 라그랑주 승수를 동적으로 업데이트하여, 주행 속도가 너무 빠르거나 트랙에서 벗어나면 보상에 페널티를 부여함.

2. **병렬 환경(Parallel Environments)**  
   - 여러 개(`env_count`)의 CarRacing 환경을 동시에 구동하여, 한 에피소드 안에서도 다수의 경험을 빠르게 쌓도록 설계.  
   - 샘플 효율 증가, 학습 시간 단축 효과.

3. **Shrink & Perturb**  
   - 일정 주기(`TR`)마다 현재 네트워크 파라미터와 초기 파라미터를 혼합해(\(\alpha\)), 모델 파라미터가 과도하게 튀는 것을 방지.  
   - 안정적인 학습 유도.

4. **맵 리셋(Map Reset Interval)**  
   - `map_reset_interval`마다 CarRacing 맵을 변경(난수 시드 재설정)해, 특정 맵 구조에만 과적합되는 현상을 막음.

5. **실험 결과**  
   - `Score_performance.png`, `Time_performance.png` 등에 다양한 실험 결과(평균 점수, 에피소드 당 시간 등)가 시각화되어 있음.  
   - `CarRacing_eval_baseline.gif`, `CarRacing_eval_Constrained.gif`는 학습이 끝난 후 에이전트 주행 모습(애니메이션 GIF).

---

## 폴더 및 파일 구성

```bash
.
├── CarRacing_eval_baseline.gif        # (GIF) 일반 DQN 학습 후 주행 모습
├── CarRacing_eval_Constrained.gif     # (GIF) Constrained DQN 학습 후 주행 모습
├── maps/                              # CarRacing 맵 데이터(커스텀 또는 기본 맵)
├── model/                             # 학습된 모델 파라미터(.pth) 저장
├── model_train.py                     # 메인 학습 스크립트 (CMD 실행 시 진입점)
├── module.py                          # DQN, ConstrainedDQN 클래스 및 CNN 구조 정의
├── Performance_Experiment.ipynb       # 성능 실험 노트북 (시각화, 통계 분석)
├── Model_Training.ipynb              # Colab/Jupyter 환경에서 학습 시연 노트북
├── requirements.txt                   # 프로젝트 의존성 패키지 목록
├── requirements2.txt                  # (참고용) 다른 버전 의존성 리스트
├── Score_performance.png              # 평균 보상 등 점수 추이 시각화
├── Time_performance.png               # 학습 시간/성능 그래프
├── statistics.pkl                     # 학습 로그/통계(에피소드별 보상, 손실 등) Pickle 파일
├── submissions/                       # (선택) 대회 제출물 관리 폴더
├── wandb/                             # W&B 로깅(Weights & Biases) 폴더
├── sh_folder/                         # (선택) Shell 스크립트 등
├── __pycache__/                       # 파이썬 캐시
└── README.md                          # 본 문서
