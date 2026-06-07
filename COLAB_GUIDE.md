# Google Colab 최종 학습 및 평가 가이드 (ipynb 셀 실행용)

이 가이드는 코랩(Google Colab) 환경에서 최종 모델을 학습하고 결과물을 생성하기 위한 매뉴얼입니다.

---

## 0. 작업 폴더 구성 (수동 준비)

작업 폴더(final_project) 내에 아래 항목들이 포함되어 있어야 합니다. (`runs` 폴더는 학습 시작 시 자동 생성되므로 없어도 무방합니다.)

- **폴더**: `datasets/`, `models/`, `configs/`, `utils/`
- **파일**: `train.py`, `evaluate.py`, `predict.py`
- **데이터**: `data.zip` (구글 드라이브 내 위치 확인)

---

## 1. 코랩 실행 셀 (순서대로 실행)

### [셀 1] 구글 드라이브 마운트
```python
from google.colab import drive
drive.mount('/content/drive')
```

### [셀 2] 데이터 압축 해제
```python
# 기존 방식대로 /content/data 폴더에 바로 압축 해제
!unzip -q "/content/drive/MyDrive/final_project/data.zip" -d /content/data
```

### [셀 3] 최종 모델 학습 실행
```python
# 학습 실행 (가중치는 runs/exp_07_final_hetero_ens_5_6/best_model.pth 에 저장됨)
!python train.py --config configs/exp_07_hetero_ens5_6_res224_e50.yaml
```

### [셀 4] 보고서용 정밀 분석 (성적표 생성)
```python
# 학습된 가중치를 불러와 상세 분석(정확도, 혼동행렬) 수행
!python evaluate.py --config configs/exp_07_hetero_ens5_6_res224_e50.yaml \
                  --checkpoint runs/exp_07_final_hetero_ens_5_6/best_model.pth
```

### [셀 5] 제출용 예측 파일 생성
```python
# 최종 테스트 정답지(test_predictions.json) 생성
!python predict.py --config configs/exp_07_hetero_ens5_6_res224_e50.yaml \
                 --checkpoint runs/exp_07_final_hetero_ens_5_6/best_model.pth
```

### [셀 6] 결과물 드라이브 백업 (필수)
**주의**: 코랩 로컬(`runs/`)에 생성된 파일은 세션이 종료되면 사라집니다. 반드시 아래 명령어로 드라이브에 복사해야 합니다.
```python
# runs 폴더 전체를 구글 드라이브의 작업 폴더로 백업
!cp -r runs "/content/drive/MyDrive/final_project/"
```

---

## 2. 기타 유의 사항

- **검증 성능 확인**: 
    - [셀 3] 실행 중 화면에 출력되는 정확도는 실시간 확인용입니다.
    - [셀 4] 실행 후 생성되는 `val_results.json` 수치가 보고서에 작성할 최종 공식 성적입니다.
- **파일 동기화**: 코랩 내부(`/content`)에서 생성된 `runs` 폴더는 구글 드라이브에 자동으로 올라가지 않습니다. 반드시 [셀 6]을 실행하여 수동으로 백업해야 합니다.
- **경로 확인**: 구글 드라이브 내 `data.zip`이나 프로젝트 폴더 경로가 다를 경우 코드 내 경로를 수정해 진행해주세요.
