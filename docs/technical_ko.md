# PlanetFlow — 알고리즘 테크니컬 가이드

---

## 목차

1. [개요](#1-개요)
2. [Step 01 — SER Crop](#2-step-01--ser-crop)
3. [Step 02 — Lucky Stacking](#3-step-02--lucky-stacking)
4. [Step 03 — 품질 평가 및 윈도우 탐지](#4-step-03--품질-평가-및-윈도우-탐지)
5. [Step 04 — De-rotation 스태킹](#5-step-04--de-rotation-스태킹)
6. [Step 05 / 07 — 웨이블릿 선명화](#6-step-05--07--웨이블릿-선명화)
7. [Step 06 — RGB 합성](#7-step-06--rgb-합성)
8. [Step 08 — 애니메이션 GIF](#8-step-08--애니메이션-gif)
9. [Step 09 — 요약 그리드](#9-step-09--요약-그리드)
10. [공통 모듈: 디스크 감지](#10-공통-모듈-디스크-감지-find_disk_center)
11. [공통 모듈: 서브픽셀 정렬](#11-공통-모듈-서브픽셀-정렬)

---

## 1. 개요

이 문서는 GUI의 각 파라미터가 내부 알고리즘에서 **실제로 무슨 일을 하는지** 설명합니다. 파라미터 사용법은 `guide_ko.md`를 참조하고, 이 문서는 파라미터의 의미와 수학적 원리를 이해하고자 할 때 활용하세요.

### 소스 파일 구조

```
pipeline/
├── modules/
│   ├── planet_detect.py        # Step 01: 행성 감지 및 검증
│   ├── lucky_stack.py          # Step 02: Lucky Stacking 핵심 알고리즘
│   ├── quality.py              # Step 03: 화질 평가 및 윈도우 열거
│   ├── derotation.py           # Step 04: 디스크 감지, 자전 보정 워프, 스태킹
│   ├── satellite_tracker.py    # Step 04: 갈릴레이 위성/그림자 천체력 투영
│   ├── position_refinement.py  # Step 04: 선택형 CV 위치 보정 (위성/그림자)
│   ├── limb_darkening.py       # Minnaert 림브 감광 피팅 (플레이트 스케일 반경, 선택형 웨이블릿 게인)
│   ├── wavelet.py              # Step 05/07: À trous 웨이블릿 선명화
│   └── composite.py            # Step 06: RGB/LRGB 합성
├── steps/                      # Step별 실행 모듈 (ser_crop.py, lucky_stack.py, …,
│   │                           #   derotate_stack.py, satellite_composite.py, summary_grid.py)
├── data/
│   ├── np_ang_table.json            # 내장 Horizons NP.ang 테이블 (오프라인)
│   └── sub_observer_lat_table.json  # 내장 Horizons 관측자 직하점 위도 테이블 (오프라인)
└── config.py                   # 전역 설정 (dataclass 기반)
```

> **GUI 기본값과 `config.py` 기본값**: 일부 `config.py` dataclass 기본값은 GUI가 실제로 넘기는 값과 다릅니다(예: `per_ap_selection`, `use_ncc`, `auto_params`). 아래 표는 **GUI가 실제로 사용하는 값**을 기준으로 하며, dataclass 기본값과 다르면 따로 표기합니다.

---

## 2. Step 01 — SER Crop

**소스**: `pipeline/modules/planet_detect.py`, `pipeline/steps/ser_crop.py`

```
입력 프레임
    │
    ▼
8비트 그레이스케일 변환
    │
    ▼
GaussianBlur(5×5)  ← 노이즈 억제
    │
    ▼
Triangle 임계값 (Zack 1977)
    │
    ▼
최대 연결 컴포넌트 (8-연결) 추출
    │
    ▼
4단계 검증 → 실패 시 프레임 거부
    │
    ▼
바운딩박스 중심으로 정사각형 ROI 크롭
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **ROI 크기 (px)** | 448 | `get_cropped_frame()`의 출력 정사각형 크기. `round()`로 정수 변환하여 0.5픽셀 체계적 편향 방지. 범위 밖 픽셀은 0(검정)으로 채움 |
| **최소 원반 지름 (px)** | 50 | 4단계 검증의 마지막 기준. `max(bw, bh) < min_diameter`이면 프레임 거부 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `padding` | 10 px | 경계 검사: 행성 바운딩박스가 이미지 엣지에서 이만큼 이상 떨어져야 함 |
| `aspect_ratio_limit` | 0.2 | 종횡비 검사: `min(w,h)/max(w,h) ≥ 1 − 0.2 = 0.8` 이어야 통과 |
| `straight_edge_limit` | 0.5 | 직선 엣지 검사: 바운딩박스 4변 중 50% 이상 점등된 변이 있으면 거부 |
| GaussianBlur 커널 | 5×5 | 임계값 전 노이즈 억제 |
| `core_percentile` | 60 | 고리 행성 처리 (아래 참고) |

### 고리 행성 (토성)

토성의 고리는 이진화 마스크에서 원반과 한 덩어리로 잡혀, 바운딩박스가 세로보다 가로가 2배 가까이 넓어집니다. 그대로 두면 종횡비 검사에서 항상 탈락합니다. 그래서 종횡비/직선 엣지 검사 전에 블롭 내부 밝기의 `core_percentile`(60번째) 백분위수로 다시 이진화해 더 밝은 원반 코어만 남기고, 이 두 검사는 코어에만 적용합니다. 경계·최소 지름·중심은 여전히 원반+고리 전체 범위를 사용합니다.

일부 필터는 이 대비가 반대입니다(예: 토성 CH4 밴드는 원반이 고리보다 어둡습니다). 필터 이름으로 예외 처리하는 대신, 그럴듯한 둥근 코어를 분리하지 못하면 "밝기로는 이 프레임의 형태를 검증할 수 없다"고 보고 해당 프레임의 종횡비/엣지 검사를 건너뜁니다.

### Triangle 자동 임계값

OpenCV `THRESH_TRIANGLE` 플래그로 구현합니다. 히스토그램의 최고 피크에서 가장 먼 최솟값을 임계값으로 결정합니다. 노출값이나 행성 크기에 관계없이 안정적으로 동작합니다.

### 바운딩박스 중심 크롭

목성은 벨트·대적점으로 밝기가 불균일합니다. 밝기 가중 무게중심은 밝은 구조물 쪽으로 편향이 생기므로, 행성 중심화 편향 방지를 위해 **바운딩박스 중심** `(x + w/2, y + h/2)`을 사용합니다.

---

## 3. Step 02 — Lucky Stacking

**소스**: `pipeline/modules/lucky_stack.py`, `pipeline/steps/lucky_stack.py`

GUI는 Lucky Stacking을 항상 `per_ap_selection=True`, `use_ncc=True`로 실행합니다. 이 경우 스태킹은 **AP별 독립 럭키 스태킹**(`_per_ap_independent_stack()`)으로 처리됩니다. 정렬점(AP)마다 *그 위치의* 선명도로 최적 프레임을 따로 고르므로, 디스크의 영역마다 서로 다른 프레임 조합으로 만들어질 수 있습니다.

```
SER 입력 파일
    │
    ▼
[1] 프레임 품질 점수 (log_disk 지표, 2프레임마다 계산 + 보간)
    │
    ▼
상위 top_percent% 프레임 선택 → selected_indices
    │
    ▼
[2] 기준 프레임 생성
    품질 75번째 백분위 순위를 중심으로 reference_n_frames(50)장
    → 그중 첫 프레임에 위상 상관 정렬 → 평균 스택
    → find_disk_center() → (cx, cy, disk_radius)
    → 토성만: 원반+고리 블롭 타원 피팅 (고리 인식 AP 범위)
    │
    ▼
[3] AP 격자 생성 (균일 격자 또는 AS!4 탐욕 PDS 3레이어)
    │
    ▼
[4] Pass 1 — 선택된 모든 프레임에 대해:
    전역 정렬: limb_center_align() (이동량이 0.5 × disk_radius를 넘으면
    위상 상관으로 대체) → 바이큐빅 서브픽셀 이동
    AP별 국소 점수: 각 AP 패치 내부의 LoG 분산
    │
    ▼
[5] Pass 2 — 각 AP마다 독립적으로:
    이 AP의 점수 기준 상위 per_ap_stack_sub_percent(50%) 프레임 선택
    프레임별 국소 이동: Hann 윈도우 NCC + QSF 서브픽셀 피크
    이동된 위치에서 2 × ap_size 패치 추출
    가중 평균 (가중치 = 국소 점수 ^ quality_weight_power)
    │
    ▼
[6] 모든 AP 패치를 가우시안 마스크(σ = 2·ap_size / 3)로 합산
    → 누적 마스크 가중치로 정규화
    │
    ▼
[7] 주파수 도메인 가우시안 롤오프 (σ_f = 0.20)
    │
    ▼
n_iterations = 2이면: 결과를 기준 프레임으로 → [4]–[7] 반복
    │
    ▼
TIF 출력
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **상위 프레임 비율 (%)** | 25 | `top_percent = 0.25`. 품질 점수 상위 N% 프레임만 스태킹에 사용. `n_select = max(min_frames, round(n_frames × top_percent))` |
| **AP 크기 (px)** | 64 | AP 기본 크기 s. 균일 격자: 간격 = s ÷ 2. AS!4 PDS: 레이어 s, round(s×1.5/8)×8, s×3. 각 AP는 2s × 2s 패치를 합성 |
| **반복 횟수** | 1 | `n_iterations`. 2이면 1차 스택이 2차 패스의 기준 프레임이 되어 AP별 NCC 이동 추정의 기준 SNR이 높아지고, 시간은 약 2배. 실제로는 1회와 차이가 미미함 |
| **SER 동시 처리 수** | 1 | 동시에 처리할 SER 파일 수. 0 = 자동 (CPU 코어 ÷ 4). 각 SER은 `n_workers ÷ N_SER` 스레드 사용. SER당 RAM 약 950 MB |
| **AS!4 AP 그리드** | 끔 | 끔 = 균일 격자. 켬 = AS!4 탐욕 Poisson-disk 샘플링. **세션당 한 번** 기준 SER에서 생성해 모든 SER이 공유 (각 SER의 디스크 오프셋만큼 이동) |
| **디베이어** | 켬 | 컬러 카메라 전용. Bayer 프레임을 서브픽셀 이동 **전에** RGB로 디베이어 (Bayer 모자이크를 이동하면 색 사이트가 섞임). NCC 이동 추정은 Bayer 휘도로 수행 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `score_metric` | `"log_disk"` | 프레임 품질 지표 (AS!4 *lapl3*와 동일). config에서 `"local_gradient"`, `"laplacian"`, `"gradient"`도 선택 가능 |
| `reference_midpoint_percentage` | 75 | 기준 프레임 구간을 75번째 백분위 순위에 맞춤 (드문 행운 프레임보다 "꾸준히 좋은" 프레임 사용 — AS!4 기본값) |
| `reference_n_frames` | 50 | 기준 프레임으로 평균 낼 프레임 수 |
| `per_ap_stack_sub_percent` | 0.5 | 각 AP가 선택 풀에서 스태킹하는 비율 (0.5 × 25% ≈ AP마다 전체 상위 12.5%) |
| `quality_weight_power` | 3.0 | 프레임별 패치 가중치 = 국소 점수³ |
| `ap_search_range` | 20 px | AP 국소 이동이 이보다 크면 버림 (이동 → 0) |
| `ap_min_brightness` | 0.196 | 패치 평균 밝기가 이보다 낮으면 AP 제외 (AS!4 Min Bright 50/255) |
| `ap_min_contrast` | 0.01 | 균일 격자만: 패치 RMS 대비가 이보다 낮으면 AP 제외 |
| `fourier_rolloff_sigma` | 0.20 | 정규화 주파수 단위 출력 저역 통과 σ (0 = DC, 0.5 = 나이퀴스트) |
| `remap_interpolation` | `INTER_CUBIC` | 서브픽셀 이동 보간 방식 |
| `sigma_clip` | 끔 | 최종 스택을 기준으로 모든 프레임을 다시 워프하고 `sigma_clip_kappa`(2.0) σ를 넘는 픽셀을 제외하는 선택형 후처리. config/세션 전용 — GUI 컨트롤 없음 |

### 프레임 품질 평가 (`log_disk`)

AS!4의 *lapl3*와 같은 지표입니다. AS!4 프레임 순위와의 Spearman 상관 0.74.
```
mask  = (frame / max) > 0.25
score = var(Laplacian(GaussianBlur(frame, σ=3.0)))  on mask
```
Pass 1에서는 같은 LoG 분산을 AP 패치마다 계산해 프레임을 국소적으로 순위 매깁니다.

### AP 격자 배치

**균일 격자** (AS!4 AP 그리드 끔): `ap_size ÷ 2` 간격으로 AP 중심을 배치합니다. 중심이 디스크 안에 있고, 패치가 이미지 안에 들어가며, 밝기·대비 하한을 통과한 AP만 남깁니다.

**탐욕 PDS** (AS!4 AP 그리드 켬): 래스터 스캔으로 독립적인 3개 레이어를 생성합니다.

| 레이어 | AP 크기 (s=64) | 최소 AP 간격 |
|-------|----------------|--------------|
| Layer 1 | 64px | `round(64 × 35/64)` = 35px |
| Layer 2 | 96px (`round(64×1.5/8)×8`) | `round(96 × 35/64)` = 52px |
| Layer 3 | 192px (`64×3`) | `round(192 × 35/64)` = 105px |

AP 채택 조건: ① 디스크 내부, ② 패치 평균 밝기 ≥ 0.196 (적분 이미지, O(1)), ③ 같은 레이어에 이미 채택된 AP와 최소 간격 충족. AS!4 실측 대비 일치율 96–100%.

### 고리 인식 AP 범위 (토성)

토성(`horizons_id == "699"`)이면 자동으로 적용되며, 별도 토글은 없습니다.

- **허용 영역**: 빛나는 블롭 전체(원반+고리)에 타원을 피팅합니다(Otsu − 10%, 7×7 closing, 최대 윤곽선, `cv2.fitEllipse`). AP는 원반 반경 안뿐 아니라 이 타원 안 어디에나 놓일 수 있습니다. 그 밖의 모든 곳에서는 `find_disk_center()`가 고리를 제거하고 구한 원반 반경을 그대로 씁니다.
- **AS!4 PDS**: 레이어마다 원반 전용 패스를 **그대로** 먼저 실행한 뒤, 같은 간격 격자에 대해 고리 전용 후보를 두 번째 패스로 추가합니다. 그래서 검증된 AS!4 원반 배치가 바뀌지 않습니다. (이전 단일 패스 버전은 림브 근처 후보가 래스터 순서에서 먼저 채택되면서 원반 AP의 최대 ~99%가 다른 위치로 바뀌었습니다.)
- **AP별 합성**: 고리 AP의 2s × 2s 합성 마스크는 원반 안쪽 깊숙이 닿을 수 있습니다. 그래서 원반 반경 안에 들어가는 부분은 마스크를 0으로 만들어(3 px 페더) 고리 내용이 원반 픽셀에 섞이지 않게 합니다. 원반 AP는 바뀌지 않습니다.
- **세션 공유 AP**: 고리 타원을 세션 AP와 함께 넘기므로, SER별 재필터링에서 고리 AP가 빠지지 않습니다.
- **미리보기**: Step 02 미리보기에 고리 타원이 자홍색으로 그려지고 격자 라벨에 `+Ring`이 붙습니다.

### 대체 스태킹 경로 (config.py 전용)

`config.py`를 직접 수정해야만(`per_ap_selection=False`) 쓸 수 있는 경로이며, GUI는 이 경로를 선택하지 않습니다. 따라서 관련 파라미터(`use_tps`, `fourier_quality_power`)도 config 전용이며, Step 02 패널에는 Warp 방식·Fourier 품질 가중치 항목이 더 이상 없습니다.

- **Fourier 도메인 품질 가중 스태킹** (`use_fourier_quality=True`, Mackay 2013, arXiv:1303.5108): 전역 정렬된 각 프레임이 공간 주파수마다 `|F_n(f)|^power × F_n(f)`로 기여하고, `Σ|F_n(f)|^power`로 정규화한 뒤 0.20 롤오프를 적용합니다. 전역 정렬만 하고 AP별 워프는 없습니다.
- **워프 필드 스태킹**: AP별 Hann 윈도우 이동량(위상 상관 또는 NCC + QSF)을 가우시안 커널 회귀(`σ = ap_step × ap_sigma_factor`, 0.9) 또는 TPS로 전체 해상도 워프 필드로 보간합니다. KR은 C∞ 연속 필드를 만듭니다. Delaunay는 C⁰ 메시라서 삼각형 경계의 꺾임이 수천 장 동안 누적되고, 웨이블릿 선명화 후 눈에 보이게 됩니다. 토성에서는 원반 AP와 고리 AP를 두 개의 독립 필드로 회귀한 뒤 디스크 경계에서 페더링해 합칩니다.

**병렬 처리 모델** (예: 32코어, SER 병렬 4):
```
전체 스레드 예산 = n_workers (기본값: 모든 코어)
SER 레벨:   SER 파일 4개 동시 처리
프레임 레벨: SER마다 n_workers ÷ 4 = 8 스레드
```

---

## 4. Step 03 — 품질 평가 및 윈도우 탐지

**소스**: `pipeline/modules/quality.py`, `pipeline/steps/quality_assess.py`

```
Step 02 TIF 파일 목록
    │
    ▼
TIF 이미지마다:
    Otsu 임계값 (−15% 여유) → 최대 컴포넌트 → 디스크 마스크
    GaussianBlur(σ=1.2) → 노이즈 제거
    라플라시안 분산 (×0.5) + Tenengrad (×0.3) + 정규화 분산 (×0.2)
    → 종합 raw_score
    │
    ▼
필터별 min-max 정규화 → norm_score ∈ [0, 1]
    │
    ▼
norm_score < 최소 품질 임계값인 프레임 제거
    │
    ▼
앵커 프레임(IR이 있으면 IR, 없으면 첫 필터)마다 후보 윈도우 하나:
    윈도우 = 앵커 시각 ± window_minutes / 2
    필터마다: 하단 σ-클리핑 (평균 − 1.5σ) → quality_post × snr_factor × stability
    │
    ▼
필터 간 기하 평균 → window_quality
    │
    ▼
모든 윈도우를 시간순 정렬 (겹침 포함)
    │
    ▼
출력: windows.json / *_ranking.csv
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **윈도우 (프레임 수)** | 3 | 필터 사이클 수 단위의 윈도우 길이. `window_minutes = 프레임 수 × cycle_minutes`, 각 앵커 프레임을 중심으로 함 |
| **필터 사이클 (초)** | 225 | 필터 한 사이클(IR→R→G→B→CH4→IR) 시간. 윈도우 길이와, `snr_factor`에 쓰는 기대 프레임 수 `n_expected = window_minutes / cycle_minutes`를 정함. 컬러 카메라 모드에서는 "단일 촬영 시간(초)"으로 라벨이 바뀌고 기본값 45초 |
| **최소 품질 임계값** | 0.05 | `norm_score < threshold`인 프레임을 윈도우 탐색 **전에** 제거. 0.0이면 모든 프레임 포함 |

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| 라플라시안 가중치 | 0.5 | 종합 점수 중 라플라시안 분산 비중 |
| Tenengrad 가중치 | 0.3 | 종합 점수 중 Tenengrad(Sobel 제곱 평균) 비중 |
| 정규화 분산 가중치 | 0.2 | 종합 점수 중 `var/mean` 비중 |
| 노이즈 제거 σ | 1.2 px | 선명도 지표 계산 전 가우시안 블러. 노이즈만 많고 흐린 프레임이 높은 점수를 받는 것을 방지 |
| σ-클리핑 기준 | 1.5σ, 하단만 | 윈도우 내에서 `norm_score < 평균 − 1.5σ`인 프레임 제외. 유난히 좋은 프레임은 잘리지 않음 |

### 윈도우 품질 계산

```
# 필터별
quality_post = mean(포함된 프레임의 norm_score)
snr_factor   = min(1.0, √(n_included / n_expected))
stability    = 1 / (1 + CV)          CV = 포함된 프레임의 std/mean
filter_quality = quality_post × snr_factor × stability

# 전체 필터
window_quality = (∏_f  filter_quality_f) ^ (1 / 필터 수)
```

**기하 평균을 쓰는 이유**: 한 필터라도 품질이 아주 나쁘면 윈도우 품질이 크게 떨어집니다. 좋은 합성 이미지를 만들려면 모든 필터가 최소 기준을 넘어야 합니다.

필요한 필터 중 하나라도 윈도우 안에 프레임이 없으면 그 윈도우는 버립니다. `windows.json`의 `rotation_degrees`는 목성 기준 ~0.6°/분으로 대략 계산한 표시용 값이며, Step 04는 실제 자전량을 타임스탬프와 설정된 자전 주기로 계산합니다.

**윈도우 선택은 여기서 하지 않습니다**: Step 03은 Step 04–06이 세션 전체를 처리할 수 있도록 모든 윈도우를 반환합니다. 상위 N개 선택과 겹침 제외는 Step 09에서 합니다.

---

## 5. Step 04 — De-rotation 스태킹

**소스**: `pipeline/steps/derotate_stack.py`, `pipeline/modules/derotation.py`, `pipeline/steps/satellite_composite.py`

```
Step 02 TIF + Step 03 윈도우
    │
    ▼
세션 단위 (1회):
    ├─ pole_pa: 우선 필터(IR > R > G > B > CH4)의 모든 프레임에서
    │   누적 측정한 이미지 공간 벨트 축 각도
    ├─ derot_flip: 실제 피처 이동에 대한 순방향 예측 NCC로 워프 방향 결정
    ├─ flip_pole_axis: 3D 재투영의 부호 모호성, 같은 방식으로 결정
    ├─ tracker_flip_ns: 카메라 남북 방향 (위성 추적기 전용)
    └─ derot_confidence: 고역 통과 NCC 스윕 (진단용)
    │
    ▼
윈도우마다:
    ├─ NP.ang (추적기 전용) + 관측자 직하점 위도 B (내장 테이블 → 캐시 → Horizons)
    └─ 필터마다:
        ├─ 기준 프레임 = 윈도우 중심 시각에 가장 가까운 프레임
        ├─ 기준 프레임에서 디스크 감지; 필터 간 보정
        │   (공유 반경 / 형태 / 위치 — 아래 참고)
        ├─ 프레임마다:
        │   ├─ 워프 전 디스크 중심 + 배율 → 원본 프레임에 적용
        │   ├─ 3D 편평 회전타원체 재투영 워프 (윈도우 중심 기준 Δt)
        │   └─ 토성: 고리 가림 가중치로 고리 픽셀의 대기 워프 억제
        ├─ 정렬 대체 경로 + 이상 이동량 제외
        └─ 품질 가중 평균 (가중치 = norm_score ^ stack_weight_power)
    │
    ▼
자전 보정된 결과물에서 pole_pa 재측정 (추적기용)
    │
    ▼
선택: 위성/그림자 합성 → 필터별 TIF 덮어쓰기
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **최소 품질 임계값** | 0.05 | `norm_score < threshold`인 프레임을 스태킹 전에 제외. 모든 프레임이 제외될 상황이면 해당 필터는 전부 사용 |
| **스택 가중치 지수** | 1.0 | `weights = norm_score ^ power`. 1.0 = 선형 혼합. 한 윈도우 안에서 프레임 품질 편차가 클 때 2.0–4.0으로 올리면 최상 프레임 비중이 커짐 (실제 토성 데이터로 검증, 범용 기본값은 아님) |
| **밝기 정규화** | 끔 | 각 프레임의 디스크 중앙값을 기준 프레임에 맞춤. 실제 밝기 정보를 잃으므로 얼룩 아티팩트가 보일 때만 사용 |
| **위성 합성 (다중-속도 역회전)** | 끔 | 위성 추적기와, 디스크 위에 있는 모든 갈릴레이 위성·그림자의 합성을 켬 (아래 참고) |
| **CV 기반 위치 보정** | 끔 | 위성 합성이 켜져 있을 때만 활성화 가능. 천체력으로 예측한 위치를 윈도우마다 신뢰도 조건부로 CV 보정 (아래 참고). 옆의 상태 라벨에 CH4 데이터 유무가 표시됨 |

**전역 설정에서 지정** (행성 프리셋): 자전 주기, **De-rotation 워프 배율** (모든 행성 기본 1.0), **실제 극/적도 반경비** (목성 0.9351, 토성 0.9021, 화성 0.9941, 천왕성 0.9771, 해왕성 0.9829).

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `use_true_reprojection` | 항상 켬 | GUI가 3D 재투영 워프를 강제. 선형 워프는 코드에서만 사용 가능 |
| `req_px` (R) | `disk_radius × 1.05` | 워프 구의 5% 여유, 림브 특이점 회피 |
| `_interp_feather_px` | 12.0 px | CUBIC(내부) / LINEAR(림브) 보간 혼합, 림브 안쪽 12 px에서 코사인 페이드 |
| `_RADIUS_SHARE_REL_TOL` | 8% | 필터 반경이 윈도우 중앙값의 8% 이내일 때만 그 값으로 맞춤 |
| `_SCALE_MIN` / `_SCALE_MAX` | 0.95 / 1.05 | 프레임별 배율 보정 제한 범위 |
| `_OUTLIER_ABS_PX` | 5.0 px | 이상 이동량 제외의 절대 하한 |

### pole_pa — Horizons가 아니라 이미지에서 측정

워프의 이동 방향을 정하는 `pole_pa_deg`는 **이미지 공간** 각도입니다. `auto_detect_equator_pa()`는 내부 디스크(r < 0.75R)에서 웨이블릿으로 선명화한 두 이미지(2–8 px 벨트 경계, 8–32 px 벨트 폭)로 그래디언트 각도 히스토그램을 만듭니다. 가장 우세한 벨트 경계 그래디언트 방향이 투영된 극축이고, 이를 90° 돌려 이동 축을 얻습니다. 더 굵은 웨이블릿 레벨은 방사형 림브 감광 그래디언트를 잡아 각도를 ~10° 치우치게 하므로 제외합니다.

세션 값은 우선 필터의 **모든 프레임을 하나의 히스토그램에 누적**한 뒤 argmax를 취합니다. 벨트 그래디언트 방식이 실패한 프레임은 `equator_pa_from_disk_ellipse()`로 대체합니다.

**NP.ang**(Horizons quantity 17, 천구 북극 방향각)은 워프에 **쓰이지 않습니다**. 위성 추적기에서 카메라 회전 `θ_cam = pole_pa + NP.ang`으로만 쓰입니다. 두 각도는 서로 다른 좌표계의 값이라 바꿔 쓸 수 없습니다.

스태킹 후에는 **자전 보정된 결과물에서 `pole_pa`를 다시 측정**해 추적기에 넘깁니다. 추적기는 자전 보정된 이미지 위에 그리는데, 실측 결과 그 이미지의 실제 방향이 워프 전 값과 ~0–3.5° 차이 났기 때문입니다.

### 워프 방향 자동 감지

- **`derot_flip`** (`auto_detect_ns_flip()`): 초기 프레임을 두 방향으로 각각 앞으로 워프해 보고, 나중 프레임과 더 잘 맞는(내부 디스크 NCC) 쪽을 고릅니다. 북쪽이 위인 카메라와 남북만 뒤집힌 카메라 모두 이미지상 피처가 같은 방향으로 움직이므로 보통 `False`가 나옵니다. 위성 추적기에 북쪽 방향을 알려주지는 **못합니다**.
- **`flip_pole_axis`** (`auto_detect_pole_axis_flip()`): 같은 순방향 예측 시험을 3D 재투영 자체의 부호 모호성에 적용합니다. 이 모호성은 관측자 직하점 위도 B를 모델링할 때만 생깁니다. 판별이 안 되면(|Δcorr| < 0.001) `False`로 둡니다.

둘 다 GUI 설정이 아니며, 세션당 한 번 자동 감지합니다.

### 3D 편평 회전타원체 재투영 워프

**소스**: `spherical_derotation_warp_3d()`

출력 픽셀마다 워프는 다음을 수행합니다.

1. **역투영**: 픽셀을 편평 회전타원체(적도 반경 `req_px`, 극 반경 `req_px × true_polar_equatorial_ratio`) 위로 되돌려 행성 중심 위도/경도를 구합니다. 이 타원체는 관측자 직하점 위도 B만큼 기울고 `pole_pa`만큼 회전되어 있습니다.
2. **경도 이동**: `Δλ = (Δt / 자전 주기) × 2π`만큼 이동합니다. Δt는 윈도우 중심 시각 기준입니다.
3. **재투영**: 정사영으로 다시 투영해 그 위치에서 원본 프레임을 샘플링합니다(디스크 내부 CUBIC, 림브 근처 LINEAR, 12 px 코사인 혼합).

이전 선형 워프(`drift = scale × Δλ × depth(x, y)`, 한 축 방향)와 달리 B를 명시적으로 모델링합니다. 그래서 피팅된 타원의 겉보기 종횡비(B에 의한 원근 단축이 이미 섞임)가 아니라 행성의 **실제** 물리적 편평률을 써야 합니다. 회전이 기하학적으로 정확하므로 `warp_scale`은 1.0으로 둬야 하며, 다른 값이면 타임스탬프·주기·방향 같은 다른 문제가 있다는 경고가 출력됩니다.

**토성 고리** (`has_rings`, `horizons_id == "699"`): 고리는 대기와 함께 회전하지 않습니다. `compute_ring_occlusion_weight_3d()`가 토성의 실제 고리 반경(C 고리 안쪽 1.239 Req, A 고리 바깥쪽 2.269 Req), B, `pole_pa`로 연속 [0, 1] 가중치를 만들어, **앞쪽** 고리가 행성 본체를 가리는 곳을 표시합니다. 그 위치에서는 워프된 픽셀 값을 프레임 자체의 워프 전 값 쪽으로 혼합하므로, 고리 픽셀에는 대기 자전이 적용되지 않습니다.

### 필터 간 디스크 기하 보정

각 필터는 자기 기준 프레임에서 디스크를 감지합니다. 그 뒤 필요한 경우에만 세 가지 보정을 적용합니다.

- **공유 반경** (`resolve_shared_radius()`): 한 윈도우 안에서도 필터별 장반경이 Otsu 노이즈로 1–3 px씩 다릅니다. 그대로 두면 필터마다 워프 유효 영역이 조금씩 다른 반경에서 끊겨 림브 색 번짐으로 나타납니다. 이미 윈도우 중앙값의 8% 이내인 필터**만** 중앙값으로 맞추고, 실제로 크기가 다른 필터(예: CH4 흡수 밴드)는 그대로 둡니다.
- **공유 형태** (`resolve_shared_shape()`): 자체 피팅으로 편평률/방향을 믿을 수 있게 측정하지 못하는 필터(예: 원반이 고리보다 어두운 토성 CH4)는 다른 필터의 종횡비와 각도를 빌려 씁니다.
- **필터 위치** (`resolve_filter_pose()`): 필터 자체 감지가 완전히 실패한 경우에만, 다른 필터 프레임에 대해 중심을 정합합니다.

### 워프 전 정렬과 배율 보정

시상 때문에 디스크는 프레임마다 ~0–2 px씩 흔들립니다. 프레임마다 **원본**(워프 전) 휘도 이미지에서 `find_disk_center()`를 실행합니다.

- 이동: `(ref_cx − cx_i, ref_cy − cy_i)`. 두 성분이 모두 15 px 이내일 때만 사용.
- 배율: `자기 필터 기준 반경 / radius_i`. 피팅 신뢰도가 있고 [0.95, 1.05] 범위일 때만 사용.

이 보정은 **워프 전에 원본 프레임에** 적용합니다(`apply_shift_and_scale()`). 워프는 가정한 중심을 기준으로 정의된 비강체·깊이 의존 변형이라, 먼저 워프하고 나중에 이동하는 것과 결과가 다릅니다. 워프 후에 이동을 적용하던 방식은 실제 토성 데이터에서 중앙값 0.58 px(최대 1.38 px)의 잔여 오차를 남겼습니다.

워프 후 중심 감지는 일반 보정으로 쓰지 않습니다. 워프가 림브 근처 벨트를 옮겨 감지된 중심이 워프 방향으로 끌려가고, 결과적으로 자전 보정을 일부 되돌리기 때문입니다.

**대체 경로** (`derotation_log.json`의 `align_method`):

| `align_method` | 출처 | 조건 |
|---|---|---|
| `"reference"` | 이동 없음 | 기준 프레임 자신 |
| `"pre_warp_center"` | 워프 전 디스크 중심 (+ 배율) | 일반적인 경우 |
| `"limb_center"` | 워프 후 림브 중심 | 워프 전 감지 실패 |
| `"phase_correlate"` | 위상 상관 | `limb_center`가 이동 0을 반환 |

자체 형태 감지를 믿을 수 없는 필터(예: 토성 CH4)는 워프 전 이동을 디스크 영역 위상 상관으로 구하며, 배율 보정은 하지 않습니다.

**이상 이동량 제외**: 기준 프레임이 아닌 프레임의 정렬 이동량이 윈도우 중앙값에서 `max(1.5, 4 × robust σ)` **그리고** 5 px보다 더 떨어져 있으면 그 프레임을 뺍니다. 예를 들어 다른 프레임은 ~1 px인데 CH4 프레임 하나만 ~8 px가 필요한 경우처럼, 조용히 실패한 정합이 디테일을 번지게 하는 것을 막습니다.

### 자전 보정 신뢰도 측정 (NCC 스윕)

**소스**: `pipeline/steps/derotate_stack.py` → `_measure_derot_confidence()`

"설정된 warp_scale을 적용했을 때 벨트 구조가 자전 보정을 얼마나 뒷받침하는가"를 보는 **진단용** 지표이며, `warp_scale`을 바꾸지 않습니다.

1. 세션에서 가장 긴 윈도우(최소 자전 3°)의 가장 이른 프레임과 가장 늦은 프레임을 고릅니다.
2. `scale`을 0.50–1.20 범위에서 13단계로, 설정값을 더해 스윕합니다.
3. 이른 프레임을 전체 Δt만큼 앞으로 워프하고, 내부 디스크(r ≤ 0.7 × semi_a)에서 늦은 프레임과의 고역 통과 NCC를 계산합니다.
4. 설정 배율에서의 NCC를 `derot_confidence`로 기록합니다.

고역 통과 필터는 먼저 σ = 30 px 가우시안 블러를 뺍니다. 이것이 없으면 부드러운 림브 감광 그래디언트가 NCC를 지배해 항상 가장 작은 배율이 이깁니다.

추정 피크 배율은 `est. peak`로 진단용으로만 기록합니다. 실제 데이터에서 NCC–배율 곡선은 노이즈가 지배적이고 진짜 피크가 없는 것으로 확인됐으므로, 이 값으로 `warp_scale`을 정하면 안 됩니다.

**경고 기준**: NCC가 0.80 미만이면 Step 04 단독 실행 후 또는 전체 실행 후 GUI 경고가 뜨며, Step 03 윈도우를 줄이라고 권장합니다.

### Horizons 조회

`query_horizons_np_ang()`과 `query_horizons_sub_observer_lat()`은 같은 우선순위를 씁니다.

1. **내장 테이블** (오프라인): `pipeline/data/np_ang_table.json`, `sub_observer_lat_table.json` — 목성(599), 토성(699), 화성(499)의 2016–2036년 값, 선형 보간 (NP.ang은 360°/0° 경계 처리 포함).
2. **사용자 캐시**: `~/.astropipe/horizons_cache.json`.
3. **실시간 Horizons API**: 내장 범위 밖이거나 사용자 지정 행성일 때.

관측자 직하점 위도는 3D 재투영이나 고리 처리에 필요할 때만 조회합니다(GUI가 3D 재투영을 강제하므로 사실상 항상). 조회에 실패하면 경고와 함께 0.0°를 씁니다.

### 위성/그림자 합성

**소스**: `pipeline/steps/satellite_composite.py` → `_apply_satellite_composite()`

**위성 합성**을 켜면, 윈도우 중심 시각에 **디스크 위에** 있다고 예측되는 모든 갈릴레이 위성(이오, 유로파, 가니메데, 칼리스토)과 위성 그림자를 캡슐 형태 마스크와 Poisson 그래디언트 도메인 블렌딩으로 각 필터의 자전 보정 TIF에 합성합니다. 디스크 밖의 위성은 건너뜁니다.

```
필터 TIF마다:
    이 필터의 디스크 중심 감지 (disk_cx, disk_cy)
        │
        ▼
    t_center 시점의 기준 위성 + 그림자 위치 조회 (Skyfield)
    — 이 필터의 디스크 좌표계 기준
        │
        ▼
    (선택) CV 위치 보정
        │
        ▼
    프레임별 위치 조회; 위성/그림자가 기준 위치에 오도록
    원본 프레임을 평행이동 스태킹
        │
        ▼
    캡슐 마스크 안에서 Poisson 방정식 풀기 → 행성에 접합
        │
        ▼
    필터 TIF를 제자리 덮어쓰기
```

#### 필터별 디스크 좌표계

기준 위치는 필터마다 **그 필터 자신의 디스크 좌표계**로 계산합니다. 자전 보정된 디스크는 필터마다 절대 픽셀 위치가 조금씩 다를 수 있고, Step 06의 `align_channels()`가 나중에 디스크를 맞추려고 채널을 이동합니다. 위성의 **디스크 기준** 오프셋이 모든 필터에서 같으므로 Step 06 정렬이 디스크와 위성을 함께 옮기고, 위성은 모든 합성 이미지에서 같은 자리에 놓입니다.

#### 블렌딩 — 캡슐 마스크 + Poisson

```
alpha(x,y) = exp(−min_dist_to_trajectory(x,y)² / (2 × sigma_perp²))
sigma_perp  = apparent_radius_px × coverage_scale     (coverage_scale = 1.5)

∇²result = ∇²sat_stack    (마스크 내부)
result    = planet         (마스크 경계 — Dirichlet 경계조건)
```

마스크 폭은 위성의 겉보기 크기를, 길이는 실제 이동 경로를 따르므로 긴 통과 중에도 좁게 유지됩니다. 그래디언트 구조만 옮기고 DC 레벨은 행성 경계에서 오므로, 필터별 DC 잔차가 색 편향을 만들지 않습니다.

#### plate_scale

`plate_scale = ang_radius_geometric / r_ref`이며, `r_ref`는 세션 디스크 반경의 중앙값입니다. 각 프레임의 원래 `find_disk_center()` 반경은 먼저 `refine_disk_radius_for_plate_scale()`로 보정합니다. Otsu 임계값은 실제 기하학적 가장자리보다 한참 안쪽에서 림브 감광 곡선과 만나므로 반경을 ~6–18% 작게 잡습니다. 디스크 자체의 방사 프로파일에 Minnaert 법칙(`limb_darkening.py`)을 피팅해 기하학적 반경을 복원합니다. 피팅이 실패하거나 비율이 [1.0, 1.3]을 벗어나면 원래 반경을 씁니다. 보정된 반경은 플레이트 스케일과 위치 계산에**만** 쓰고, 워프와 페더링은 원래 반경을 유지합니다.

위성/그림자 픽셀 위치는 천체력 투영 결과 그대로입니다. 플레이트 스케일이나 디스크 중심을 CV로 피팅하던 전역 보정은 2026-08-31에 제거됐습니다. 목성에서는 이미지 기반 "실측값"이 벨트 구조에 속아 그럴듯한 오검출을 낸다는 것이 확인됐기 때문입니다.

#### 그림자 위치 알고리즘 (편평 회전타원체 광선 교차)

**소스**: `pipeline/modules/satellite_tracker.py` → `_shadow_pos_skyfield()`

1. **광행시**: 기하는 `t_emit = t_obs − d_EJ/c`(목성 기준 ≈ 47분)에서 계산합니다. 태양 쪽 끝점은 태양→목성 광행시(~43분)만큼 한 번 더 거슬러 올라갑니다.
2. **광선–회전타원체 교차**: 태양에서 위성을 지나는 광선을 편평 회전타원체 목성(R_eq = 71 492 km, R_pol = 66 854 km, IAU 극축)과 교차시킵니다. 앞쪽 반구의 교점이 그림자입니다. λ > 1(태양 → 위성 → 표면)일 때만 유효합니다.
3. **투영**: 그 점을 `t_obs` 시점 지구에서 본 겉보기 RA/Dec로 바꾸고(지구 속도에 의한 **광행차** 포함), `plate_scale`, `θ_cam = pole_pa + NP.ang`, 뒤집기 규칙으로 픽셀에 투영합니다.

> **알려진 한계, 미해결**: 예측과 실제 그림자 위치 사이에 ~5–14°(평균 ~+7–8°)의 체계적 *회전* 오차가 남아 있습니다. 여러 원인 후보를 시험해 배제했고, 부분 원인 하나(~1–3°, `pole_pa`를 자전 보정 결과물이 아닌 원본 프레임에서 측정한 것)는 수정됐습니다. 일반적인 통과 반경에서 잔여 오차는 ~9 px로, 대상 자체의 반경과 비슷합니다. [`investigation_shadow_rotational_bias.md`](investigation_shadow_rotational_bias.md) 참고.

#### CV 위치 보정 (선택)

**소스**: `pipeline/modules/position_refinement.py`

위에서 폐기한 전역 보정과 달리, 범위를 좁힌 **윈도우 단위** 보정입니다.

- **위성 본체**: 대비가 믿을 만한 것은 CH4뿐입니다(메탄이 위성이 아니라 행성 디스크를 어둡게 함). 오프셋은 윈도우마다 CH4에서 한 번 측정해 모든 필터의 예측 본체 위치에 더합니다. 세션에 CH4가 없으면 조용히 건너뜁니다.
- **그림자**: 모든 필터에서 어둡게 보이므로 필터마다 자기 그림자를 따로 보정합니다.
- **조건**: 재검출을 네 가지 박스 크기(대상 반경의 2–5배)로 실행합니다. 검출이 **3σ** 신뢰 기준(박스 크기 간 안정성 + 피크 유의성)을 넘고 **동시에** 오프셋이 20 px 이하일 때만 적용하고, 아니면 천체력 값을 그대로 씁니다. 윈도우나 세션 사이에 캐시하지 않습니다.

#### 카메라 남북 방향 (tracker_flip_ns)

**소스**: `satellite_composite.py` → `resolve_tracker_flip_ns()`, `satellite_tracker.py` → `detect_tracker_flip_ns()`

추적기는 이미지에서 북쪽이 위인지 아래인지 알아야 하며, `derot_flip`으로는 알 수 없습니다. 우선순위는 다음과 같습니다.

1. config의 명시적 `satellite.flip_ns`.
2. **벨트 비대칭 자동 감지, 윈도우별 투표**: 각 윈도우의 프레임만 평균 내서 따로 판정합니다. SEB가 NEB보다 넓고 어두우므로 `score = 0.7 × 폭 비대칭 + 0.3 × 밝기 신호`, `confidence = min(1, 2.5 × |score|)`. 신뢰도 0.25 이상인 윈도우가 투표하고 다수결로 정합니다. 목성은 시간당 ~36° 자전하므로 세션 전체를 한 번에 평균하면 몇 시간치 프레임에서 벨트가 번집니다. 실제 한 세션에서 세션 전체 평균은 신뢰도 0.02였지만 윈도우별로는 0.58–0.80이었습니다.
3. 둘 다 안 되면 그 세션의 위성/그림자 추적을 **건너뜁니다**. `derot_flip`으로 대체했을 때 실제 데이터에서 최대 ~60 px 오차가 났습니다.

#### 그림자 천체력 파일 (Skyfield BSP)

| 파일 | 크기 | URL |
|---|---|---|
| `de440s.bsp` | 32 MB | `naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/` |
| `jup365.bsp` | 1.1 GB | `naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/` |

저장 위치 순서: `PLANETFLOW_SKYFIELD_DIR` → `~/.planetflow/skyfield/` → `/tmp/skyfield/`. 파일이 없고 인터넷에 연결되어 있으면 첫 실행 시 자동으로 내려받습니다.

체크박스 옆 **BSP 상태 표시**는 백그라운드에서 다음을 확인합니다.
1. `skyfield` 임포트 불가 → 빨강, 체크박스 비활성 (`pip install skyfield` 필요)
2. BSP 파일 있음 → 초록 (OK)
3. 파일 없음, `naif.jpl.nasa.gov:443` 접속 가능 → 주황 (첫 실행 시 자동 다운로드); 불가 → 빨강, 체크박스 비활성

### 실험적 선택 옵션 (config.py 전용)

기본값은 꺼져 있고 GUI 컨트롤이 없습니다. 검증 상태는 `DerotationConfig`에 기록되어 있습니다.

| 옵션 | 상태 |
|---|---|
| `sharpness_selection_enabled` | **켜지 말 것**: 목성과 토성 모두에서 림브 안쪽에 얇고 밝은 호가 생김 |
| `s0_sl_blend_enabled` | 자전 커버리지가 낮은 곳에서 스택을 기준 프레임 쪽으로 혼합; 미검증 |
| `compute_ring_only_stack` | 토성 고리 전용 보조 스택(`<filt>_ring_only.tif`); 일반 스택보다 약간 나쁘게 측정됨, 비권장 |

---

## 6. Step 05 / 07 — 웨이블릿 선명화

**소스**: `pipeline/modules/wavelet.py`, `pipeline/steps/wavelet_master.py`, `pipeline/steps/wavelet_preview.py`

두 Step은 같은 à trous 분해와 게인 모델을 쓰지만 마스킹이 다릅니다.

| | Step 05 (마스터) | Step 07 (미리보기) |
|---|---|---|
| 입력 | Step 04 자전 보정 TIF | Step 02 TIF |
| 함수 | `sharpen_disk_aware()` — 림브에서 게인을 0으로 페더링 | `sharpen()` / `sharpen_color()` — 균일 게인, 디스크 마스크 없음 |
| 엣지 파라미터 자동 추정 | 항상 켬 (GUI에서 `auto_params=True`) | — |
| `sharpen_filter` (노이즈 게이트) | 0.0 (끔) | 0.1 |
| 토성 고리 마스크 | 있음 (아래 참고) | — |

Step 07은 디스크 인식 마스크를 일부러 쓰지 않습니다. 원형 페더가 목성의 편평한 적도 부분을 잘라먹고, 럭키 스택 TIF는 이미 림브가 부드럽게 사라지므로 일반 선명화로도 림브 링잉이 생기지 않습니다.

### Step 05 파이프라인

```
Step 04 TIF
    │
    ▼
디스크 감지 → (cx, cy, rx, ry, angle)
    │
    ▼
auto_wavelet_params():
    expand_px = sqrt(rx·ry) × 0.0505        (마스크 경계를 실제 시각적 림브까지 확장)
    eff       = 림브 안쪽 밝기 그래디언트 폭 / 2
                (대체값: sqrt(rx·ry) × 0.030)
    │
    ▼
À trous B3 분해 (6레벨) → [detail_0 (~2px), …, detail_5 (~64px), residual]
    │
    ▼
레벨 i마다:
    detail_i → 레이어별 노이즈 제거 (소프트 임계, amount × MAD)
             → 전역 소프트 임계 (sharpen_filter × MAD)
             × gain_i × spatial_weight_i
    gain_i = (amount_i / 200)^power × MAX_GAIN[i]
    spatial_weight_i = 디스크 타원 내부 코사인 S-커브, 페더 = 2^i × eff
                       (토성: ∪ 고리 환형 마스크)
    │
    ▼
재구성: 원본 + Σ 기여분 → PNG
```

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **L1 (0–500)** | 200 | ~2 px 스케일. `gain = (200/200)^1.0 × 29.15 = 29.15`. 가장 미세한 디테일 |
| **L2 (0–500)** | 200 | ~4 px 스케일. `gain = 9.48`. 미세 구조 (벨트, 밴드) |
| **L3 (0–500)** | 200 | ~8 px 스케일. `MAX_GAIN[2] = 0.0` → 현재 효과 없음 |
| **L4–L6 (0–500)** | 0 | ~16/32/64 px 스케일. `MAX_GAIN = 0.0` → 효과 없음 |
| **레이어별 노이즈 제거 L1–L6** (Step 05) | 0.0 | 게인 전 레이어별 소프트 임계: `thr = amount × MAD(detail_i)`, 이미지의 밝은 절반에서 측정. 0.1 = WaveSharp 기본(약하게), 1.0 = 강하게. L1–L2의 센서 노이즈 억제에 유용 |
| **스트레치** (Step 07) | 끔 | 미리보기 PNG의 대비 스트레치 |
| **채도 강화** (Step 07, 컬러 전용) | 켬 | 미리보기 PNG의 Lab 채도 강화 |
| **색상 보정 적용** (Step 07, 컬러 전용) | 켬 | 선명화 전 자동 화이트밸런스 + 색수차 보정 |

> **amount의 의미**: `gain_i = (amount/200)^power × MAX_GAIN[i]`. amount = 200 → gain = MAX_GAIN, 400 → 2 × MAX_GAIN, 100 → 0.5 × MAX_GAIN. `MAX_GAIN[2:] = 0`이므로 현재 결과에 영향을 주는 것은 L1과 L2뿐입니다.

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| `MAX_GAINS` | [29.15, 9.48, 0, 0, 0, 0] | WaveSharp 기준 출력에 대한 OLS 회귀로 정한 레벨별 최대 게인 |
| `power` | 1.0 | `gain = (amount/200)^power × MAX_GAIN`. 1.0 = 선형 |
| `edge_feather_factor` | 자동 | `auto_wavelet_params()`가 이미지마다 측정 |
| `disk_expand_px` | 자동 | `sqrt(rx×ry) × 0.0505`. 림브 감광 때문에 Otsu 윤곽선이 시각적 림브보다 안쪽에 잡히므로, 페더 시작점을 실제 림브까지 밀어냄 |
| `master_filter_type` | `"gaussian"` | B3-스플라인 분해. config에서 `"zerogauss"`, `"bilateral"`도 선택 가능 |

### À Trous B3-스플라인 웨이블릿 분해

"À trous"(구멍 뚫린)는 비데시메이션 웨이블릿으로, 다운샘플링 대신 커널 탭 사이에 0을 넣어 스케일을 키웁니다.

```
_B3 = [1, 4, 6, 4, 1] / 16   (5탭 분리형 커널)

레벨 i에서 탭 간격 = 2^i:
  smoothed_i  = B3_i ⊗ image_i    (reflect 패딩)
  detail_i    = image_i − smoothed_i
  image_{i+1} = smoothed_i
```

### 디스크 인식 엣지 페더링 (Step 05)

게인은 디스크 타원 안에서만 적용되고 코사인 S-커브로 림브를 향해 줄어듭니다. 굵은 레벨일수록 페더가 넓습니다.

```
feather_L = 2^L × edge_feather_factor
t = clip(dist_from_boundary / feather_L, 0, 1)
weight_L = 0.5 × (1 − cos(π × t))
```

페더 위치는 회색 헤일로(게인이 실제 림브보다 너무 안쪽에서 0이 됨)와 흰 테두리 오버슈트(가파른 림브 그래디언트에 게인이 그대로 걸림) 사이의 절충입니다. 흰 테두리는 받아들일 수 있는 절충이 아니라 **치명적 결함**으로 취급합니다.

### 토성 고리 선명화 마스크

토성(`has_rings`)에서는 `compute_ring_sharpening_mask()`가 실제 고리 **환형(annulus)**을 게인 마스크에 더합니다(`master_ring_extension_enabled`, 기본 **켬**). 고리 반경, `pole_pa`, 관측자 직하점 위도 B로 해석적으로 계산합니다. 고리가 행성 본체와 겹치는 곳은 앞쪽 호만 쓰고, 본체와 고리 사이 틈에는 게인을 **전혀** 주지 않습니다. 이전의 채워진 타원 방식은 그 틈(본체 자체의 PSF 꼬리)에 최대 게인을 줘서 본체-고리 경계에 흰 테두리/어두운 골이 생겼습니다.

### 실험적 선택 옵션 (config.py 전용)

모두 기본 꺼짐이며 GUI 컨트롤이 없습니다. `WaveletConfig` docstring의 상태 목록을 참고하세요.

| 옵션 | 상태 |
|---|---|
| `master_edge_extension_enabled` | 분해 전 디스크 타원 바깥을 채움; 합성 테스트만 있음 |
| `master_coverage_aware_sharpening` | 자전 커버리지로 게인 조절; 실효 미미 |
| `master_limb_darkening_confidence_enabled` | Minnaert 피팅으로 게인 조절; 미결 |
| `master_navigation_limb_fit_enabled` | 토성 전용 제약 림브 피팅; 결론 없음 |
| `master_overshoot_clamp_radius_px` | **켜지 말 것**: 실제 디테일의 92–98%를 지움 |
| `master_limb_fit_refinement_enabled` | **켜지 말 것**: 목성에서 흰 테두리 발생 |
| `master_ring_scatter_subtraction_enabled` | **켜지 말 것**: 어두운 선/밝은 호 아티팩트 발생 |

---

## 7. Step 06 — RGB 합성

**소스**: `pipeline/modules/composite.py`, `pipeline/steps/rgb_composite.py`

```
Pass 1 — 윈도우마다, 합성 설정마다:
    (선택) 필터 정규화: (필터, 윈도우)별 디스크 중앙값 배율 조정
        │
        ▼
    채널 스트레치 (스트레치 켬이면 "joint", 아니면 "none")
        │
        ▼
    기준 채널: spec.align_ref > L > IR > R > G > B > CH4
    나머지 채널 → 전체 프레임 위상 상관
        → 최대 이동 이내이면서 이동하지 않을 때보다 디스크 고역 통과
          NCC가 좋아질 때만 이동 적용
        │
        ▼
    np.stack([R, G, B])
        ├─ [RGB]  그대로
        └─ [LRGB] RGB → Lab, Lab L을 L 채널로 교체, Lab → RGB
        │
        ▼
Pass 2 — 윈도우마다:
    글로벌 정규화 (윈도우 간 평균 휘도) → 채도 강화
    → 밝기 배율 → 16비트 PNG
```

### Step 06 GUI 파라미터 → 내부 동작 (모노 모드)

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **최대 채널 정렬 이동 (px)** | 15.0 | 위상 상관 이동량이 어느 축으로든 이 값을 넘으면 적용하지 않음. 대기 분산이 심한 날은 20–30으로 올림 |
| **글로벌 정규화** | 켬 | 합성 후 각 윈도우의 합성 이미지 평균 휘도를 윈도우 전체 평균에 맞춤. GIF 밝기 깜빡임 제거 |
| **필터 정규화** | 끔 | 합성 전 모든 (필터, 윈도우)의 디스크 중앙값을 구해, 필터마다 모든 윈도우가 같은 중앙값을 갖도록 배율 조정. 순수 배율이라 어두운 배경이 유지되고 클리핑이 없음. 세션 동안의 투명도 변화 보정 |
| **밝기 배율** | 1.0 | `output = composite × brightness_scale`, 마지막에 적용. 범위 0.1–2.0 |
| **스트레치** | 끔 | 켜면 색 채널을 공통 lo/hi(`"joint"`)로 백분위 스트레치해 색 비율 유지 |
| **채도 강화** | 켬 | 디스크 채도의 99.5번째 백분위가 `headroom × 127`에 이르도록 Lab 채도 게인 적용. 게인은 [1, 4]로 제한, 채도를 낮추지 않음 |
| **합성 설정 (R/G/B/L)** | RGB, IR-RGB, CH4-G-IR | 합성 이미지별 필터–채널 매핑. L을 지정하면 그 합성은 LRGB 모드 |

### 채널 정렬 품질 검사

`cv2.phaseCorrelate`는 기준 채널과 상관된 신호가 적은 채널에서 그럴듯하지만 **틀린** 피크를 낼 수 있습니다. 실제 목성 세션 하나에서 B 채널(IR 기준)이 윈도우의 ~25%에서 이런 결과를 냈고(예: 이웃 윈도우는 −0.4–−0.9인데 dy = +2.16), 크기만 보는 최대 이동 제한으로는 걸러지지 않는 림브 색 번짐이 생겼습니다. 그래서 후보 이동은 디스크 내부(r < 0.95R)의 고역 통과 NCC(σ = 4 px)가 이동하지 않을 때보다 이동했을 때 더 높아야만 적용합니다. 디스크를 감지하지 못하면 이전 방식(최대 이동 이내면 신뢰)으로 처리합니다.

전체 프레임 상관만 사용합니다. 디스크 ROI 크롭은 시도했다가 명시적으로 폐기했습니다.

### 내부 고정값

| 파라미터 | 값 | 역할 |
|---|---|---|
| 기준 채널 우선순위 | `align_ref` > L > IR > R > G > B > CH4 | 고정 순서. 채널 밝기가 바뀌어도 윈도우마다 합성 행성 위치가 튀지 않게 함 |
| `color_stretch_mode` (스트레치 켬) | `"joint"` | config에서 채널별 `"independent"`도 선택 가능 |
| `saturation_headroom` | 0.15 | 99.5번째 백분위 채도 목표 = 0.15 × 127 |

### LRGB 합성

L 채널을 지정하면 Lab 휘도를 외부 채널로 바꿉니다.

```
Lab = cv2.cvtColor(rgb, COLOR_RGB2Lab)
Lab[:,:,0] = lrgb_weight × (L_external × 100) + (1−w) × Lab[:,:,0]
result = cv2.cvtColor(Lab, COLOR_Lab2RGB)
```

IR-RGB에서는 더 선명한 IR 채널이 휘도 디테일을, R/G/B가 색을 담당합니다.

### 사후 림브 채도 감소 — 비활성

0.89r–1.04r 구간의 Lab a/b 테이퍼로 림브 색 번짐을 억제하던 기능입니다. **2026-08-09에 비활성화**했습니다. ~10 px 폭(반경 바깥 15%)의 띠가, 고친다고 했던 ~1.5 px 파장 의존 림브 효과보다 훨씬 넓었습니다. 채널 정렬 오차(토성에서 최대 14.6 px)를 가리도록 튜닝된 것으로 보입니다. 실제 토성 데이터에서는 디스크 바깥쪽의 실제 색을 지워, 회색 구 안에 작은 색 있는 구가 들어 있는 것처럼 보였습니다. 코드는 `compose()`에 주석으로 남아 있습니다. 얇은 번짐이 다시 보이면 먼저 정합 버그부터 확인하고, 그다음에야 훨씬 좁은 띠(예: 0.97r–1.02r)로 다시 켜야 합니다.

### 컬러 카메라 모드

채널 매핑이 없습니다. 윈도우마다 자동 화이트밸런스(채널별 게인, G 기준)와 색수차 보정(R/B 이동, G 기준)을 적용하며, 패널에서 첫 윈도우의 보정 전후를 미리 볼 수 있습니다.

---

## 8. Step 08 — 애니메이션 GIF

**소스**: `pipeline/steps/gif.py`

```
합성 이미지별 step06_rgb_composite/ PNG (윈도우 시각순 정렬)
    │
    ▼
행성 중심 맞추기: 프레임마다 디스크 중심 감지 → 5프레임 이동 중앙값
→ 평활화된 중심이 이미지 중앙에 오도록 각 프레임 이동
    │
    ▼
uint8 변환 → 선택적 리사이즈 (cv2 INTER_LANCZOS4)
    │
    ▼
프레임별 Floyd–Steinberg 디더링, 256색 팔레트
    │
    ▼
{composite}_animation.gif (loop = 0, 무한 반복)
```

중심 맞추기는 합성된 행성이 프레임마다 흔들리는 현상을 없앱니다. 이동 중앙값은 흐릿한 림브 감지 노이즈 때문에 대적점 같은 피처가 흔들려 보이는 것을 막습니다.

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **재생 속도 (FPS)** | 6.0 | `frame_duration = int(1000 / fps)` [ms], Pillow `duration`으로 전달 |
| **리사이즈 비율** | 1.0 | `new_size = (int(w × factor), int(h × factor))`, LANCZOS4 리샘플링 |

---

## 9. Step 09 — 요약 그리드

**소스**: `pipeline/steps/summary_grid.py`

Step 09는 항상 `summary_grid_simple.png`를 생성하고, 모노 모드 + Step 05 출력이 있는 경우 추가로 `summary_grid.png`(2존)와 선택적으로 `analytic/` 하위 폴더에 윈도우별 Analytic PNG를 생성합니다.

### 출력 파일

| 파일 | 내용 | 생성 조건 |
|------|------|-----------|
| `summary_grid_simple.png` | 합성 이미지만 (전체 윈도우 × 전체 합성) | 항상 |
| `summary_grid.png` | 합성(왼쪽 존) + Step 05 필터 이미지(오른쪽 존), 동일 셀 크기, 수직 구분선 | 모노 모드 + Step 05 데이터 있을 때 |
| `analytic/window_XX_analytic.png` | 윈도우별 상세 분석 뷰 (하단 참고) | `save_analytic=True`, 모노 모드 |

### 단순 그리드 (`summary_grid_simple.png`)

```
Step 06 합성 PNG (전체 윈도우)
    │
    ▼
각 이미지:
    블랙 포인트: pixel = clip((p − bp) / (1 − bp), 0, 1)
    감마:        pixel = pixel ^ (1 / gamma)
    cell_size로 리샘플링
    │
    ▼
그리드 배치 (행=윈도우, 열=합성) → PNG
```

### 2존 그리드 (`summary_grid.png`)

```
왼쪽 존: Step 06 합성 (각 cell_size × cell_size)
오른쪽 존: Step 05 필터 PNG (동일 cell_size)
    │
    ▼
존 사이에 수직 구분선
각 이미지 위에 열 라벨
왼쪽에 행(윈도우) 시간 라벨
    │
    ▼
summary_grid.png
```

두 존 모두 동일한 `cell_px`(설정된 **셀 크기** 값)를 사용합니다. 필터 존 너비 = `n_filters × cell_px + 간격`.

### Analytic View (`analytic/window_XX_analytic.png`)

윈도우당 한 장씩 생성됩니다. 레이아웃(위에서 아래):

```
[헤더: 윈도우 시간 범위]

[필터 이미지 행]              ← Step 05 PNG, 각 cell_size
[필터 통계 블록]              ← 필터별 Frames / Q.Post / Stab. / Stacked, 위 이미지와 열 정렬

─────────────────────────────────── (구분선)

[합성 이미지 행]              ← Step 06 PNG, 각 cell_size
[회전 지시자]                ← N/S 극 방향 선분 + 자전 방향 화살표,
                               첫 번째 합성 이미지의 디스크 바깥에 표시
[Align 테이블]               ← 행=필터명; 열=합성명
                               셀값 = "[역할] 이동량" 또는 "[역할] ref" 또는 "—"
[Sat 행]                     ← 합성별 채도 강화 배율

─────────────────────────────────── (구분선)

[글로벌 파라미터 줄]          ← Win.Q / Rot / Wvl / bp / γ
```

**회전 지시자**: 디스크 림브 바깥에 그려진 짧은 선분이 북극(N, 파란색)과 남극(S, 빨간색) 방향을 나타내며, 로그에 기록된 `pole_pa_deg`와 `tracker_flip_ns` 값을 기반으로 합니다. 디스크 바깥에 그려진 곡선 화살표는 자전(프로그레이드) 방향을 나타냅니다.

#### 필터 통계 블록

구분선 위, 필터 이미지 열과 x좌표 정렬. 필터명 헤더 행 없음(위의 이미지 라벨이 이미 표시).

| 행 | 값 |
|----|----|
| **Frames** | `n_used / n_total` (품질 임계값을 통과한 프레임 / 윈도우 전체 프레임) |
| **Q.Post** | σ-클리핑 후 남은 프레임들의 평균 품질 점수 (0~1) |
| **Stab.** | `1 / (1 + CV)`, `CV = std/mean` (프레임별 품질 점수의 변동계수) |
| **Stacked** | Lucky stacking에서 실제 합산된 최종 프레임 수 |

#### Align 테이블

- **행** = 필터명 (IR, R, G, B, CH4 …) — 전체 합성의 `CompositeSpec` 필드에서 도출
- **열** = 합성명 (RGB, IR-RGB, …)
- **셀 값** = `[역할] 이동량`, `역할` ∈ {L, R, G, B}, `이동량` = `composite_log.json`의 `(Δx, Δy)`; 기준 채널이면 `ref`; 해당 합성에 미사용이면 `—`
- **`composite_log.json`의 정렬 키**는 채널 역할(L/R/G/B)이 아닌 필터명(IR/R/G/B/CH4)

#### 캔버스 높이 사전 계산

`Image.new()` 호출 전에 1×1 프로브 드로우로 높이를 계산합니다:

```python
canvas_h = (pad + header_h
            + filter_lbl_h + filter_px   # 필터 이미지
            + fstats_h                   # 필터 통계 (구분선 위)
            + section_gap                # 구분선
            + comp_lbl_h + comp_px       # 합성 이미지 (이름만 라벨)
            + apar_h                     # align 테이블 + 구분선 + 글로벌 파라미터
            + pad)
```

`label_margin`(가장 넓은 행 라벨 너비 + 12px)을 `canvas_w`에 추가하여 행 라벨이 왼쪽 경계 밖으로 넘치지 않도록 합니다.

### GUI 파라미터 → 내부 동작

| GUI 파라미터 | 기본값 | 내부 동작 |
|---|---|---|
| **블랙 포인트** | 0.04 | `pixel = clip((p − 0.04) / (1 − 0.04), 0, 1)`. 배경 노이즈를 순수 검정으로 밀어냄. 0.02~0.08 권장 |
| **감마 (Gamma)** | 0.9 | `pixel = pixel ^ (1/0.9) ≈ pixel ^ 1.11`. <1.0=밝아짐(기본 0.9는 행성을 약간 밝게), >1.0=어두워짐, 1.0=변화 없음 |
| **셀 크기 (px)** | 300 | 그리드의 합성 및 필터 이미지를 이 크기로 리샘플링. 2존 그리드에서도 두 존 모두 동일한 셀 크기 사용 |
| **Analytic View 저장** | True | True 시 각 시간 윈도우에 대해 `analytic/window_XX_analytic.png` 생성. 모노 모드 전용 |

---

## 10. 공통 모듈: 디스크 감지 (find_disk_center)

**소스**: `pipeline/modules/derotation.py`

Step 02, 04, 05, 06에서 공통으로 사용합니다.

```
Phase 1 — 중심과 형태 (Otsu 이진화)
1. arr8 = clip(image × 255, 0, 255).uint8
2. effective_thresh = Otsu × (1 − 0.10)
   (margin_factor = 0.10: 어두운 림브를 포함하도록 임계값을 낮춤)
3. 모폴로지 Closing (7×7 타원 커널) → 작은 틈 메우기
4. 최대 윤곽선 → cv2.fitEllipse → (cx, cy, semi_a, semi_b, angle)

Phase 1b — 고리 블롭 (원래 종횡비 < 0.80, 예: 토성)
5. 블롭 내부 밝기 60번째 백분위수로 다시 이진화해 밝은 원반 코어만
   분리하고 코어에 타원을 다시 피팅.
6. 그럴듯한 코어를 찾지 못하면 (예: 원반이 고리보다 어두운 토성 CH4):
   semi_a를 방사형 림브 탐색으로 대신 구함. 형태는 신뢰할 수 없음으로
   표시하고(shape_reliable = False) Step 04에서 다른 필터의 값을 빌림.

Phase 2 — 반경 정밀화 (그래디언트 림브 검출)
7. (cx, cy)에서 5° 간격 72개 광선, 각 광선을
   [0.75, 1.30] × semi_a 범위에서 샘플링 (고리 제거 코어: [0.6, 1.8])
8. 1-D 가우시안 평활화 (σ = 1.5 px) → 최대 하강 지점 → 포물선
   서브픽셀 보정
9. 이상치 제거 (중앙값 ± 2σ) → 중앙값 → semi_a_refined
   (semi_b는 피팅된 종횡비를 유지하도록 같은 비율로 조정)

반환: (cx, cy, semi_a_refined, semi_b, angle_deg)
```

내부 함수 `_find_disk_center_impl()`은 **신뢰도**도 반환합니다. 고리 없는 피팅은 1.0, 코어 분리 성공은 (0, 1], 방사형 림브 대체 경로는 확인된 엣지가 8개 이상이면 0.5·미만이면 0.3, 피팅 실패는 0.0입니다. `shape_reliable` 플래그도 함께 반환합니다. Step 04는 이 값으로 필터가 다른 필터의 형태나 위치를 빌려야 하는지 판단합니다.

이진화 방식은 중심은 정확하게 잡지만, Otsu가 어두운 바깥 림브를 잘라내므로 반경을 작게 잡습니다. 그래디언트 방식은 광선마다 실제 밝기 변곡점을 찾습니다. 다만 이 경계도 림브 감광 때문에 실제 기하학적 림브보다 안쪽에 있으므로, 위성 플레이트 스케일에는 별도로 Minnaert 보정 반경을 씁니다(§5).

---

## 11. 공통 모듈: 서브픽셀 정렬

**소스**: `pipeline/modules/derotation.py`

### apply_shift

```python
M = np.float32([[1, 0, dx], [0, 1, dy]])
cv2.warpAffine(image, M, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)
```

- **INTER_CUBIC**: 바이큐빅 보간 (림브에서 링잉이 생기는 LANCZOS4는 쓰지 않음)
- **BORDER_REPLICATE**: 가장자리 픽셀 복제 (BORDER_CONSTANT = 0이면 이미지 가장자리에 검정이 스며듦)

### apply_shift_and_scale

```
output = ref_center + scale × (input − target_center)
```

프레임 **자신의** 측정 중심을 축으로 배율을 조정한 뒤, 그 중심을 기준 중심으로 옮깁니다. 기준 중심을 축으로 배율을 조정한 뒤 이동하는 것과는 다르며, 둘은 scale = 1일 때만 같습니다. Step 04의 워프 전 정렬에서 씁니다. 프레임 간 반경이 1–2% 달라지면 디스크에서는 1 px 미만이지만 토성 카시니 간극에서는 1.6–2.8 px가 되어, 스택에서 간극이 지워질 만큼 큽니다. 프레임 간 회전 보정도 시도했지만, 측정된 3–4° 각도 흔들림이 윤곽선 피팅 노이즈로 밝혀져 되돌렸습니다.

### subpixel_align (위상 상관)

```python
(raw_dx, raw_dy), _ = cv2.phaseCorrelate(ref_f32, tgt_f32)
return -raw_dx, -raw_dy
```

`cv2.phaseCorrelate`는 보정량이 아니라 기준 대비 대상의 순방향 이동량을 반환합니다. 그래서 결과의 **부호를 뒤집어**, `apply_shift(target, *subpixel_align(ref, target))`가 대상을 정렬하도록 합니다. 정밀도는 ~0.1 px입니다.

### limb_center_align

`find_disk_center()`로 대상의 디스크 중심을 찾아 `(ref_cx − cx, ref_cy − cy)`를 반환합니다. Step 02의 프레임별 전역 정렬과 Step 04의 워프 후 대체 정렬에 씁니다.

---

*참고문헌: Starck & Murtagh (2006), Bijaoui (1991), Mackay (2013 arXiv:1303.5108), Zack (1977)*
