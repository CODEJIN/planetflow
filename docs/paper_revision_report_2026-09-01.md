# PlanetFlow 논문(`camera_ready.tex`) 개정 보고서

**작성일**: 2026-09-01
**대상**: PASP 투고본 `paper/camera_ready.tex`
**배경 자료**: `paper/Paper_Fix.md`(자기완결적 이슈 기록), `docs/investigation_shadow_rotational_bias.md`(회전편향 조사 전체 로그, 13개 섹션+§12-8)

이 문서는 2026-08-29~2026-09-01에 걸쳐 진행된 논문-코드 대조 감사와 그 후속 조사 전체를 하나로 정리한 것이다. 상세 근거·원자료 경로는 각 절에 인용된 파일들에 있다.

---

## 1. 작업 배경

논문은 몇 달 전 코드 상태를 기준으로 작성됐고, 이후 코드가 계속 진화했다. 10-cluster 병렬 대조 감사(`experiments/paper_vs_code_diff_2026-08-29.md`)로 CRITICAL 2건 + HIGH 2건을 발견했고, 이후 저자가 직접 작업하며 CRITICAL 1건(plate-scale 자동보정 기능 완전 제거)이 추가로 발견됐다. 이 시점까지의 기록이 `Paper_Fix.md` 이슈 1~6이다.

이 세션(2026-09-01)에서는 이 이슈들이 현재 코드에서도 여전히 유효한지 재검증하고, 별도로 발견된 **위성/그림자 위치추적 회전편향**(이슈 7)이 논문의 핵심 정량 결과(Table 2)에 실제로 영향을 주는지를 끝까지 추적했다.

---

## 2. 이슈별 현재 상태

`camera_ready.tex`를 직접 읽어 대조한 결과, **이슈 1~4는 이미 반영되어 있었다** (별도 세션/에이전트가 `Paper_Fix.md`의 권장 문구를 이미 적용한 것으로 보임):

| 이슈 | 내용 | 현재 상태 |
|---|---|---|
| 1 (CRITICAL) | Sec 2.2가 2D shear를 기본으로 서술 | **반영 완료** — line 314-320, 현재 oblate-spheroid 3D reprojection으로 재서술됨 |
| 2 (CRITICAL) | alpha=1.00 "NCC로 검증됨" 자기참조 주장 | **반영 완료** — line 318, "진단 지표일 뿐 이 값을 검증/수정하지 않는다"로 수정됨 |
| 3 (HIGH) | Quality Score "r<0.85R" 서술이 코드에 없음 | **반영 완료** — line 284, Otsu+margin_factor 방식으로 정확히 재서술됨 |
| 4 (HIGH) | Poisson blending 3단계 인과 서술이 실제와 다름 | **반영 완료** — line 356, Dirichlet 경계조건 메커니즘으로 정확히 재서술됨 |
| 5 (MEDIUM) | `aperture_contrast.py`의 r 정의 모순 | 코드 수정 완료(2026-08-29), 논문 서술과 무관 |
| 6 (CRITICAL) | plate_scale CV 자동보정 서술 — 코드에서 완전히 제거됨 | **미반영** — line 338-340과 Limitations line 535가 여전히 존재하지 않는 기능을 서술 중 |
| 7 (신규) | 위성/그림자 회전편향, 원인 미상 | **미반영** — Limitations에 항목 없음 |

이번 작업에서 **이슈 6·7을 반영하고, Table 2 및 관련 본문 수치를 갱신**한다(§4 참고).

---

## 3. Table 2 재계산 — 세 겹의 독립적 문제 발견

사용자가 "wavelet을 유지"하기로 결정했지만, 그 결정과 별개로 **현재 논문 Table 2 자체가 세 가지 이유로 갱신이 필요함**을 확인했다.

### 3-1. 코드 버전 차이 (published 값 자체가 낡음)

`experiments/reprocess_2026-08-31/`(현재 HEAD 코드로 6개 세션 전부 재처리)를 `aggregate_reprocess.py`로 집계하니 표본 수부터 다르다 — 위성 9→14창, 그림자 15/16→25창. 이번 세션의 위성 트래커 버그 수정들(`7f6de15`, `3b88bb6`, `58b64f6`, `26ec5f5`)과 이전 3D reprojection 승격(`ebe674f`)이 논문 작성 이후 누적된 결과다.

### 3-2. 위치 정확도 (이슈 7의 회전편향)

위성/그림자 예측 위치에 세션별 회전편향이 존재함이 확인됐다(2026-04-17, 2026-03-21에서 평균 +4~8°, 5~15px 상당). 이를 반영하기 위해, **de-rotation/tracker 자체는 순수 ephemeris를 유지**하되(생산 코드 변경 없음), **측정 시점에만** 진단용 CV로 정밀 좌표를 확인하는 방법론을 전체 데이터셋(203행)에 적용했다:

- 신호가 확실한 CH4 필터에서 로버스트 박스-스윕 CV 탐지(3-sigma 문턱, box_mult 2~5)를 수행하고, 신뢰도가 medium/high면 그 오프셋을 같은 창의 IR/R/G/B에도 동일하게 적용(같은 물리적 사건이므로 오프셋이 필터 무관하게 같다는 전제).
- 위성 14그룹 중 6그룹, 그림자 25행 중 6행에서 CH4 앵커링 성공. 나머지는 CH4조차 신호가 약해(브로드밴드 위성은 100% 신호 없음, 기존 로버스트 탐지 결과와 일치) 예측좌표를 그대로 사용.
- **결과: 좌표 보정 전후로 집계 수치 차이는 0.0~0.4%p 수준으로 미미함.** 신호가 없는 곳은 애초에 고칠 게 없고, 신호가 있는 곳도 오프셋(평균 1.6~2.9px)이 ROI 반경(6~11px)에 비해 작아서 측정값 자체를 크게 바꾸지 못함. → **좌표 편향은 실재하지만 Table 2 수치에 대한 실질적 영향은 작다**는 게 최종 결론.

원자료: `experiments/table2_cv_verified_current_head.tsv`, `experiments/full_cv_verified_contrast.py`.

### 3-3. 측정 방법 (wavelet vs raw)

`_measure_window_pair()`가 대비 측정 직전 강한 wavelet sharpening(`_apply_wavelet_display`, 원래 docstring은 "visual display only, not for contrast measurement"라고 명시돼 있었음에도 실제로는 측정에 사용됨)을 적용하는데, 이게 측정값을 체계적으로 부풀린다:

- 6개 세션·전체 193행 기준, wavelet은 raw 대비 평균 4~5배 큰 Δ%를 만든다.
- 포화(정확히 1.0/0.0 클리핑)가 그림자 23~36%, 위성 CH4 30%대에서 발생.
- 포화 안 된 구간만 봐도 원신호와의 상관이 필터별로 0.07(거의 무관)~-0.55(역상관)까지 나옴 — "같은 효과를 크게 보여주는 것"이 아니라 "다른 걸 측정"하는 수준.
- **사용자 결정**: 그럼에도 wavelet 기준을 유지하기로 함 — production이 실제로 산출하는 최종 이미지가 wavelet 적용본이므로, 최종 산출물 기준의 측정치가 더 대표성이 있다는 판단.
- 경계(limb) 근접 시 측정이 더 불안정해진다는 것도 확인됨(경계 20px 이내 |Δ%| 평균 26.7% vs 20px 이상 9.3%) — 다만 "경계라서 특정 방향(음수)으로 치우친다"는 주장은 반증 사례(2026-05-05 window_07)로 기각됨. 방향과 무관하게 변동성만 커진다는 게 정확한 서술.

### 3-4. 최종 채택 수치 (코드 최신화 + CV 앵커링 위치 + wavelet 측정)

```
                    IR         R          G          B         CH4
satellite Total  14x+27.7%  14x+10.3%  14x+4.2%   14x+1.3%   13x+13.3%
shadow    Total  25x+11.5%  25x+13.2%  25x+8.3%   25x+9.4%   24x+10.3%
```

세션별 상세는 `experiments/table2_cv_verified_current_head.tsv` 및 `experiments/full_cv_verified_contrast.py` 실행 출력 참고.

**중요한 새 발견**: 이 최종 수치에서는 위성 R 필터가 +10.3%로, 논문이 기존에 주장하는 "브로드밴드는 원반-위성 반사율이 비슷해서 개선폭이 거의 0"이라는 물리적 설명과 어긋난다(published 값은 R -0.1%로 이 설명과 잘 맞았음). raw 측정으로는 R이 여전히 +0.5%(거의 0)이므로, 이 +10.3%는 wavelet의 비선형 증폭이 R 채널에만 유독 크게 작용한 결과로 보인다 — B(+1.3%)는 기존 설명과 부합, G(+4.2%)는 애매한 중간값. **본문의 필터별 설명(Sec 3.4)에서 이 R 채널의 예외를 정직하게 언급하도록 조정했다**(§4-3 참고).

### 3-5. 04-24 세션에 대한 정정

세션 도중 "2026-04-24 간니메데가 현재 코드로는 재현이 안 된다(composite==baseline)"는 의심이 있었으나, 직접 조사한 결과 **잘못된 경보였음을 확인**했다. 실제 프로덕션 함수(`get_positions()`)를 정확한 파라미터로 호출하면 지금도 명확하게(92px vs disk_r 98.8px, 여유 6px) on-disk로 판정되며, 다른 재현 데이터셋(`experiments/reprocess_2026-08-31_final/`)에서는 정상적으로 합성됨을 확인했다. 문제가 됐던 건 특정 진단 스크립트(`tracking_check_2026-08-31_v2`, 지금은 소실됨)의 버그였지, 프로덕션 코드나 Table 2의 04-24 데이터 자체는 문제 없음.

또한 "04-24가 경계(limb)에 가까워서 음수 값이 나온다"는 가설도 검증했으나, 거의 동일하게 경계에 가까운 2026-05-05 window_07이 오히려 강한 양수를 보이는 반증 사례가 있어 기각했다(§3-3 참고). 경계 근접은 변동성을 키울 뿐 방향을 정하지 않는다.

---

## 4. 논문에 실제로 반영한 변경사항

### 4-1. Sec 2.3 "Automated Plate Scale Calibration" 재작성 (이슈 6)

기존: CV 기반 argmin 탐지 + 최소제곱 피팅 서술(코드에 더 이상 존재하지 않는 기능) → 신규: Horizons 각반경 + Minnaert-law 방사형 밝기 프로파일 피팅 기반의 순수 광도측정(photometric) 방법으로 교체. 관측된 그림자/위성 위치를 사후 보정에 쓰지 않는다는 점을 명시.

### 4-2. Limitations 절 갱신 (이슈 6, 7)

- "Shadow-Transit Plate Scale Calibration" 항목: 더 이상 존재하지 않는 38px 조건부 서술을 삭제하고, Minnaert 피팅의 실제 한계(디스크 자체가 아주 흐릿하거나 반경 추정이 실패하는 극단적 시상 조건)로 교체.
- 신규 항목 "Residual Angular Placement Bias" 추가: 6개 세션 중 2개에서만 확인된 회전편향, 조사한 원인 후보 전부(천체력/광선추적/부호/plate_scale/limb-darkening/PSF/극축정렬/marker propagation), 원인 미상이라는 점, 나머지 4개 세션은 신호가 약해 검증 자체가 불가능했다는 점을 정직하게 서술.
- 신규 항목 "Wavelet-Sharpening Measurement Variance Near the Limb" 추가: 대비 측정에 사용하는 wavelet sharpening이 경계 근접 시 변동성을 키운다는 실측 결과 반영.

### 4-3. 본문 수치 전면 갱신

Table 2, 초록(Abstract), Sec 3.4(Quantitative Accuracy 요약), Sec 4.1(Advantages), Sec 3.5/discussion(필터별 설명)의 모든 퍼센트 수치를 §3-4의 최종 채택 수치로 교체. 위성 R 필터의 예외적 거동(§3-4 참고)을 필터별 설명에 한 문장으로 추가.

### 4-4. Sec 3.2 측정 방법론에 CV 검증 절차 추가

"측정 위치는 CH4 필터의 로버스트 신호 탐지로 진단 검증했으며, 이 검증은 오직 측정에만 쓰이고 트래커 자체는 순수 ephemeris 예측을 사용한다"는 문장을 추가해, 이슈 7에서 발견한 편향을 측정 단계에서 완화했음을 투명하게 밝힘.

---

## 5. 재현 이미지 (완료)

`experiments/paper_figures_2026-09-01/`에 논문의 데이터 의존 그림 4개를 전부 현재 코드로 재생성했다:

- `workflow_v2.pdf` — 아키텍처 다이어그램. 순수 스키매틱이지만 "2D shear" 표기를 "Oblate-Spheroid 3D Reprojection"으로 갱신(이슈 1 반영).
- `roi_diagram_v2.pdf` — ROI 측정 다이어그램. 2026-03-21 Io 그림자(window_03, IR)의 실제 baseline/composite 크롭 사용. 코드에서 제거된 `set_plate_scale_calibration()` 호출을 스크립트에서도 제거(이슈 6과 동일한 정리).
- `comparison_grid_v2.pdf` — Figure 2(정성적 비교). 2026-04-17 Ganymede 그림자, **2026-03-28 Io**(2026-04-24 대체 — 경계 근접으로 시각적으로 washed out되는 문제, 이전에 이미 발견되어 대체돼 있었음), 2026-05-05 Europa. 논문 캡션도 이 대체를 명시하도록 수정함.
- `derotation_flow_gradient_r_v2.pdf` — 2026-05-05 Europa 위성+그림자 중간 처리 단계 시각화. window 순번(window_06)과 프레임 인덱스(BODY_IDX=21, SHAD_IDX=4)가 재현 실행에서도 동일하게 나와 기존 하드코딩된 인덱스를 그대로 사용 가능함을 확인.

모두 현재 코드(`experiments/reprocess_2026-08-31/`, 이번 세션 트래커 수정 전 시점 데이터셋이지만 3D reprojection 등 핵심 경로는 현재와 동일) 기준으로 재생성됨. `camera_ready.tex`의 `\includegraphics` 경로는 아직 예전 파일(`paper/comparison_grid.pdf` 등)을 가리키고 있음 — 검토 후 교체 여부 결정 필요.

---

## 6. 남은 작업 / 권고사항

- `camera_ready.tex`는 git 추적 대상이 아님 — 별도 백업/버전관리 필요.
- 위성 R 필터의 wavelet 예외적 증폭(§3-4)은 이번엔 "예외로 명기"하는 선에서 처리했으나, 근본 원인(왜 R만 유독 크게 증폭되는가)은 조사되지 않았음 — 후속 조사 후보.
- Table 1의 세션별 프레임 수, Table 3(처리 성능)은 이번 조사 범위 밖이며 변경하지 않았음.
