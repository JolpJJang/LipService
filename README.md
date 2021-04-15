# LipService
딥러닝 기반 입술 모양 인식하여 구어 텍스트화

---

## 원본 저장소 커밋 내역을 로컬에 반영시키는 법

Fork 떠서 Clone한 레포지토리 디렉토리에서 Git bash 키고 치는 명령어들

### 1. 원격저장소 주소 저장

- upstream이라는 이름으로 remote 주소 저장

`git remote add upstream <<원본 저장소 주소>>`

### 2. `upstream` 내역을 스테이징 영역에 fetch

`git fetch upstream`

### 3. 로컬에 원하는 브런치 커밋 내역 merge

- upstream으로 저장된 레포지토리의 main 브랜치에 반영된 커밋 내역을 내 로컬 개발물에 병합한다. (Pull: upstream/main -> origin/main)

`git merge upstream/main`