# Python 3.11을 기반으로 사용
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 모든 파일을 컨테이너로 복사
COPY . /app

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너 실행 시 Gunicorn을 사용하여 Flask 실행
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
