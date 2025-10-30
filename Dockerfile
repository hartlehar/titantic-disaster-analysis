FROM python

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src

CMD ["python3", "train.py"]