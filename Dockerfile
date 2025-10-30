FROM python

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /src

CMD ["py/python3", "py/train.py"]