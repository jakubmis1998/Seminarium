1. Clone git repository:
```git clone https://gitlab.mga.com.pl/ceuo/ceuo.git```

1. Create virtual environment:
```python3 -m venv venv```

1. Active environment:
```source ./venv/bin/activate```

1. Upgrade pip
```cd project```
```pip install --upgrade pip```

1. Install wheel
```pip install wheel```

1. Install all requirements
```pip install -r requirements.txt```

1. Create and migrate database
```python manage.py makemigrations```
```python manage.py migrate```

1. Create superuser account for django admin panel.
```python manage.py createsuperuser```

1. Run developer server
```python manage.py runserver 0.0.0.0:4444```

To develop server with gunicorn:
```gunicorn -c gunicorn.conf.py project.wsgi```
