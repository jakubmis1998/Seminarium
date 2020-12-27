from multiprocessing import cpu_count

daemon = False
worker_class = "gevent"
loglevel = "info"
bind = "0.0.0.0:4444"
timeout = 300
workers = cpu_count() + 2 + 1
