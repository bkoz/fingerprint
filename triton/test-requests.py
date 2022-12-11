import requests
import logging

logging.basicConfig(level=logging.INFO)

host = "http://fingerprint-sandbox.apps.hou.edgelab.online"
url = f'{host}/v2'
r = requests.get(url)
if r:
    logging.info(f'{r.content.decode()}')
else:
    logging.error(f"Requests {r}")
