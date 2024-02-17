import MicroServiceDL
from fastapi.testclient import TestClient
from datetime import datetime

start_time = datetime.now()

client = TestClient(MicroServiceDL.app)

user_id = 1000 
time = datetime(2021, 12, 20)

try: 
    r = client.get(f"/post/recommendations/",
                   params={'id': user_id, 'time': time, 'limit': 5},
                   )

    
except Exception as e: 
    raise ValueError(F'ERROR')

print(r.json())
print(f"--- {start_time} ---")

print(f"--- {(datetime.now() - start_time).total_seconds()} ---")


    