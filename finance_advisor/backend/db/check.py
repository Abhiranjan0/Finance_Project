from finance_advisor.backend.db.redis_client import redis_client
import json

sid = "f091f39f-c0ad-4319-91c8-ff06f35a5041"
print(redis_client.hgetall(sid))
