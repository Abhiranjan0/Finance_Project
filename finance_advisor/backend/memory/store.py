from finance_advisor.backend.db.redis_client import save_session_memory, get_session_memory
import json

class MemoryStore:

    def save_entity(self, session_id, data):
        existing = self.get_entity(session_id)  # returns dict now
        updated = {**existing, **data}          # merge
        save_session_memory(session_id, "entity", json.dumps(updated))

    def get_entity(self, session_id):
        raw = get_session_memory(session_id, "entity")
        if raw:
            try:
                return json.loads(raw)
            except:
                return {}
        return {}

    def save_summary(self, session_id, data):
        save_session_memory(session_id, "summary", data)

    def get_summary(self, session_id):
        return get_session_memory(session_id, "summary")

memory_store = MemoryStore()
