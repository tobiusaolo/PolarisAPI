import uuid
from collections import defaultdict

user_conversations = defaultdict(dict)
def create_new_conversation(user_id: str):
    conversation_id = str(uuid.uuid4())
    user_conversations[user_id][conversation_id] = []
    return conversation_id