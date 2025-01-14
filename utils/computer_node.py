import os
import uuid
import json
def get_node_id():
    # Path to the file where the UUID is saved
    uuid_file = "node_id.json"

    # Check if the file exists
    if os.path.exists(uuid_file):
        # Load the UUID from the file
        with open(uuid_file, "r") as f:
            return json.load(f)["node_id"]

    # Generate a new UUID if the file doesn't exist
    node_id = str(uuid.uuid4())
    with open(uuid_file, "w") as f:
        json.dump({"node_id": node_id}, f)

    return node_id