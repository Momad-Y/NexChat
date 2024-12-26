class ChatMemory:
    def __init__(self):
        self.history = []

    def add_message(self, sender, message):
        self.history.append({"sender": sender, "message": message})

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []