import re

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/test_chatbot.py', 'r') as f:
    code = f.read()
    
old_q8 = """    def test_q8(self):
        # "Analyze everything" defaults to carbon or needs a multi-module intent
        self._run_test("Analyze everything.")"""
new_q8 = """    def test_q8(self):
        # "Analyze everything" defaults to carbon or needs a multi-module intent
        self._run_test("Analyze everything.")
        
    def test_q9(self):
        res = self.chatbot.process_query("Who won yesterday's cricket match?", [])
        self.assertTrue(res['success'])
        self.assertIn("designed to assist with EcoSphere", res['answer'])
        self.assertNotIn("Model Used", res['answer'])"""

code = code.replace(old_q8, new_q8)

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/test_chatbot.py', 'w') as f:
    f.write(code)

