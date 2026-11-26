import re

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/llm_provider.py', 'r') as f:
    code = f.read()

old_intent = """    def _detect_intent(self, q, history=None):
        q = q.lower()
        if any(w in q for w in ['solar performance', 'solar loss', 'dust', 'shading', 'degrad']):
            return 'solar-loss'
        elif any(w in q for w in ['solar']):
            return 'solar-energy'
        elif any(w in q for w in ['carbon', 'co2', 'footprint', 'emission']):
            return 'carbon'
        elif any(w in q for w in ['air', 'aqi', 'pm2.5', 'pollution']):
            return 'air-quality'
        elif any(w in q for w in ['e-waste', 'recycle', 'electronics', 'classify', 'phone']):
            return 'ewaste'
        elif any(w in q for w in ['food', 'waste', 'compost']):
            return 'food-waste'
        
        # Check history for context if it's a follow-up
        if history:
            for msg in reversed(history):
                if msg.get('role') == 'assistant' and msg.get('domain'):
                    return msg.get('domain')
                    
        return 'carbon' # default fallback"""

new_intent = """    def _detect_intent(self, q, history=None):
        q_lower = q.lower()
        
        # OOD check
        ood_words = ['movie', 'sport', 'cricket', 'football', 'politics', 'celebrity', 'who won', 'who is', 'president', 'actor', 'music']
        if any(w in q_lower for w in ood_words):
            return 'out-of-domain'
            
        if any(w in q_lower for w in ['solar performance', 'solar loss', 'dust', 'shading', 'degrad']):
            return 'solar-loss'
        elif any(w in q_lower for w in ['solar']):
            return 'solar-energy'
        elif any(w in q_lower for w in ['carbon', 'co2', 'footprint', 'emission']):
            return 'carbon'
        elif any(w in q_lower for w in ['air', 'aqi', 'pm2.5', 'pollution']):
            return 'air-quality'
        elif any(w in q_lower for w in ['e-waste', 'recycle', 'electronics', 'classify', 'phone']):
            return 'ewaste'
        elif any(w in q_lower for w in ['food', 'waste', 'compost']):
            return 'food-waste'
            
        if 'analyze everything' in q_lower:
            return 'all'
            
        # Check history for context if it's a follow-up
        if history:
            for msg in reversed(history):
                if msg.get('role') == 'assistant' and msg.get('domain'):
                    return msg.get('domain')
                    
        return 'carbon' # default fallback"""

code = code.replace(old_intent, new_intent)

old_process = """            # 1. Intent Detection
            domain = self._detect_intent(query, history)
            
            # 2. Retrieve Latest Data & Validate
            data = self.engine.get_latest_module_calculation(domain)"""

new_process = """            # 1. Intent Detection
            domain = self._detect_intent(query, history)
            
            if domain == 'out-of-domain':
                return {
                    "success": True,
                    "answer": "I’m designed to assist with EcoSphere and its environmental sustainability modules. Please ask me a question related to EcoSphere.",
                    "sources": [],
                    "data_used": None,
                    "tools_used": ["IntentDetection"],
                    "model_info": None,
                    "warnings": [],
                    "request_id": req_id
                }
                
            # 2. Retrieve Latest Data & Validate
            if domain == 'all':
                data = self.engine.get_latest_module_calculation('carbon') # fallback to carbon for 'all' to show something
                domain = 'carbon'
            else:
                data = self.engine.get_latest_module_calculation(domain)"""
code = code.replace(old_process, new_process)

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/llm_provider.py', 'w') as f:
    f.write(code)

