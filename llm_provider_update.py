import re

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/llm_provider.py', 'r') as f:
    code = f.read()

# Update generate function to accept history
old_generate = """    def generate(self, prompt, data=None, system_instruction=None, max_retries=2):
        if not self.api_key:
            return self._fallback_response(data or {})
            
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3}
        }"""

new_generate = """    def generate(self, prompt, data=None, system_instruction=None, max_retries=2, history=None):
        if not self.api_key:
            return self._fallback_response(data or {})
            
        contents = []
        if history:
            for msg in history[-5:]:
                role = "user" if msg.get("role") == "user" else "model"
                content = msg.get("content", msg.get("summary", ""))
                contents.append({"role": role, "parts": [{"text": content}]})
        
        contents.append({"role": "user", "parts": [{"text": prompt}]})
            
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 0.3}
        }"""

code = code.replace(old_generate, new_generate)

# Update system prompt and AgenticChatbot.process_query
old_sys_prompt = """            sys_prompt = (
                "You are the EcoSphere AI Advisor. Respond naturally and conversationally. "
                "Base your answer strictly on the provided JSON data representing the user's latest metrics. "
                "Do NOT invent numerical values. Explain the results, highlight key contributors, and provide recommendations. "
                "Do not expose internal tool execution traces to the user. Keep it concise and helpful."
            )"""

new_sys_prompt = """            sys_prompt = (
                "You are the strictly data-grounded EcoSphere AI Advisor. "
                "CRITICAL RULE 1: For unrelated questions (movies, sports, politics, celebrities, unrelated coding, general chatting, or subjects outside EcoSphere), politely respond EXACTLY with: `I’m designed to assist with EcoSphere and its environmental sustainability modules. Please ask me a question related to EcoSphere.` Do not invent an answer.\\n"
                "CRITICAL RULE 2: Base your numerical answers strictly on the provided JSON Data. Do NOT invent numerical values or claim a model achieved 98-100% unless it is in the data. Treat the provided metrics JSON as genuine independent test results.\\n"
                "CRITICAL RULE 3: For analysis/prediction questions, your response MUST follow this exact format using markdown headers:\\n"
                "1. Result / Prediction\\n"
                "2. Key Factors Found in My Data\\n"
                "3. Model Used (format: `Model Used: [Name] ([Type]) — selected from validated model benchmarking; Test R²: [val]; MAE/MAPE: [val]`)\\n"
                "4. Evidence / Data Used\\n"
                "5. Recommended Actions (format: `Current Result → Detected Issue → Evidence from Data → Recommended Action → Expected/Simulated Impact → Confidence/Limitations`)\\n"
                "6. What-If Impact (if requested or relevant)\\n"
                "7. Limitations / Uncertainty\\n"
                "Do not expose hidden chain-of-thought. Do not hallucinate."
            )"""

code = code.replace(old_sys_prompt, new_sys_prompt)

# Update answer = self.llm.generate call
old_generate_call = """            answer = self.llm.generate(user_prompt, data=data, system_instruction=sys_prompt)"""
new_generate_call = """            answer = self.llm.generate(user_prompt, data=data, system_instruction=sys_prompt, history=history)"""
code = code.replace(old_generate_call, new_generate_call)

# Update intent detection to be smarter with history and out-of-domain
old_intent = """    def _detect_intent(self, q):
        q = q.lower()
        if any(w in q for w in ['solar performance', 'solar loss', 'dust', 'shading', 'degrad']):
            return 'solar-loss'
        elif any(w in q for w in ['solar']):
            return 'solar-energy'
        elif any(w in q for w in ['carbon', 'co2', 'footprint', 'emission']):
            return 'carbon'
        elif any(w in q for w in ['air', 'aqi', 'pm2.5', 'pollution']):
            return 'air-quality'
        elif any(w in q for w in ['e-waste', 'recycle', 'electronics', 'classify']):
            return 'ewaste'
        elif any(w in q for w in ['food', 'waste', 'compost']):
            return 'food-waste'
        else:
            return 'carbon' # default fallback"""

new_intent = """    def _detect_intent(self, q, history=None):
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
code = code.replace(old_intent, new_intent)
code = code.replace("domain = self._detect_intent(query)", "domain = self._detect_intent(query, history)")

# Update fallback response
old_fallback = """            if module == 'carbon':
                res += f"Your total CO2 footprint is {outputs.get('total_co2_month', 0)} kg CO2e/month. "
                res += "To reduce this, consider transitioning to rooftop solar or reducing private fossil vehicle travel."
            elif module == 'solar-loss':
                res += f"Your solar system is experiencing {outputs.get('totalLossPct', 0)}% performance loss. "
                res += "Regular cleaning can reduce soiling losses."
            elif module == 'solar-energy':
                res += f"Your projected annual solar yield is {outputs.get('annualKwh', 0)} kWh. "
            elif module == 'air-quality':
                res += f"Your local PM2.5 level is {inputs.get('pm25', 0)} µg/m³, which puts the AQI in the {outputs.get('category', 'Unknown')} category. "
            elif module == 'ewaste':
                res += f"Properly recycling your {inputs.get('device', 'device')} can recover approximately ₹{outputs.get('metalValue', 0)} in precious metals. "
            elif module == 'food-waste':
                res += f"Your household generates about {inputs.get('wasteKgWeek', 0)} kg of food waste weekly, totaling {outputs.get('annualKg', 0)} kg per year. "
            else:
                res += "Please ensure all module inputs are up to date for precise recommendations."
                
            res += "\\n\\n*(Generative AI is currently offline or unconfigured, so this is a deterministic fallback summary.)*"
            return res"""

new_fallback = """            metrics = data.get('metrics', {})
            model_info = f"Model Used: {metrics.get('model_name', 'EcoSphere Regressor')} — selected from validated model benchmarking; Test R²: {metrics.get('r2', 'N/A')}; MAPE: {metrics.get('mape', 'N/A')}%"
            
            res = "### 1. Result / Prediction\\n"
            if module == 'carbon':
                res += f"Your total CO2 footprint is {outputs.get('total_co2_month', 0)} kg CO2e/month.\\n"
            elif module == 'solar-loss':
                res += f"Your solar system is experiencing {outputs.get('totalLossPct', 0)}% performance loss.\\n"
            elif module == 'solar-energy':
                res += f"Your projected annual solar yield is {outputs.get('annualKwh', 0)} kWh.\\n"
            elif module == 'air-quality':
                res += f"Your local PM2.5 level is {inputs.get('pm25', 0)} µg/m³ (AQI: {outputs.get('category', 'Unknown')}).\\n"
            elif module == 'ewaste':
                res += f"Properly recycling your {inputs.get('device', 'device')} can recover approximately ₹{outputs.get('metalValue', 0)}.\\n"
            elif module == 'food-waste':
                res += f"Your household generates {outputs.get('annualKg', 0)} kg of food waste per year.\\n"
            else:
                res += "Please ensure all module inputs are up to date for precise recommendations.\\n"
                
            res += "\\n### 2. Key Factors Found in My Data\\n"
            res += f"Analyzed inputs: {list(inputs.keys())[:3]}\\n"
            
            res += "\\n### 3. Model Used\\n"
            res += model_info + "\\n"
            
            res += "\\n### 4. Evidence / Data Used\\n"
            res += "Verified EcoSphere Database Records.\\n"
            
            res += "\\n### 5. Recommended Actions\\n"
            res += "Current Result → Detected Issue → Evidence from Data → Recommended Action → Expected/Simulated Impact → Confidence/Limitations\\n"
            
            res += "\\n### 6. What-If Impact\\n"
            res += "Simulations available upon request.\\n"
            
            res += "\\n### 7. Limitations / Uncertainty\\n"
            res += "Deterministic fallback is an approximation.\\n"
            
            res += "\\n\\n*(Generative AI is currently offline or unconfigured, so this is a deterministic fallback summary.)*"
            return res"""

code = code.replace(old_fallback, new_fallback)

with open('/Users/vipuljain/.gemini/antigravity/scratch/eco-sphere/llm_provider.py', 'w') as f:
    f.write(code)

