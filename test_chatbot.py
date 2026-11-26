import unittest
import ingestion_engine
from llm_provider import AgenticChatbot

class TestEcoSphereFinalChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ingestion_engine
        cls.bot = AgenticChatbot(cls.engine)

    def test_flow_1_greeting(self):
        res = self.bot.process_query("Hello")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'GREETING')
        self.assertIn("Hello! I’m EcoSphere AI", res['answer'])
        self.assertIsNone(res['data_used'])
        self.assertIn("GreetingClassifier", res['tools_used'])

    def test_flow_2_carbon_footprint_high(self):
        res = self.bot.process_query("Why is my carbon footprint high?")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'ECOSPHERE_RELATED')
        self.assertEqual(res['data_used']['module'], 'carbon')
        self.assertIn("kg CO₂e", res['answer'])
        self.assertIn("Home Electricity", res['answer'])
        self.assertIn("Stacking Meta-Ensemble", res['answer'])
        self.assertIn("Model Used", res['answer'])
        self.assertIn("Recommended Actions", res['answer'])
        self.assertNotIn("offline or unconfigured", res['answer'])

    def test_flow_3_reduce_solar_loss(self):
        res = self.bot.process_query("How can I reduce solar loss?")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'ECOSPHERE_RELATED')
        self.assertEqual(res['data_used']['module'], 'solar-loss')
        self.assertIn("performance deficit", res['answer'])
        self.assertIn("Dust/Soiling", res['answer'])
        self.assertIn("Shading", res['answer'])
        self.assertIn("Recommended Actions", res['answer'])
        self.assertNotIn("offline or unconfigured", res['answer'])

    def test_flow_4_solar_loss_6_months_future(self):
        res = self.bot.process_query("What will happen if my solar loss continues for the next 6 months?")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'ECOSPHERE_RELATED')
        self.assertEqual(res['data_used']['module'], 'solar-loss')
        self.assertIn("Future Prediction / Possible Problem", res['answer'])
        self.assertIn("Current Situation → Future Prediction → Possible Future Problem/Risk → Evidence from My Data → Uncertainty → Recommended Action", res['answer'])
        self.assertIn("TimeSeriesRiskForecaster", res['tools_used'])
        self.assertNotIn("offline or unconfigured", res['answer'])

    def test_flow_5_what_if_electricity_reduction(self):
        res = self.bot.process_query("What if I reduce electricity consumption by 20%?")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'ECOSPHERE_RELATED')
        self.assertEqual(res['data_used']['module'], 'carbon')
        self.assertIn("What-If Scenario", res['answer'])
        self.assertIn("Current Value → Scenario Change → New Estimated Value → Difference → Possible Impact → Recommendation", res['answer'])
        self.assertIn("kg CO₂e/month", res['answer'])
        self.assertIn("WhatIfScenarioSimulator", res['tools_used'])
        self.assertNotIn("offline or unconfigured", res['answer'])

    def test_flow_6_which_model_solar_prediction(self):
        res = self.bot.process_query("Which model was used for my solar prediction?")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'ECOSPHERE_RELATED')
        self.assertEqual(res['data_used']['module'], 'solar-energy')
        self.assertIn("Model Used", res['answer'])
        self.assertIn("R²", res['answer'])
        self.assertNotIn("offline or unconfigured", res['answer'])

    def test_flow_7_unrelated_movie(self):
        res = self.bot.process_query("Tell me about a movie.")
        self.assertTrue(res['success'])
        self.assertEqual(res['mode'], 'UNRELATED')
        self.assertEqual(res['answer'], "Sorry, I don’t know about this topic. I can help you with EcoSphere and its environmental sustainability modules.")
        self.assertIsNone(res['data_used'])
        self.assertIn("GuardrailClassifier", res['tools_used'])

    def test_flow_8_follow_up_memory(self):
        history = [
            {"role": "user", "content": "How can I reduce solar loss?"},
            {"role": "assistant", "content": "Your solar loss is 22.4%...", "domain": "solar-loss"}
        ]
        res = self.bot.process_query("What caused it?", history)
        self.assertTrue(res['success'])
        self.assertEqual(res['data_used']['module'], 'solar-loss')

if __name__ == '__main__':
    unittest.main()
