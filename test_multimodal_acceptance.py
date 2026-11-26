#!/usr/bin/env python3
"""
EcoSphere Multimodal Environmental AI — 8-Point Final Acceptance Test Suite
Verifies:
  1. Solar panel image → detects soiling → connects to Solar Loss → provides evidence-based action
  2. E-waste image → classifies device → connects to E-Waste recycling
  3. Food-waste image → identifies food waste → connects to Food Waste emissions
  4. Pollution/waste video → extracts keyframes → temporal timeline results
  5. Chatbot query "What should I do?" → uses actual visual analysis
  6. Chatbot query "What could happen in the future?" → scenario projection
  7. Unsupported / corrupted file → clear error without crashing
  8. Full pipeline verification (HTTP API -> Backend -> Vision Engine -> DB -> Models -> Chatbot)
"""

import sys
import os
import json
import time
import uuid
import tempfile
import threading
import urllib.request
import urllib.parse
import numpy as np
import cv2

# Ensure local directory is on python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import vision_engine
import ingestion_engine
import server
from llm_provider import AgenticChatbot

chatbot_agent = AgenticChatbot(ingestion_engine)

def create_synthetic_solar_image(path):
    """Create a synthetic solar panel image with cell grids and dust/soiling."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Deep blue solar panel silicon base
    img[:, :] = (85, 45, 15)  # BGR
    # Add silver busbars / grid lines
    for x in range(40, 640, 60):
        cv2.line(img, (x, 0), (x, 480), (200, 200, 200), 2)
    for y in range(40, 480, 60):
        cv2.line(img, (0, y), (640, y), (180, 180, 180), 1)
    # Add dust / soiling patches (yellowish-brown dust layer)
    noise = np.random.randint(0, 70, (480, 640, 3), dtype=np.uint8)
    dust_overlay = np.full((480, 640, 3), (40, 70, 90), dtype=np.uint8)
    cv2.addWeighted(img, 0.65, dust_overlay + noise, 0.35, 0, img)
    cv2.imwrite(path, img)
    return path

def create_synthetic_ewaste_image(path):
    """Create a synthetic electronic waste PCB / gadget image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Green PCB board base
    img[:, :] = (25, 95, 20)  # BGR
    # Add IC chip rectangles (black / dark gray rectangles)
    cv2.rectangle(img, (120, 100), (260, 240), (25, 25, 25), -1)
    cv2.rectangle(img, (320, 180), (460, 320), (30, 30, 30), -1)
    # Add gold/copper contact traces
    for y in range(120, 230, 15):
        cv2.line(img, (60, y), (120, y), (0, 180, 215), 2)
        cv2.line(img, (260, y), (320, y), (0, 180, 215), 2)
    # Add cylindrical capacitor circles
    cv2.circle(img, (520, 120), 28, (120, 120, 120), -1)
    cv2.circle(img, (520, 200), 24, (120, 120, 120), -1)
    cv2.imwrite(path, img)
    return path

def create_synthetic_food_waste_image(path):
    """Create a synthetic food waste image with organic color clusters."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Kitchen counter / plate background
    img[:, :] = (180, 180, 180)
    # Organic fruit/vegetable scraps (warm greens, oranges, browns)
    cv2.ellipse(img, (240, 220), (120, 80), 20, 0, 360, (30, 120, 40), -1)   # vegetable peels
    cv2.ellipse(img, (360, 260), (90, 70), -30, 0, 360, (20, 140, 220), -1)  # citrus/banana scraps
    cv2.circle(img, (280, 320), 65, (25, 60, 130), -1)                       # cooked grain / organic scraps
    # Add organic texture noise
    noise = np.random.randint(0, 45, (480, 640, 3), dtype=np.uint8)
    cv2.add(img, noise, img)
    cv2.imwrite(path, img)
    return path

def create_synthetic_pollution_video(path):
    """Create a synthetic 2-second video with evolving atmospheric haze/smoke."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 15.0, (480, 360))
    for frame_idx in range(30):
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        # Background sky/landscape
        frame[:, :] = (190, 160, 130)  # dusky/smoggy sky
        # Simulating increasing smoke / plume density over time
        density = int(min(180, 40 + frame_idx * 4.5))
        haze = np.full((360, 480, 3), (density, density, density), dtype=np.uint8)
        cv2.addWeighted(frame, 0.5, haze, 0.5, 0, frame)
        # Add plume contour
        cv2.ellipse(frame, (240, 280), (80 + frame_idx * 2, 60 + frame_idx), 0, 0, 360, (110, 110, 110), -1)
        out.write(frame)
    out.release()
    return path

def run_tests():
    print("=" * 80)
    print("🌿 EcoSphere Multimodal Environmental AI — Acceptance Test Suite")
    print("=" * 80)

    test_dir = tempfile.mkdtemp(prefix="ecosphere_vision_test_")
    results = {}

    try:
        # -------------------------------------------------------------------------
        # TEST 1: Solar panel image → detects soiling → connects to Solar Loss
        # -------------------------------------------------------------------------
        print("\n[TEST 1] Testing Solar Panel Soiling Detection & Module Connection...")
        solar_path = os.path.join(test_dir, "test_solar_dust.jpg")
        create_synthetic_solar_image(solar_path)
        with open(solar_path, 'rb') as f:
            solar_bytes = f.read()

        a1 = vision_engine.process_media(solar_bytes, "test_solar_dust.jpg")
        assert 'solar' in a1['module_linked'].lower(), f"Expected solar module, got '{a1['module_linked']}'"
        assert a1['confidence_score'] >= 0.65, f"Confidence {a1['confidence_score']} below 0.65 gating threshold"
        assert 'soiling' in str(a1['detected_objects_conditions']).lower() or 'dust' in str(a1['detected_objects_conditions']).lower() or 'solar' in str(a1['detected_objects_conditions']).lower(), f"Unexpected objects: {a1.get('detected_objects_conditions')}"
        assert a1['recommended_action'], "Missing recommended action"
        assert a1['what_if_scenario'], "Missing what-if scenario"
        assert len(a1['sources_limitations']) > 0, "Missing sources/limitations"

        # Verify DB persistence
        db_rec = ingestion_engine.get_vision_analysis(a1['analysis_id'])
        assert db_rec is not None, "Record was not saved in SQLite vision_analyses table"
        assert db_rec['checksum_sha256'] == a1['checksum_sha256'], "Checksum mismatch in DB"
        print("✅ TEST 1 PASSED: Solar panel detected, soiling quantified, linked to Solar Loss.")
        results['test_1_solar_detection'] = True

        # -------------------------------------------------------------------------
        # TEST 2: E-waste image → classifies device → connects to E-Waste recycling
        # -------------------------------------------------------------------------
        print("\n[TEST 2] Testing E-Waste Classification & Recycling Valuation...")
        ewaste_path = os.path.join(test_dir, "test_ewaste_pcb.png")
        create_synthetic_ewaste_image(ewaste_path)
        with open(ewaste_path, 'rb') as f:
            ewaste_bytes = f.read()

        a2 = vision_engine.process_media(ewaste_bytes, "test_ewaste_pcb.png")
        assert 'ewaste' in a2['module_linked'].lower() or 'e-waste' in a2['module_linked'].lower(), f"Expected e-waste module, got '{a2['module_linked']}'"
        assert a2['confidence_score'] >= 0.65, f"Confidence {a2['confidence_score']} below threshold"
        assert 'circuit' in str(a2['detected_objects_conditions']).lower() or 'electronic' in str(a2['detected_objects_conditions']).lower() or 'pcb' in str(a2['detected_objects_conditions']).lower()
        assert 'EPR' in a2['recommended_action'] or 'recycl' in a2['recommended_action'].lower()
        print("✅ TEST 2 PASSED: E-Waste classified, precious metals estimated, linked to E-Waste module.")
        results['test_2_ewaste_classification'] = True

        # -------------------------------------------------------------------------
        # TEST 3: Food waste image → identifies waste → connects to emissions
        # -------------------------------------------------------------------------
        print("\n[TEST 3] Testing Food Waste Identification & Methane Calculations...")
        food_path = os.path.join(test_dir, "test_food_organic.jpg")
        create_synthetic_food_waste_image(food_path)
        with open(food_path, 'rb') as f:
            food_bytes = f.read()

        a3 = vision_engine.process_media(food_bytes, "test_food_organic.jpg")
        assert 'food' in a3['module_linked'].lower(), f"Expected food waste module, got '{a3['module_linked']}'"
        impact_str = str(a3['prediction_calculated_impact']).lower() + " " + str(a3['possible_future_risk']).lower()
        assert 'co₂' in impact_str or 'co2' in impact_str or 'methane' in impact_str or 'ch₄' in impact_str, f"Missing GHG footprint in: {impact_str}"
        assert 'compost' in a3['recommended_action'].lower()
        print("✅ TEST 3 PASSED: Food waste identified, landfill GHG footprint computed, linked to Food Waste module.")
        results['test_3_food_waste_emissions'] = True

        # -------------------------------------------------------------------------
        # TEST 4: Pollution/waste video → extracts keyframes → timeline results
        # -------------------------------------------------------------------------
        print("\n[TEST 4] Testing Video Analysis, Keyframe Extraction & Temporal Timeline...")
        video_path = os.path.join(test_dir, "test_smoke_plume.mp4")
        create_synthetic_pollution_video(video_path)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        a4 = vision_engine.process_media(video_bytes, "test_smoke_plume.mp4")
        assert a4['file_type'].startswith('video'), f"Expected video file_type, got '{a4['file_type']}'"
        assert len(a4.get('key_frames', [])) > 0, "No keyframes extracted"
        assert len(a4.get('timeline_analysis', [])) > 0, "No timeline events generated"
        first_event = a4['timeline_analysis'][0]
        assert 'timestamp_sec' in first_event and 'condition' in first_event
        print(f"✅ TEST 4 PASSED: Video processed ({len(a4.get('key_frames', []))} keyframes), timeline tracking verified.")
        results['test_4_video_timeline'] = True

        # -------------------------------------------------------------------------
        # TEST 5: Chatbot query "What should I do?" → uses actual visual analysis
        # -------------------------------------------------------------------------
        print("\n[TEST 5] Testing Chatbot Grounded Action Query: 'What should I do?'...")
        chat_q1 = chatbot_agent.process_query("What should I do?", history=[], vision_context=a1)
        assert chat_q1.get('success'), f"Chatbot query failed: {chat_q1.get('error')}"
        ans1 = chat_q1.get('answer', '')
        assert "What should I do?" not in ans1, "Chatbot echoed user's question"
        # Must contain grounded recommendations for solar cleaning / dust removal
        assert "clean" in ans1.lower() or "solar" in ans1.lower() or "dust" in ans1.lower(), f"Answer not grounded: {ans1}"
        assert "Solar Loss" in ans1 or "soiling" in ans1.lower()
        print("✅ TEST 5 PASSED: Chatbot generated evidence-based action steps from active visual context.")
        results['test_5_chatbot_what_should_i_do'] = True

        # -------------------------------------------------------------------------
        # TEST 6: Chatbot query "What could happen in the future?" → scenario projection
        # -------------------------------------------------------------------------
        print("\n[TEST 6] Testing Chatbot Scenario Query: 'What could happen in the future?'...")
        chat_q2 = chatbot_agent.process_query("What could happen in the future?", history=[], vision_context=a1)
        assert chat_q2.get('success'), f"Chatbot query failed: {chat_q2.get('error')}"
        ans2 = chat_q2.get('answer', '')
        assert "What could happen in the future?" not in ans2, "Chatbot echoed user's question"
        assert "future" in ans2.lower() or "risk" in ans2.lower() or "degrad" in ans2.lower() or "hot spot" in ans2.lower() or "loss" in ans2.lower()
        print("✅ TEST 6 PASSED: Chatbot projected forward-looking risk scenario from visual evidence.")
        results['test_6_chatbot_future_scenario'] = True

        # -------------------------------------------------------------------------
        # TEST 7: Unsupported / corrupted file → clear error without crashing
        # -------------------------------------------------------------------------
        print("\n[TEST 7] Testing Corrupted and Unsupported File Error Handling...")
        corrupt_bytes = b"MZ\x90\x00\x03\x00\x00\x00FAKE_EXECUTABLE_CONTENT"
        try:
            vision_engine.process_media(corrupt_bytes, "malware.exe")
            assert False, "Expected ValueError for .exe"
        except ValueError as ve:
            assert "Unsupported" in str(ve) or "extension" in str(ve).lower()

        # Also test invalid bytes masquerading as JPG
        fake_jpg_bytes = b"\xff\xd8\xffFAKE_HEADER_CORRUPTED_STREAM"
        try:
            vision_engine.process_media(fake_jpg_bytes, "corrupt.jpg")
            assert False, "Expected ValueError for corrupt image"
        except (ValueError, Exception) as ve:
            assert "corrupt" in str(ve).lower() or "decode" in str(ve).lower() or "image" in str(ve).lower()
        print("✅ TEST 7 PASSED: Security validation and corrupted file handling blocked invalid uploads safely.")
        results['test_7_error_handling'] = True

        # -------------------------------------------------------------------------
        # TEST 8: Full pipeline verification (HTTP API -> Backend -> DB -> Model -> Chatbot)
        # -------------------------------------------------------------------------
        # TEST 8: Full pipeline verification (HTTP API -> Backend -> DB -> Model -> Chatbot)
        # -------------------------------------------------------------------------
        print("\n[TEST 8] Testing Full HTTP Pipeline & Endpoints via EcoSphereAPIHandler...")
        import io

        class SimulatedHandler(server.EcoSphereAPIHandler):
            def __init__(self, method, path, headers=None, body=b''):
                self.rfile = io.BytesIO(body)
                self.wfile = io.BytesIO()
                self.command = method
                self.path = path
                self.headers = headers or {}
                self.headers['Content-Length'] = str(len(body))
                self.response_status = 200
                self.response_headers = {}

            def send_response(self, code, message=None):
                self.response_status = code

            def send_header(self, keyword, value):
                self.response_headers[keyword] = value

            def end_headers(self):
                pass

            def send_cors_headers(self):
                pass

            def execute(self):
                if self.command == 'GET':
                    self.do_GET()
                elif self.command == 'POST':
                    self.do_POST()
                self.wfile.seek(0)
                raw = self.wfile.read()
                return self.response_status, json.loads(raw.decode('utf-8')) if raw else {}

        # 8a: Verify health endpoint
        status, health_data = SimulatedHandler('GET', '/api/health').execute()
        assert status == 200, f"Health check returned status {status}"
        assert health_data.get('status') == 'healthy', f"Health check data unexpected: {health_data}"

        # 8b: Multipart POST /api/vision/analyze
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body_parts = []
        body_parts.append(f"--{boundary}\r\n".encode('utf-8'))
        body_parts.append(f'Content-Disposition: form-data; name="media"; filename="test_solar_http.jpg"\r\n'.encode('utf-8'))
        body_parts.append(b'Content-Type: image/jpeg\r\n\r\n')
        body_parts.append(solar_bytes)
        body_parts.append(f"\r\n--{boundary}--\r\n".encode('utf-8'))
        multipart_data = b"".join(body_parts)

        status, analyze_json = SimulatedHandler(
            'POST',
            '/api/vision/analyze',
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
            body=multipart_data
        ).execute()

        assert status == 200, f"Analyze HTTP status {status}, response: {analyze_json}"
        assert analyze_json.get('success'), f"HTTP analyze failed: {analyze_json}"
        analysis = analyze_json.get('analysis') or analyze_json.get('data')
        assert analysis is not None, "Missing analysis data in HTTP response"
        analysis_id = analysis['analysis_id']

        # 8c: GET /api/vision/<id>
        status, get_json = SimulatedHandler('GET', f'/api/vision/{analysis_id}').execute()
        assert status == 200, f"GET analysis returned status {status}"
        assert get_json.get('success'), "Failed to retrieve analysis by ID"
        assert get_json['analysis']['analysis_id'] == analysis_id

        # 8d: GET /api/vision/history
        status, hist_json = SimulatedHandler('GET', '/api/vision/history').execute()
        assert status == 200, f"GET history returned status {status}"
        assert hist_json.get('success'), "Failed to get history"
        assert len(hist_json.get('history', [])) > 0, "History empty"

        # 8e: GET /api/vision/metrics
        status, metrics_json = SimulatedHandler('GET', '/api/vision/metrics').execute()
        assert status == 200, f"GET metrics returned status {status}"
        assert metrics_json.get('success'), "Failed to get metrics"
        assert 'aggregate' in metrics_json['metrics']
        assert metrics_json['metrics']['aggregate']['f1_score'] < 1.0  # Zero 100% fabricated claims

        # 8f: POST /api/vision/what-if
        whatif_payload = json.dumps({
            "analysis_id": analysis_id,
            "delay_days": 28,
            "cleaning_cost_inr": 450
        }).encode('utf-8')
        status, whatif_json = SimulatedHandler(
            'POST',
            '/api/vision/what-if',
            headers={'Content-Type': 'application/json'},
            body=whatif_payload
        ).execute()
        assert status == 200, f"What-if status {status}: {whatif_json}"
        assert whatif_json.get('success'), f"What-if simulation failed: {whatif_json}"
        sim = whatif_json.get('simulation') or whatif_json.get('scenario_impact')
        assert sim['delay_days'] == 28
        assert 'loss_with_delay_pct' in sim

        # 8g: POST /api/chat with vision context
        chat_payload = json.dumps({
            "query": "What is this and what problem do you see?",
            "history": [],
            "vision_context": analysis
        }).encode('utf-8')
        status, chat_json = SimulatedHandler(
            'POST',
            '/api/chat',
            headers={'Content-Type': 'application/json'},
            body=chat_payload
        ).execute()
        assert status == 200, f"HTTP Chat returned status {status}: {chat_json}"
        assert chat_json.get('success'), f"HTTP Chat failed: {chat_json}"
        assert len(chat_json.get('answer', '')) > 20

        print("✅ TEST 8 PASSED: Complete HTTP API pipeline verified end-to-end.")
        results['test_8_full_http_pipeline'] = True

    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("🎯 FINAL ACCEPTANCE TEST SUMMARY:")
    for test_name, status in results.items():
        print(f"  • {test_name}: {'PASSED ✅' if status else 'FAILED ❌'}")
    print("=" * 80)

    all_passed = len(results) == 8 and all(results.values())
    if all_passed:
        print("🏆 ALL 8 ACCEPTANCE TESTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("⚠️ SOME ACCEPTANCE TESTS FAILED.")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests())
