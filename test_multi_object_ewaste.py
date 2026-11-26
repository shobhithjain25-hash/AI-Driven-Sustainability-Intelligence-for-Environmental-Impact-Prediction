"""
Test Suite: Multi-Object E-Waste Vision AI System
Tests:
  1. EWASTE_REFERENCE_DB integrity and calculate_inventory_aggregates calculations
  2. Multi-object image detection with bounding boxes, object IDs, and gating
  3. Video multi-object tracking and spatial de-duplication across keyframes
  4. Database persistence and non-destructive user corrections
  5. What-If scenario endpoint with dynamic recycling rate
  6. Grounded analytical chatbot responses across all 9 specified queries
"""

import sys
import os
import json
import unittest
import numpy as np
import cv2

# Ensure workspace is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import vision_engine
from vision_engine import (
    EWASTE_REFERENCE_DB,
    calculate_inventory_aggregates,
    compute_iou,
    nms_boxes,
    detect_multi_object_ewaste,
    track_video_multi_objects
)
import ingestion_engine
from ingestion_engine import (
    save_vision_analysis,
    get_vision_analysis,
    update_vision_inventory
)
import llm_provider


class TestMultiObjectEwaste(unittest.TestCase):

    def setUp(self):
        ingestion_engine.init_db()

    def test_01_reference_db_completeness(self):
        """Verify all 14 e-waste reference classes contain required schema fields."""
        required_keys = [
            "label", "category", "default_weight_kg", "embodied_factor_kg_co2_per_kg",
            "avoided_co2e_kg", "materials", "is_hazardous", "hazard_profile",
            "reusable", "pathway"
        ]
        expected_classes = [
            "laptop", "smartphone", "keyboard", "cables", "battery",
            "pcb", "desktop", "monitor", "tablet", "printer",
            "washing_machine", "refrigerator", "router", "tv"
        ]
        for dev in expected_classes:
            self.assertIn(dev, EWASTE_REFERENCE_DB, f"Missing device class {dev}")
            entry = EWASTE_REFERENCE_DB[dev]
            for rk in required_keys:
                self.assertIn(rk, entry, f"Missing field {rk} in {dev}")
            self.assertGreater(entry["default_weight_kg"], 0)
            self.assertGreater(entry["embodied_factor_kg_co2_per_kg"], 0)
            self.assertIsInstance(entry["materials"], dict)

    def test_02_calculate_inventory_aggregates(self):
        """Test mathematical aggregation across a realistic multi-object inventory."""
        sample_items = [
            {
                "object_id": "OBJ-001",
                "label": "Laptop Computer",
                "device_key": "laptop",
                "category": "Computing & IT",
                "actual_weight_kg": 2.2,
                "co2e_impact_kg": 2.2 * 140.0, # 308.0
                "avoided_co2e_kg": 65.0,
                "is_hazardous": False,
                "materials": {"gold_mg": 220.0, "silver_mg": 1000.0, "copper_g": 480.0, "aluminum_g": 350.0, "plastics_g": 550.0},
                "reusable": True
            },
            {
                "object_id": "OBJ-002",
                "label": "Smartphone",
                "device_key": "smartphone",
                "category": "Mobile Communications",
                "actual_weight_kg": 0.18,
                "co2e_impact_kg": 0.18 * 300.0, # 54.0
                "avoided_co2e_kg": 22.0,
                "is_hazardous": False,
                "materials": {"gold_mg": 35.0, "silver_mg": 350.0, "copper_g": 15.0, "aluminum_g": 25.0, "plastics_g": 45.0},
                "reusable": True
            },
            {
                "object_id": "OBJ-003",
                "label": "Lithium-Ion Battery Pack",
                "device_key": "battery",
                "category": "Energy Storage & Batteries",
                "actual_weight_kg": 0.60,
                "co2e_impact_kg": 0.60 * 70.0, # 42.0
                "avoided_co2e_kg": 8.5,
                "is_hazardous": True,
                "hazard_profile": "Contains cobalt, lithium, flammable organic electrolytes",
                "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 90.0, "aluminum_g": 80.0, "plastics_g": 60.0},
                "reusable": False
            },
            {
                "object_id": "OBJ-004",
                "label": "Copper Power / Data Cables",
                "device_key": "cables",
                "category": "Accessories & Cabling",
                "actual_weight_kg": 0.20,
                "co2e_impact_kg": 0.20 * 25.0, # 5.0
                "avoided_co2e_kg": 2.8,
                "is_hazardous": False,
                "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 120.0, "aluminum_g": 10.0, "plastics_g": 65.0},
                "reusable": False
            }
        ]

        aggr = calculate_inventory_aggregates(sample_items)
        self.assertEqual(aggr["total_objects"], 4)
        self.assertAlmostEqual(aggr["total_weight_kg"], 3.18, places=2)
        self.assertAlmostEqual(aggr["total_co2e_impact"], 409.0, places=1)
        self.assertAlmostEqual(aggr["total_co2e_avoided"], 98.3, places=1)
        self.assertEqual(aggr["hazardous_count"], 1)
        self.assertEqual(len(aggr["reusable_items"]), 2)
        self.assertEqual(len(aggr["recyclable_items"]), 2)
        self.assertGreater(aggr["materials_summary"]["copper_g"], 600.0)

    def test_03_compute_iou_and_nms(self):
        """Test bounding box IoU and non-maximum suppression."""
        boxA = [50, 50, 100, 100]
        boxB = [50, 50, 100, 100] # Identical
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0, places=3)

        boxC = [200, 200, 100, 100] # Completely disjoint
        self.assertEqual(compute_iou(boxA, boxC), 0.0)

        boxD = [100, 50, 100, 100] # Half overlapping horizontally
        iou_d = compute_iou(boxA, boxD)
        self.assertGreater(iou_d, 0.3)
        self.assertLess(iou_d, 0.5)

        proposals = [
            (boxA, 0.95, None),
            ([52, 52, 98, 98], 0.88, None), # Overlapping duplicate of A
            (boxC, 0.90, None)
        ]
        kept = nms_boxes(proposals, iou_thresh=0.35)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0][0], boxA)
        self.assertEqual(kept[1][0], boxC)

    def test_04_detect_multi_object_ewaste(self):
        """Test multi-object image detection on a synthetic composite image."""
        img = np.full((480, 640, 3), 235, dtype=np.uint8)

        # 1. Laptop patch
        img[40:200, 40:260] = [100, 116, 139]
        img[55:170, 55:245] = [20, 20, 20] # dark display

        # 2. Smartphone patch
        img[50:210, 320:400] = [30, 41, 59]
        img[65:195, 330:390] = [246, 130, 59] # blueish screen

        # 3. Green PCB patch
        img[270:430, 50:250] = [40, 150, 40]
        img[300:340, 90:130] = [10, 10, 10]

        # 4. Battery pack
        img[60:190, 460:560] = [20, 180, 230] # yellow/orange pack

        detections = detect_multi_object_ewaste(img, min_conf=0.60)
        self.assertGreaterEqual(len(detections), 2, "Expected at least 2 distinct object detections")
        obj_ids = [d["object_id"] for d in detections]
        self.assertEqual(len(obj_ids), len(set(obj_ids)), "Object IDs must be strictly unique")
        for d in detections:
            self.assertTrue(d["object_id"].startswith("OBJ-"))
            self.assertIn("box", d)
            self.assertEqual(len(d["box"]), 4)
            self.assertGreater(d["confidence"], 0.0)
            self.assertIn("avoided_co2e_kg", d)
            self.assertIn("materials", d)

    def test_05_track_video_multi_objects(self):
        """Test cross-keyframe spatial tracking and IoU de-duplication."""
        # Create 3 synthetic keyframes containing identical objects at nearly identical coordinates
        keyframes = []
        for i in range(3):
            shift = i * 4 # slight camera jitter
            items = [
                {
                    "object_id": f"OBJ-{i}-1",
                    "device_key": "laptop",
                    "label": "Laptop Computer",
                    "box": [50 + shift, 50 + shift, 180, 140],
                    "weight_kg": 2.2,
                    "co2e_impact_kg": 308.0,
                    "avoided_co2e_kg": 65.0,
                    "is_hazardous": False,
                    "reusable": True,
                    "materials": {"gold_mg": 220.0, "silver_mg": 1000.0, "copper_g": 480.0, "aluminum_g": 350.0, "plastics_g": 550.0}
                },
                {
                    "object_id": f"OBJ-{i}-2",
                    "device_key": "smartphone",
                    "label": "Smartphone",
                    "box": [320 + shift, 60 + shift, 70, 130],
                    "weight_kg": 0.18,
                    "co2e_impact_kg": 54.0,
                    "avoided_co2e_kg": 22.0,
                    "is_hazardous": False,
                    "reusable": True,
                    "materials": {"gold_mg": 35.0, "silver_mg": 350.0, "copper_g": 15.0, "aluminum_g": 25.0, "plastics_g": 45.0}
                }
            ]
            keyframes.append({
                "timestamp_sec": float(i * 3.0),
                "timestamp_formatted": f"00:0{i*3}",
                "detected_objects": items
            })

        tracked_inventory = track_video_multi_objects(keyframes)
        # Spatial de-duplication must collapse recurring observations across frames into exactly 2 distinct devices
        self.assertEqual(len(tracked_inventory), 2, "De-duplication should recognize 2 physical devices across 3 keyframes")
        dev_keys = [t["device_key"] for t in tracked_inventory]
        self.assertIn("laptop", dev_keys)
        self.assertIn("smartphone", dev_keys)
        # Check that observation history is tracked
        for t in tracked_inventory:
            self.assertGreaterEqual(t.get("observation_count", 0), 2)
            self.assertIn("frames_present", t)

    def test_06_database_inventory_and_user_corrections(self):
        """Test database storage, keeping original_ai_inventory intact while recording user edits."""
        initial_items = [
            {
                "object_id": "OBJ-001",
                "label": "Laptop Computer",
                "device_key": "laptop",
                "actual_weight_kg": 2.2,
                "weight_kg": 2.2,
                "co2e_impact_kg": 308.0,
                "avoided_co2e_kg": 65.0,
                "is_hazardous": False,
                "reusable": True,
                "materials": {"gold_mg": 220.0, "silver_mg": 1000.0, "copper_g": 480.0, "aluminum_g": 350.0, "plastics_g": 550.0}
            },
            {
                "object_id": "OBJ-002",
                "label": "Smartphone",
                "device_key": "smartphone",
                "actual_weight_kg": 0.18,
                "weight_kg": 0.18,
                "co2e_impact_kg": 54.0,
                "avoided_co2e_kg": 22.0,
                "is_hazardous": False,
                "reusable": True,
                "materials": {"gold_mg": 35.0, "silver_mg": 350.0, "copper_g": 15.0, "aluminum_g": 25.0, "plastics_g": 45.0}
            }
        ]

        analysis_record = {
            "analysis_id": "test-ewaste-multi-001",
            "file_hash": "testhash123456",
            "original_filename": "test_devices.jpg",
            "stored_filename": "test_devices.jpg",
            "file_type": "image",
            "mime_type": "image/jpeg",
            "file_size_bytes": 10240,
            "image_resolution": "640x480",
            "frame_count": 1,
            "duration_seconds": 0.0,
            "overall_condition": "Multi-Object Electronics Detection",
            "confidence_score": 0.92,
            "model_architecture": "EcoSphere Multi-Object Vision AI v2.2",
            "module_linked": "ewaste",
            "detected_objects": initial_items,
            "recommendations": ["Refurbish laptop", "Recycle smartphone"],
            "uncertainty_metrics": {"confidence": 0.92},
            "is_multi_object": 1,
            "total_objects": 2,
            "total_weight_kg": 2.38,
            "total_co2e_impact": 362.0,
            "total_co2e_avoided": 87.0,
            "hazardous_count": 0,
            "original_ai_inventory_json": json.dumps(initial_items),
            "user_corrected_inventory_json": json.dumps(initial_items),
            "inventory_aggregates_json": json.dumps(calculate_inventory_aggregates(initial_items))
        }

        save_vision_analysis(analysis_record)

        # Retrieve and verify initial state
        rec = get_vision_analysis("test-ewaste-multi-001")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["total_objects"], 2)
        self.assertEqual(len(rec["original_ai_inventory"]), 2)

        # Apply User Correction: modify laptop weight to 3.0 kg, and delete smartphone
        updated_items = [
            {
                "object_id": "OBJ-001",
                "label": "Laptop Computer",
                "device_key": "laptop",
                "actual_weight_kg": 3.0,
                "weight_kg": 3.0,
                "co2e_impact_kg": 420.0,
                "avoided_co2e_kg": 88.64,
                "is_hazardous": False,
                "reusable": True,
                "materials": {"gold_mg": 300.0, "silver_mg": 1363.0, "copper_g": 654.0, "aluminum_g": 477.0, "plastics_g": 750.0}
            }
        ]
        res = update_vision_inventory(
            analysis_id="test-ewaste-multi-001",
            updated_items=updated_items,
            edit_notes="User adjusted laptop weight and removed false detection"
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["total_objects"], 1)
        self.assertEqual(res["total_weight_kg"], 3.0)

        # Re-fetch from DB and verify non-destructive audit retention
        rec2 = get_vision_analysis("test-ewaste-multi-001")
        self.assertEqual(len(rec2["original_ai_inventory"]), 2, "Original AI inventory must NOT be overwritten")
        self.assertEqual(len(rec2["user_corrected_inventory"]), 1, "User corrected inventory should contain 1 item")
        self.assertEqual(rec2["total_weight_kg"], 3.0)

    def test_07_analytical_chatbot_all_9_queries(self):
        """Test llm_provider analytical response grounding for all 9 user queries."""
        sample_context = {
            "analysis_id": "vis-test-9q",
            "is_multi_object": True,
            "overall_condition": "4 E-Waste Items Detected",
            "inventory": [
                {
                    "object_id": "OBJ-001",
                    "label": "Laptop Computer",
                    "device_key": "laptop",
                    "actual_weight_kg": 2.2,
                    "co2e_impact_kg": 308.0,
                    "avoided_co2e_kg": 65.0,
                    "is_hazardous": False,
                    "reusable": True,
                    "materials": {"gold_mg": 220.0, "silver_mg": 1000.0, "copper_g": 480.0, "aluminum_g": 350.0, "plastics_g": 550.0}
                },
                {
                    "object_id": "OBJ-002",
                    "label": "Smartphone",
                    "device_key": "smartphone",
                    "actual_weight_kg": 0.18,
                    "co2e_impact_kg": 54.0,
                    "avoided_co2e_kg": 22.0,
                    "is_hazardous": False,
                    "reusable": True,
                    "materials": {"gold_mg": 35.0, "silver_mg": 350.0, "copper_g": 15.0, "aluminum_g": 25.0, "plastics_g": 45.0}
                },
                {
                    "object_id": "OBJ-003",
                    "label": "Lithium-Ion Battery Pack",
                    "device_key": "battery",
                    "actual_weight_kg": 0.60,
                    "co2e_impact_kg": 42.0,
                    "avoided_co2e_kg": 8.5,
                    "is_hazardous": True,
                    "hazard_profile": "Contains cobalt, lithium, flammable organic electrolytes",
                    "reusable": False,
                    "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 90.0, "aluminum_g": 80.0, "plastics_g": 60.0}
                },
                {
                    "object_id": "OBJ-004",
                    "label": "Copper Power / Data Cables",
                    "device_key": "cables",
                    "actual_weight_kg": 0.20,
                    "co2e_impact_kg": 5.0,
                    "avoided_co2e_kg": 2.8,
                    "is_hazardous": False,
                    "reusable": False,
                    "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 120.0, "aluminum_g": 10.0, "plastics_g": 65.0}
                }
            ],
            "inventory_aggregates": calculate_inventory_aggregates([
                {"device_key": "laptop", "actual_weight_kg": 2.2, "co2e_impact_kg": 308.0, "avoided_co2e_kg": 65.0, "is_hazardous": False, "reusable": True, "materials": {"gold_mg": 220.0, "silver_mg": 1000.0, "copper_g": 480.0, "aluminum_g": 350.0, "plastics_g": 550.0}},
                {"device_key": "smartphone", "actual_weight_kg": 0.18, "co2e_impact_kg": 54.0, "avoided_co2e_kg": 22.0, "is_hazardous": False, "reusable": True, "materials": {"gold_mg": 35.0, "silver_mg": 350.0, "copper_g": 15.0, "aluminum_g": 25.0, "plastics_g": 45.0}},
                {"device_key": "battery", "actual_weight_kg": 0.60, "co2e_impact_kg": 42.0, "avoided_co2e_kg": 8.5, "is_hazardous": True, "hazard_profile": "Contains cobalt, lithium, flammable organic electrolytes", "reusable": False, "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 90.0, "aluminum_g": 80.0, "plastics_g": 60.0}},
                {"device_key": "cables", "actual_weight_kg": 0.20, "co2e_impact_kg": 5.0, "avoided_co2e_kg": 2.8, "is_hazardous": False, "reusable": False, "materials": {"gold_mg": 0.0, "silver_mg": 0.0, "copper_g": 120.0, "aluminum_g": 10.0, "plastics_g": 65.0}}
            ])
        }

        # Query 1: What e-waste is in my image?
        ans1 = llm_provider._vision_analytical_response("What e-waste is in my image?", sample_context)
        self.assertIn("Laptop Computer", ans1)
        self.assertIn("Smartphone", ans1)
        self.assertIn("Battery Pack", ans1)

        # Query 2: How many devices were detected?
        ans2 = llm_provider._vision_analytical_response("How many devices were detected?", sample_context)
        self.assertIn("4", ans2)
        self.assertIn("distinct e-waste items", ans2)

        # Query 3: What is the total e-waste weight?
        ans3 = llm_provider._vision_analytical_response("What is the total e-waste weight?", sample_context)
        self.assertIn("3.18 kg", ans3)

        # Query 4: How much CO2e is associated with this?
        ans4 = llm_provider._vision_analytical_response("How much CO2e is associated with this?", sample_context)
        self.assertIn("409.0", ans4)
        self.assertIn("kg CO₂e", ans4)

        # Query 5: How much CO2e could potentially be avoided by recycling?
        ans5 = llm_provider._vision_analytical_response("How much CO2e could potentially be avoided by recycling?", sample_context)
        self.assertIn("98.3", ans5)
        self.assertIn("avoided", ans5.lower())

        # Query 6: Which items should be reused instead of recycled?
        ans6 = llm_provider._vision_analytical_response("Which items should be reused instead of recycled?", sample_context)
        self.assertIn("Laptop", ans6)
        self.assertIn("Smartphone", ans6)
        self.assertIn("refurbish", ans6.lower())

        # Query 7: What happens if I recycle 80%?
        ans7 = llm_provider._vision_analytical_response("What happens if I recycle 80%?", sample_context)
        self.assertIn("80%", ans7)
        self.assertIn("78.6", ans7) # 98.3 * 0.8 = 78.64

        # Query 8: What should I do with the battery?
        ans8 = llm_provider._vision_analytical_response("What should I do with the battery?", sample_context)
        self.assertIn("HAZARDOUS", ans8)
        self.assertIn("cobalt", ans8.lower())
        self.assertIn("CPCB-authorized", ans8)

        # Query 9: Show me the total environmental impact.
        ans9 = llm_provider._vision_analytical_response("Show me the total environmental impact.", sample_context)
        self.assertIn("Total Weight", ans9)
        self.assertIn("Embodied Carbon", ans9)
        self.assertIn("Precious & Industrial Materials Recoverable", ans9)
        self.assertIn("Diversion", ans9)


if __name__ == "__main__":
    unittest.main()
