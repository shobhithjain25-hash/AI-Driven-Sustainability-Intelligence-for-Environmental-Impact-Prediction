#!/usr/bin/env python3
"""
EcoSphere Full-Stack Development Server & REST Ingestion API
Serves static frontend assets and provides backend endpoints for:
- /api/upload: Multipart & raw file ingestion (CSV/XLSX/JSON)
- /api/datasets: Persisted dataset inventory & validation metadata
- /api/datasets/<id>: Detailed dataset inspection & row preview
- /api/benchmark: Real ML/DL model training & benchmarking on ingested data
- /api/logs: Ingestion & audit trail
"""

import sys
import http.server
import socketserver
import os
import json
import time
import urllib.parse
import uuid
from llm_provider import AgenticChatbot

import traceback
import ingestion_engine
import vision_engine

chatbot = AgenticChatbot(ingestion_engine)


PORT = 5500
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class EcoSphereAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        print(f"  🌐 {self.address_string()} → {format % args}")

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Filename, X-Module, X-Sheet')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_json_response(self, data, status_code=200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # ── API: Health Check ──
        if path == '/api/health':
            self.send_json_response({'status': 'healthy', 'port': PORT, 'engine': 'EcoSphere Ingestion Engine v4.0'})
            return

        # ── API: List Persisted Datasets ──
        if path == '/api/datasets':
            try:
                conn = ingestion_engine.get_db()
                c = conn.cursor()
                c.execute('SELECT id, filename, module, target_col, raw_count, clean_count, duplicate_count, outlier_count, missing_count, status, created_at FROM datasets ORDER BY created_at DESC')
                datasets = [dict(row) for row in c.fetchall()]
                conn.close()
                self.send_json_response({'success': True, 'count': len(datasets), 'datasets': datasets})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Detailed Dataset Inspection ──
        if path.startswith('/api/datasets/'):
            parts = path.strip('/').split('/')
            if len(parts) >= 3:
                ds_id = parts[2]
                try:
                    conn = ingestion_engine.get_db()
                    c = conn.cursor()
                    c.execute('SELECT * FROM datasets WHERE id = ?', (ds_id,))
                    ds = c.fetchone()
                    if not ds:
                        conn.close()
                        self.send_json_response({'success': False, 'error': f'Dataset {ds_id} not found'}, 404)
                        return

                    c.execute('SELECT row_idx, raw_json, clean_json, is_outlier, is_imputed FROM dataset_rows WHERE dataset_id = ? ORDER BY row_idx ASC LIMIT 100', (ds_id,))
                    rows = []
                    for r in c.fetchall():
                        rows.append({
                            'row_idx': r['row_idx'],
                            'clean': json.loads(r['clean_json']),
                            'is_outlier': r['is_outlier'],
                            'is_imputed': r['is_imputed']
                        })
                    conn.close()

                    result = dict(ds)
                    result['columns'] = json.loads(result['columns_json'])
                    result['schema_mapping'] = json.loads(result['schema_mapping_json'])
                    result['validation_report'] = json.loads(result['validation_report_json'])
                    result['sample_rows'] = rows
                    self.send_json_response({'success': True, 'dataset': result})
                except Exception as e:
                    self.send_json_response({'success': False, 'error': str(e)}, 500)
                return

        # ── API: Audit Logs ──
        if path == '/api/logs':
            try:
                conn = ingestion_engine.get_db()
                c = conn.cursor()
                limit = int(query.get('limit', [50])[0])
                c.execute('SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?', (limit,))
                logs = []
                for r in c.fetchall():
                    item = dict(r)
                    item['details'] = json.loads(item['details_json']) if item.get('details_json') else {}
                    logs.append(item)
                conn.close()
                self.send_json_response({'success': True, 'count': len(logs), 'logs': logs})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Latest Solar Energy Calculation ──
        if path == '/api/solar/latest':
            try:
                result = ingestion_engine.get_latest_module_calculation('solar-energy')
                if result:
                    self.send_json_response({'success': True, 'data': result})
                else:
                    self.send_json_response({'success': False, 'message': 'No solar energy calculations found yet.'})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Latest Solar Loss Calculation ──
        if path == '/api/solar-loss/latest':
            try:
                result = ingestion_engine.get_latest_module_calculation('solar-loss')
                if result:
                    self.send_json_response({'success': True, 'data': result})
                else:
                    self.send_json_response({'success': False, 'message': 'No solar loss calculations found yet.'})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Latest Calculation for Any Module ──
        if path.startswith('/api/module-latest/'):
            module_key = path.replace('/api/module-latest/', '').strip('/')
            if module_key:
                try:
                    result = ingestion_engine.get_latest_module_calculation(module_key)
                    if result:
                        self.send_json_response({'success': True, 'module': module_key, 'data': result})
                    else:
                        self.send_json_response({'success': False, 'message': f'No calculations found for module {module_key}.'})
                except Exception as e:
                    self.send_json_response({'success': False, 'error': str(e)}, 500)
        # ── API: Multimodal Vision Analyses History ──
        if path == '/api/vision/history':
            try:
                limit = int(query.get('limit', [25])[0])
                analyses = ingestion_engine.list_vision_analyses(limit)
                self.send_json_response({'success': True, 'count': len(analyses), 'analyses': analyses, 'history': analyses})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Multimodal Vision Validation Benchmark Metrics ──
        if path == '/api/vision/metrics':
            try:
                metrics = vision_engine.get_vision_validation_metrics()
                self.send_json_response({'success': True, 'metrics': metrics})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Detailed Vision Analysis by ID ──
        if path.startswith('/api/vision/'):
            analysis_id = path.replace('/api/vision/', '').strip('/')
            if analysis_id and analysis_id not in ('history', 'metrics', 'analyze', 'what-if'):
                try:
                    record = ingestion_engine.get_vision_analysis(analysis_id)
                    if record:
                        self.send_json_response({'success': True, 'analysis': record})
                    else:
                        self.send_json_response({'success': False, 'error': f'Vision analysis {analysis_id} not found.'}, 404)
                except Exception as e:
                    self.send_json_response({'success': False, 'error': str(e)}, 500)
                return

        # Otherwise serve static assets
        super().do_GET()


    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ── API: Multimodal Vision Analysis (Image & Video) ──
        if path == '/api/vision/analyze':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                content_type = self.headers.get('Content-Type', '')
                raw_body = self.rfile.read(content_len)

                filename = 'uploaded_media.jpg'
                file_bytes = b''

                # Case A: Multipart Form-Data
                if 'multipart/form-data' in content_type:
                    boundary = content_type.split('boundary=')[-1].strip().encode('utf-8')
                    parts = raw_body.split(b'--' + boundary)
                    for part in parts:
                        if b'Content-Disposition' in part and b'filename="' in part:
                            disp_header = part.split(b'\r\n\r\n')[0]
                            for line in disp_header.split(b'\r\n'):
                                if b'filename="' in line:
                                    fn_part = line.split(b'filename="')[1].split(b'"')[0]
                                    filename = fn_part.decode('utf-8', errors='replace')
                            file_bytes = part.split(b'\r\n\r\n')[1].rstrip(b'\r\n')
                else:
                    # Case B: Direct Binary Stream Upload
                    file_bytes = raw_body
                    if self.headers.get('X-Filename'):
                        filename = self.headers.get('X-Filename')

                if not file_bytes:
                    self.send_json_response({'success': False, 'error': 'No file content received in payload.'}, 400)
                    return

                # Process through vision engine
                analysis_res = vision_engine.process_multimodal_analysis(file_bytes, filename)
                self.send_json_response({'success': True, 'data': analysis_res, 'analysis': analysis_res}, 200)

            except ValueError as ve:
                self.send_json_response({'success': False, 'error': str(ve), 'status': 'VALIDATION_ERROR'}, 422)
            except Exception as e:
                traceback.print_exc()
                self.send_json_response({'success': False, 'error': f"Vision analysis error: {str(e)}"}, 500)
            return

        # ── API: Multimodal Vision Inventory Update / User Correction ──
        if path == '/api/vision/inventory/update':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}

                analysis_id = data.get('analysis_id')
                updated_items = data.get('items', [])
                notes = data.get('notes', 'User edited inventory')

                if not analysis_id:
                    self.send_json_response({'success': False, 'error': 'Missing analysis_id'}, 400)
                    return

                updated_record = ingestion_engine.update_vision_inventory(analysis_id, updated_items, edit_notes=notes)
                self.send_json_response({
                    'success': True,
                    'analysis_id': analysis_id,
                    'active_inventory': updated_record.get('active_inventory', []),
                    'inventory_aggregates': updated_record.get('inventory_aggregates', {}),
                    'total_objects': updated_record.get('total_objects', len(updated_items)),
                    'total_weight_kg': updated_record.get('total_weight_kg', 0.0),
                    'total_co2e_impact': updated_record.get('total_co2e_impact', 0.0),
                    'total_co2e_avoided': updated_record.get('total_co2e_avoided', 0.0),
                    'hazardous_count': updated_record.get('hazardous_count', 0),
                    'original_ai_inventory': updated_record.get('original_ai_inventory', []),
                    'updated_record': updated_record
                }, 200)
            except Exception as e:
                traceback.print_exc()
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Multimodal Vision What-If Scenario Re-run ──
        if path == '/api/vision/what-if':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}

                analysis_id = data.get('analysis_id')
                scenario_action = data.get('action', 'clean_panels')
                delay_days = int(data.get('delay_days', 30))

                analysis = ingestion_engine.get_vision_analysis(analysis_id) if analysis_id else None
                if not analysis:
                    self.send_json_response({'success': False, 'error': 'Analysis record not found'}, 404)
                    return

                module_str = str(analysis.get('module_linked', '')).lower()
                if 'solar' in module_str:
                    soiling_recovery = 85.0 if scenario_action == 'clean_panels' else 50.0
                    scenario_impact = {
                        "scenario_action": scenario_action,
                        "delay_days": delay_days,
                        "immediate_energy_recovery_pct": soiling_recovery,
                        "loss_with_delay_pct": round(24.6 + (delay_days * 0.15), 1),
                        "avoided_loss_inr_estimate": round(450 * (delay_days / 30.0), 0),
                        "recommendation": "Perform surface washing within 48h to prevent permanent hot-spot cell damage."
                    }
                elif 'ewaste' in module_str or 'e-waste' in module_str or 'electronic' in module_str:
                    agg = analysis.get('inventory_aggregates', {})
                    tot_wt = float(agg.get('total_weight_kg', analysis.get('total_weight_kg', 2.5)))
                    tot_avoided = float(agg.get('total_co2e_avoided', analysis.get('total_co2e_avoided', 25.0)))
                    tot_co2 = float(agg.get('total_co2e_impact', analysis.get('total_co2e_impact', 50.0)))

                    rate_pct = float(data.get('recycling_rate_pct', 80.0))
                    diverted_kg = round(tot_wt * (rate_pct / 100.0), 2)
                    avoided_kg = round(tot_avoided * (rate_pct / 100.0), 2)
                    unmitigated_kg = round(max(0.0, tot_co2 - avoided_kg), 2)

                    scenario_impact = {
                        "scenario_action": f"Certified Circular Recycling ({rate_pct:.0f}% Target)",
                        "delay_days": delay_days,
                        "recycling_rate_pct": rate_pct,
                        "material_recovery_rate": f"{rate_pct:.0f}%",
                        "landfill_diversion_kg": diverted_kg,
                        "avoided_co2e_kg": avoided_kg,
                        "unmitigated_co2e_kg": unmitigated_kg,
                        "recommendation": f"At {rate_pct:.0f}% recycling rate, you divert {diverted_kg} kg from landfills and abate {avoided_kg} kg CO₂e virgin emissions."
                    }
                elif 'food' in module_str:
                    scenario_impact = {
                        "scenario_action": "Decentralized Vermicomposting",
                        "delay_days": delay_days,
                        "methane_abatement_pct": 92.0,
                        "soil_amendment_kg": 1.5,
                        "recommendation": "Compost kitchen scraps aerobically in 1:2 green-to-brown ratio."
                    }
                else:
                    scenario_impact = {
                        "scenario_action": "Immediate Mitigation",
                        "delay_days": delay_days,
                        "avoided_loss_inr_estimate": 300,
                        "recommendation": "Follow verified action guidelines."
                    }

                self.send_json_response({'success': True, 'scenario_impact': scenario_impact, 'simulation': scenario_impact}, 200)
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Chatbot Generative AI Agent ──
        if path == '/api/chat':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body)
                query = data.get('query', '')
                history = data.get('history', [])
                vision_context = data.get('vision_context')
                
                result = chatbot.process_query(query, history, vision_context=vision_context)
                self.send_json_response(result, 200 if result['success'] else 500)
            except Exception as e:
                self.send_json_response({
                    'success': False, 
                    'answer': f"Sorry, an internal API error occurred: {str(e)}", 
                    'request_id': str(uuid.uuid4())
                }, 500)
            return
            
        # ── API: Benchmark ML Models on Persisted Dataset ──
        if path == '/api/benchmark':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}
                dataset_id = data.get('dataset_id')
                if not dataset_id:
                    self.send_json_response({'success': False, 'error': 'Missing required parameter: dataset_id'}, 400)
                    return
                bench_res = ingestion_engine.benchmark_dataset_models(dataset_id)
                self.send_json_response(bench_res, 200 if bench_res.get('success') else 400)
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Solar Energy Prediction (guaranteed response with fallback) ──
        if path == '/api/solar/predict':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}

                city = data.get('city', 'delhi')
                area = float(data.get('area', 30))
                eff = float(data.get('eff', 0.20))
                tilt = float(data.get('tilt', 20))
                orient = data.get('orient', 'south')
                losses = float(data.get('losses', 0.14))
                tariff = float(data.get('tariff', 8))
                month_str = data.get('month', 'October')

                result = ingestion_engine.SolarPhysicsEngine.predict(
                    city, area, eff, tilt, orient, losses, tariff, month_str
                )

                # Persist to database
                calc_id = ingestion_engine.save_module_calculation(
                    module='solar-energy',
                    inputs=result['inputs'],
                    outputs=result['outputs'],
                    model_name=result['model_name'],
                    model_version=result['model_version'],
                    r2=result['r2'],
                    mape=result['mape'],
                    confidence_lower=result['confidence_lower'],
                    confidence_upper=result['confidence_upper'],
                    input_summary=result['input_summary'],
                    validation_status=result['validation_status'],
                    explanation=result['explanation']
                )
                result['calc_id'] = calc_id
                self.send_json_response(result)
            except Exception as e:
                import traceback as _tb
                print(f"[/api/solar/predict ERROR] {e}\n{_tb.format_exc()}")
                # Guaranteed fallback — never return empty
                fallback = ingestion_engine.SolarPhysicsEngine.predict(
                    'delhi', 30, 0.20, 20, 'south', 0.14, 8, 'October'
                )
                fallback['_fallback'] = True
                fallback['_error'] = str(e)
                self.send_json_response(fallback)
            return

        # ── API: Solar Loss Diagnosis (guaranteed response with fallback) ──
        if path == '/api/solar/diagnose-loss':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}

                panel_type = data.get('panel_type', 'monocrystalline')
                age = float(data.get('age', 5))
                capacity_kw = float(data.get('capacity_kw', 5))
                rated_eff = float(data.get('rated_eff', 20))
                dust = float(data.get('dust', 40))
                shade = float(data.get('shade', 15))
                temp = float(data.get('temp', 55))
                inv_age = float(data.get('inv_age', 3))
                psh = float(data.get('psh', 5.5))
                tariff = float(data.get('tariff', 8))

                result = ingestion_engine.SolarLossDiagnosticEngine.diagnose(
                    panel_type, age, capacity_kw, rated_eff, dust, shade, temp, inv_age, psh, tariff
                )

                # Persist to database
                calc_id = ingestion_engine.save_module_calculation(
                    module='solar-loss',
                    inputs=result['inputs'],
                    outputs=result['outputs'],
                    model_name=result['model_name'],
                    model_version=result['model_version'],
                    r2=result['r2'],
                    mape=result['mape'],
                    confidence_lower=result['confidence_lower'],
                    confidence_upper=result['confidence_upper'],
                    input_summary=result['input_summary'],
                    validation_status=result['validation_status'],
                    explanation=result['explanation'],
                    diagnosis=result['diagnosis'],
                    recommendations=result['recommendations']
                )
                result['calc_id'] = calc_id
                self.send_json_response(result)
            except Exception as e:
                import traceback as _tb
                print(f"[/api/solar/diagnose-loss ERROR] {e}\n{_tb.format_exc()}")
                # Guaranteed fallback
                fallback = ingestion_engine.SolarLossDiagnosticEngine.diagnose(
                    'monocrystalline', 5, 5, 20, 40, 15, 55, 3, 5.5, 8
                )
                fallback['_fallback'] = True
                fallback['_error'] = str(e)
                self.send_json_response(fallback)
            return

        # ── API: Generic Module Calculation Save (from frontend) ──
        if path == '/api/module-save':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}

                module = data.get('module', '')
                if not module:
                    self.send_json_response({'success': False, 'error': 'Missing module field'}, 400)
                    return

                calc_id = ingestion_engine.save_module_calculation(
                    module=module,
                    inputs=data.get('inputs', {}),
                    outputs=data.get('outputs', {}),
                    model_name=data.get('model_name', ''),
                    model_version=data.get('model_version', 'v1.0'),
                    r2=float(data.get('r2', 0)),
                    mape=float(data.get('mape', 0)),
                    confidence_lower=float(data.get('confidence_lower', 0)),
                    confidence_upper=float(data.get('confidence_upper', 0)),
                    input_summary=data.get('input_summary', ''),
                    validation_status=data.get('validation_status', 'VALID'),
                    explanation=data.get('explanation', ''),
                    diagnosis=data.get('diagnosis'),
                    recommendations=data.get('recommendations')
                )
                self.send_json_response({'success': True, 'calc_id': calc_id, 'module': module, 'timestamp': time.time()})
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return


        # ── API: Ingest Raw Text / JSON ──
        if path == '/api/ingest-raw':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len).decode('utf-8')
                data = json.loads(body) if body else {}
                records = data.get('records', [])
                headers = data.get('headers', list(records[0].keys()) if records else [])
                module_key = data.get('module') or 'carbon'
                filename = data.get('filename') or 'raw_input.json'

                val_res = ingestion_engine.harmonize_and_validate(records, headers, module_key, data.get('mapping'))
                if not val_res.get('success'):
                    self.send_json_response(val_res, 422)
                    return

                ds_id = ingestion_engine.persist_ingested_dataset(
                    filename, module_key, records, val_res['cleaned_records'],
                    val_res['mapping'], val_res['target'], val_res['validation_report']
                )
                val_res['dataset_id'] = ds_id
                self.send_json_response(val_res, 200)
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)}, 500)
            return

        # ── API: Multipart or Direct File Upload ──
        if path == '/api/upload':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                content_type = self.headers.get('Content-Type', '')
                raw_body = self.rfile.read(content_len)

                filename = 'uploaded_file.csv'
                file_bytes = b''
                module_key = self.headers.get('X-Module')
                target_sheet = self.headers.get('X-Sheet')

                # Case A: Multipart Form-Data
                if 'multipart/form-data' in content_type:
                    boundary = content_type.split('boundary=')[-1].strip().encode('utf-8')
                    parts = raw_body.split(b'--' + boundary)
                    for part in parts:
                        if b'Content-Disposition' in part and b'filename="' in part:
                            # Extract filename
                            disp_header = part.split(b'\r\n\r\n')[0]
                            for line in disp_header.split(b'\r\n'):
                                if b'filename="' in line:
                                    fn_part = line.split(b'filename="')[1].split(b'"')[0]
                                    filename = fn_part.decode('utf-8', errors='replace')
                            file_bytes = part.split(b'\r\n\r\n')[1].rstrip(b'\r\n')
                        elif b'name="module"' in part:
                            module_key = part.split(b'\r\n\r\n')[1].rstrip(b'\r\n').decode('utf-8').strip()
                        elif b'name="sheet"' in part:
                            target_sheet = part.split(b'\r\n\r\n')[1].rstrip(b'\r\n').decode('utf-8').strip()
                else:
                    # Case B: Direct Binary / Stream Upload
                    file_bytes = raw_body
                    if self.headers.get('X-Filename'):
                        filename = self.headers.get('X-Filename')

                if not file_bytes:
                    self.send_json_response({'success': False, 'error': 'No file content received in payload.'}, 400)
                    return

                # Save raw copy in data/uploads/
                save_path = os.path.join(ingestion_engine.UPLOAD_DIR, f"{int(time.time())}_{filename}")
                with open(save_path, 'wb') as f:
                    f.write(file_bytes)

                # Parse file according to extension
                is_xlsx = filename.lower().endswith('.xlsx') or file_bytes[:4] == b'PK\x03\x04'
                parsed_sheets = []
                selected_sheet = None

                if is_xlsx:
                    xlsx_res = ingestion_engine.parse_xlsx_content(file_bytes, target_sheet)
                    if xlsx_res.get('error'):
                        self.send_json_response({'success': False, 'error': xlsx_res['error']}, 400)
                        return
                    headers = xlsx_res['headers']
                    records = xlsx_res['records']
                    parsed_sheets = xlsx_res.get('sheets', [])
                    selected_sheet = xlsx_res.get('selected_sheet')
                    encoding = 'binary_xlsx'
                    delimiter = 'xlsx_cells'
                else:
                    encoding, text = ingestion_engine.detect_encoding(file_bytes)
                    csv_res = ingestion_engine.parse_csv_content(text)
                    if csv_res.get('error'):
                        self.send_json_response({'success': False, 'error': csv_res['error']}, 400)
                        return
                    headers = csv_res['headers']
                    records = csv_res['records']
                    delimiter = csv_res['delimiter']

                # Auto-detect Module if not explicitly specified or if set to 'auto'
                auto_detect_info = ingestion_engine.auto_detect_module(headers)
                if not module_key or module_key == 'auto':
                    module_key = auto_detect_info['detected_module'] or 'carbon'

                # Harmonize, Validate, Deduplicate, Impute, and Winsorize
                val_res = ingestion_engine.harmonize_and_validate(records, headers, module_key)
                if not val_res.get('success'):
                    # Incompatible schema with helpful repair plan
                    val_res['filename'] = filename
                    val_res['encoding'] = encoding
                    val_res['delimiter'] = delimiter
                    val_res['detected_module'] = auto_detect_info['detected_module']
                    self.send_json_response(val_res, 422)
                    return

                # Persist Clean Dataset & Rows in SQLite Database
                dataset_id = ingestion_engine.persist_ingested_dataset(
                    filename, module_key, records, val_res['cleaned_records'],
                    val_res['mapping'], val_res['target'], val_res['validation_report']
                )

                # Prepare Clean Response with Real Uploaded Rows
                response_data = {
                    'success': True,
                    'status': 'INGESTED_PERSISTED',
                    'dataset_id': dataset_id,
                    'filename': filename,
                    'module': module_key,
                    'encoding': encoding,
                    'delimiter': delimiter,
                    'sheets': parsed_sheets,
                    'selected_sheet': selected_sheet,
                    'target': val_res['target'],
                    'mapping': val_res['mapping'],
                    'auto_detect_info': auto_detect_info,
                    'validation_report': val_res['validation_report'],
                    'sample_rows': val_res['cleaned_records'][:15]
                }
                self.send_json_response(response_data, 200)

            except Exception as e:
                traceback.print_exc()
                self.send_json_response({'success': False, 'error': f"Upload processing error: {str(e)}"}, 500)
            return

        self.send_json_response({'error': 'Not Found'}, 404)

def run(port=PORT, auto_open=False):
    import sys
    import threading
    import webbrowser

    print("\n" + "="*55)
    print("  🌿 EcoSphere REST Ingestion Server & Intelligence API")
    print("="*55)
    print(f"  📁 Serving Directory: {DIRECTORY}")
    print(f"  🔗 Listening on:     http://localhost:{port}")
    print(f"  🗄️ Database Path:    {ingestion_engine.DB_PATH}")
    if auto_open:
        print(f"  🚀 Auto-opening:     http://localhost:{port}/index.html")
    print("="*55 + "\n")

    socketserver.TCPServer.allow_reuse_address = True
    active_port = port

    try:
        httpd = socketserver.TCPServer(("", active_port), EcoSphereAPIHandler)
    except OSError as e:
        if e.errno == 48:
            print(f"⚠️  Port {port} is currently in use. Attempting to free port {port}...")
            os.system(f"lsof -ti :{port} | xargs kill -9 2>/dev/null")
            time.sleep(0.5)
            try:
                httpd = socketserver.TCPServer(("", port), EcoSphereAPIHandler)
                active_port = port
                print(f"  🔗 Successfully freed and bound to port {port}!\n")
            except Exception:
                active_port = port + 1
                httpd = socketserver.TCPServer(("", active_port), EcoSphereAPIHandler)
                print(f"  🔗 Server bound to backup port: http://localhost:{active_port}\n")
        else:
            raise e

    if auto_open:
        def _launch_browser():
            time.sleep(0.6)
            try:
                webbrowser.open(f"http://localhost:{active_port}/index.html")
            except Exception:
                pass
        threading.Thread(target=_launch_browser, daemon=True).start()

    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  🛑 Server gracefully stopped. Goodbye!")

if __name__ == '__main__':
    auto_open = '--no-browser' not in sys.argv and os.environ.get('NO_BROWSER') != '1'
    run(auto_open=auto_open)

