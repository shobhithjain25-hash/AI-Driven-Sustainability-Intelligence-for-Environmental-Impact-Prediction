#!/usr/bin/env python3
"""
Generate comprehensive test fixtures for EcoSphere Ingestion Pipeline
"""

import os
import io
import zipfile
import xml.etree.ElementTree as ET

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_fixtures')
os.makedirs(FIXTURE_DIR, exist_ok=True)

# 1. Carbon Standard (Comma-separated)
carbon_standard = """carKm,busKm,elec,hh,flights,lpg,diet,shopping,total_co2
25,12,240,3,1,1,moderate,moderate,385.2
40,5,320,4,2,2,non-veg,frequent,490.5
10,25,180,2,0,1,vegetarian,minimal,265.8
0,30,150,1,0,1,vegan,minimal,195.4
55,0,410,5,3,2,non-veg,frequent,580.1
30,15,220,3,1,1,moderate,moderate,355.0
15,20,190,2,0,1,vegetarian,moderate,290.3
35,10,280,4,1,2,moderate,moderate,410.7
"""
with open(os.path.join(FIXTURE_DIR, 'carbon_standard.csv'), 'w', encoding='utf-8') as f:
    f.write(carbon_standard.strip())

# 2. Carbon Semicolon (Semicolon-delimited with synonyms and units)
carbon_semicolon = """daily_commute_km;public_transport_km;electricity_consumption_kwh;family_members;air_travel;co2_monthly
30;10;250;3;1;390.2
15;20;180;2;0;270.5
45;5;350;4;2;510.8
5;35;140;1;0;185.0
60;0;420;5;3;600.4
25;15;210;3;1;345.1
20;18;200;2;0;305.6
35;8;290;4;1;430.2
10;22;160;2;0;240.1
50;2;380;4;2;540.8
18;16;195;3;0;295.4
40;6;310;3;2;470.5
8;28;150;1;0;205.2
28;12;230;3;1;360.7
52;0;400;5;3;575.0
22;14;205;2;1;320.4
38;7;300;4;1;445.6
12;24;175;2;0;260.8
48;4;365;4;2;530.1
6;32;145;1;0;195.8
32;10;260;3;1;400.5
24;16;215;3;1;350.2
42;5;330;4;2;495.0
14;20;185;2;0;280.4
58;0;415;5;3;590.2
"""
with open(os.path.join(FIXTURE_DIR, 'carbon_semicolon.csv'), 'w', encoding='utf-8') as f:
    f.write(carbon_semicolon.strip())

# 3. Solar Energy TSV (Tab-separated)
solar_tsv = """surface_area\tpanel_efficiency\tsunlight_hours\tgrid_tariff\tsystem_tilt\tannual_yield_kwh
25\t0.19\t5.4\t8.5\t20\t4580
40\t0.20\t5.2\t8.0\t22\t7250
15\t0.18\t5.6\t9.0\t18\t2890
50\t0.21\t5.1\t7.5\t25\t9100
30\t0.19\t5.5\t8.5\t20\t5620
20\t0.18\t5.3\t8.0\t20\t3650
35\t0.20\t5.4\t8.5\t22\t6480
"""
with open(os.path.join(FIXTURE_DIR, 'solar_energy_tabs.tsv'), 'w', encoding='utf-8') as f:
    f.write(solar_tsv.strip())

# 4. Solar Loss Latin-1 (Encoded with Latin-1 and degree symbols)
solar_loss_latin1 = "panel_age,soiling_index,shading_pct,ambient_temp_\xb0C,inverter_years,total_loss_pct\n" + """3,15,5,32,3,12.5
7,35,15,38,7,28.4
12,50,25,42,12,44.2
1,5,0,28,1,5.1
15,65,30,45,15,56.8
5,25,10,35,5,20.3
9,40,20,40,9,35.7
"""
with open(os.path.join(FIXTURE_DIR, 'solar_loss_latin1.csv'), 'wb') as f:
    f.write(solar_loss_latin1.strip().encode('latin-1'))

# 5. Air Quality Pipe (| Delimited)
air_pipe = """station_city|fine_particulate_pm25|coarse_particulate_pm10|nitrogen_dioxide|sulfur_dioxide|carbon_monoxide|cpcb_aqi
Delhi|145.2|260.5|48.2|18.5|2.4|315
Mumbai|65.4|120.8|28.4|12.1|1.1|142
Bangalore|35.1|70.2|19.5|8.4|0.8|88
Kolkata|110.8|210.4|42.1|15.2|1.9|260
Chennai|48.6|95.1|22.3|9.8|0.9|112
Hyderabad|58.2|115.6|26.1|11.4|1.0|128
Pune|52.0|102.3|24.5|10.2|0.9|118
"""
with open(os.path.join(FIXTURE_DIR, 'air_quality_pipe.txt'), 'w', encoding='utf-8') as f:
    f.write(air_pipe.strip())

# 6. E-Waste Multi-Sheet XLSX
def create_multisheet_xlsx(filepath):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>'''
        z.writestr('[Content_Types].xml', content_types)
        
        rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
        z.writestr('_rels/.rels', rels)
        
        wb_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>'''
        z.writestr('xl/_rels/workbook.xml.rels', wb_rels)
        
        wb = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="EWasteInventory" sheetId="1" r:id="rId1"/>
    <sheet name="CollectionCenters" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>'''
        z.writestr('xl/workbook.xml', wb)
        
        strings = []
        string_map = {}
        def get_str_id(s):
            if s not in string_map:
                string_map[s] = len(strings)
                strings.append(s)
            return string_map[s]
            
        sheet1_rows = [
            ['item_count', 'device_type', 'working_status', 'unit_weight', 'total_kg'],
            [4, 'smartphone', 'broken', 0.18, 0.72],
            [2, 'laptop', 'partially', 2.2, 4.4],
            [1, 'desktop', 'broken', 8.5, 8.5],
            [5, 'tablet', 'working', 0.5, 2.5],
            [3, 'printer', 'broken', 7.0, 21.0],
            [10, 'battery', 'broken', 0.6, 6.0],
            [1, 'washing-machine', 'mixed', 70.0, 70.0]
        ]
        
        sheet2_rows = [
            ['center_name', 'city', 'pincode'],
            ['E-Parisaraa', 'Bangalore', 560099],
            ['Attero Recycling', 'Noida', 201301]
        ]
        
        def build_sheet_xml(rows_data):
            sheet_xml_rows = []
            for r_idx, row in enumerate(rows_data, start=1):
                cell_strs = []
                for c_idx, val in enumerate(row, start=1):
                    col_letter = chr(64 + c_idx)
                    cell_ref = f'{col_letter}{r_idx}'
                    if isinstance(val, (int, float)):
                        cell_strs.append(f'<c r="{cell_ref}"><v>{val}</v></c>')
                    else:
                        s_id = get_str_id(str(val))
                        cell_strs.append(f'<c r="{cell_ref}" t="s"><v>{s_id}</v></c>')
                sheet_xml_rows.append(f'<row r="{r_idx}">{" ".join(cell_strs)}</row>')
            return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {''.join(sheet_xml_rows)}
  </sheetData>
</worksheet>'''

        s1_xml = build_sheet_xml(sheet1_rows)
        s2_xml = build_sheet_xml(sheet2_rows)
        
        sst_items = ''.join([f'<si><t>{s}</t></si>' for s in strings])
        sst = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(strings)}" uniqueCount="{len(strings)}">
{sst_items}
</sst>'''
        z.writestr('xl/sharedStrings.xml', sst)
        z.writestr('xl/worksheets/sheet1.xml', s1_xml)
        z.writestr('xl/worksheets/sheet2.xml', s2_xml)
        
    with open(filepath, 'wb') as f:
        f.write(buf.getvalue())

create_multisheet_xlsx(os.path.join(FIXTURE_DIR, 'ewaste_multisheet.xlsx'))

# 7. Food Waste Dirty (With duplicates, nulls, and extreme outliers)
food_dirty = """weekly_spend,family_size,diet_type,meal_planning,leftover_use,food_waste_kg
2500,3,vegetarian,often,sometimes,2.8
4000,4,non-veg,rarely,rarely,5.4
2500,3,vegetarian,often,sometimes,2.8
1800,2,vegan,always,always,1.2
,3,vegetarian,sometimes,sometimes,3.0
5000000,4,non-veg,never,never,85.0
3200,3,moderate,often,often,2.5
2800,3,vegetarian,sometimes,sometimes,
2500,3,vegetarian,often,sometimes,2.8
1500,1,moderate,always,always,0.9
"""
with open(os.path.join(FIXTURE_DIR, 'food_waste_dirty.csv'), 'w', encoding='utf-8') as f:
    f.write(food_dirty.strip())

# 8. Incompatible Schema (Irrelevant business survey columns)
incompatible = """survey_id,respondent_name,satisfaction_score,nps_rating,comments
101,John Doe,4,8,Great customer service
102,Jane Smith,5,10,Very satisfied with portal speed
103,Bob Taylor,2,4,Needs better design
104,Alice Brown,5,9,Excellent support team
"""
with open(os.path.join(FIXTURE_DIR, 'incompatible_schema.csv'), 'w', encoding='utf-8') as f:
    f.write(incompatible.strip())

print(f"✅ Successfully created 8 test fixtures in {FIXTURE_DIR}")
