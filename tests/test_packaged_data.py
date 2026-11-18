from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import hashlib
import json
import unittest

import pandas as pd
import numpy as np

from epidemic_rl.engine import ROOT


class PackagedDataTests(unittest.TestCase):
    def test_packaged_inputs_match_provenance_hashes(self):
        hashes=json.loads((ROOT/'provenance/data_sha256.json').read_text())
        for name,expected in hashes.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)

    def test_dates_channels_and_country_separation(self):
        e=pd.read_csv(ROOT/'data/england/pf_observations.csv')
        b=pd.read_csv(ROOT/'data/belgium/pf_observations.csv')
        self.assertEqual(e.date.min(),'2020-10-12');self.assertEqual(e.date.max(),'2021-01-31')
        self.assertEqual(b.date.min(),'2020-09-01');self.assertEqual(b.date.max(),'2021-03-12')
        self.assertIn('mv_beds_all_ages',e);self.assertNotIn('icu_occupied_total',e)
        self.assertIn('cases_0_39',b);self.assertNotIn('cases_0_44',b)
        self.assertFalse(e.isna().any().any());self.assertFalse(b.isna().any().any())

    def test_excel_workbooks_contain_real_source_values(self):
        ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        for country,columns in [('england',['date','cases_0_44','cases_45_64','cases_65_plus']),
                                ('belgium',['date','cases_0_39','cases_40_59','cases_60_plus'])]:
            with self.subTest(country=country):
                expected=pd.read_csv(ROOT/f'data/{country}/pf_observations.csv')[columns]
                with ZipFile(ROOT/f'data/{country}/real_observations.xlsx') as z:
                    self.assertIn('[Content_Types].xml',z.namelist())
                    sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
                    rows=[]
                    for row in sheet.find('s:sheetData',ns):
                        vals=[]
                        for cell in row:
                            if cell.attrib.get('t')=='inlineStr':vals.append(cell.find('s:is/s:t',ns).text)
                            else:vals.append(float(cell.find('s:v',ns).text))
                        rows.append(vals)
                self.assertEqual(rows[0],columns)
                self.assertEqual([v[0] for v in rows[1:]],expected.date.tolist())
                np.testing.assert_array_equal(np.array(rows[1:],dtype=object)[:,1:].astype(float),expected.iloc[:,1:].to_numpy())

    def test_required_calendars_are_complete_and_no_france_payload(self):
        for country in ['england','belgium']:
            a=pd.read_csv(ROOT/f'data/{country}/government_actions.csv')
            self.assertEqual(len(a),152)
            self.assertEqual(a.date.tolist(),pd.date_range('2020-10-12','2021-03-12').strftime('%Y-%m-%d').tolist())
            self.assertTrue(((a.iloc[:,1:]>0)&(a.iloc[:,1:]<=1)).all().all())
        self.assertFalse(list((ROOT/'data').rglob('*france*')))


if __name__=='__main__':unittest.main()
