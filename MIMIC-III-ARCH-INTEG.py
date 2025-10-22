#!/usr/bin/env python3
"""
MIMIC-III Integration Architecture
Complete pipeline for pattern discovery and clinical validation
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: MIMIC-III WAVEFORM DATA LOADER (Pattern Discovery)
# ============================================================================

@dataclass
class WaveformRecord:
    """Container for waveform data from MIMIC-III Matched Subset"""
    subject_id: str
    record_name: str
    ecg_signal: np.ndarray
    ppg_signal: Optional[np.ndarray]
    abp_signal: Optional[np.ndarray]
    resp_signal: Optional[np.ndarray]
    sampling_rate: int
    duration_seconds: float
    signal_quality: float
    timestamp: datetime
    
    def has_ecg_ppg_pair(self) -> bool:
        """Check if record contains both ECG and PPG signals"""
        return self.ecg_signal is not None and self.ppg_signal is not None


class MIMICWaveformLoader:
    """
    Loads ECG+PPG signal pairs from MIMIC-III Waveform Matched Subset
    
    Directory structure:
    mimic3wdb-matched/1.0/
    ├── p00/
    │   ├── p000020/
    │   │   ├── p000020-2183-04-28-17-47.hea
    │   │   ├── p000020-2183-04-28-17-47n.hea
    │   │   └── segment files...
    """
    
    def __init__(self, waveform_root: Path):
        self.waveform_root = Path(waveform_root)
        self.valid_records = []
        
        logger.info(f"Initializing MIMIC Waveform Loader from {waveform_root}")
        
    def scan_available_records(self) -> List[str]:
        """Scan directory structure for available waveform records"""
        records = []
        
        # Iterate through p00-p09 directories
        for pdir in self.waveform_root.glob("p*/"):
            # Iterate through patient subdirectories
            for patient_dir in pdir.glob("p*/"):
                # Find master header files (not numerics, not segments)
                for hea_file in patient_dir.glob("*.hea"):
                    if not hea_file.stem.endswith('n') and not '_' in hea_file.stem:
                        records.append(str(hea_file.with_suffix('')))
        
        logger.info(f"Found {len(records)} waveform records")
        return records
    
    def load_record(self, record_path: str) -> Optional[WaveformRecord]:
        """
        Load a single waveform record with ECG and PPG signals
        
        Args:
            record_path: Path to record without extension (e.g., 'p00/p000020/p000020-2183-04-28-17-47')
        
        Returns:
            WaveformRecord if successful, None if signals missing or corrupted
        """
        try:
            # Read WFDB record
            signals, fields = wfdb.rdsamp(record_path)
            
            # Extract metadata
            subject_id = Path(record_path).parent.name  # e.g., 'p000020'
            record_name = Path(record_path).name
            
            # Parse timestamp from record name (format: pXXXXXX-YYYY-MM-DD-HH-MM)
            timestamp_str = '-'.join(record_name.split('-')[1:])
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M')
            
            # Identify signal channels
            sig_names = fields['sig_name']
            ecg_idx = self._find_signal_index(sig_names, ['II', 'ECG', 'V', 'MLI', 'MLII'])
            ppg_idx = self._find_signal_index(sig_names, ['PLETH', 'PPG'])
            abp_idx = self._find_signal_index(sig_names, ['ABP', 'ART'])
            resp_idx = self._find_signal_index(sig_names, ['RESP'])
            
            # Extract signals
            ecg_signal = signals[:, ecg_idx] if ecg_idx is not None else None
            ppg_signal = signals[:, ppg_idx] if ppg_idx is not None else None
            abp_signal = signals[:, abp_idx] if abp_idx is not None else None
            resp_signal = signals[:, resp_idx] if resp_idx is not None else None
            
            # Must have at least ECG
            if ecg_signal is None:
                logger.warning(f"No ECG signal found in {record_name}")
                return None
            
            # Calculate signal quality (basic check)
            quality = self._assess_signal_quality(ecg_signal, ppg_signal)
            
            # Create record
            record = WaveformRecord(
                subject_id=subject_id,
                record_name=record_name,
                ecg_signal=ecg_signal,
                ppg_signal=ppg_signal,
                abp_signal=abp_signal,
                resp_signal=resp_signal,
                sampling_rate=fields['fs'],
                duration_seconds=len(ecg_signal) / fields['fs'],
                signal_quality=quality,
                timestamp=timestamp
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error loading {record_path}: {str(e)}")
            return None
    
    def _find_signal_index(self, sig_names: List[str], possible_names: List[str]) -> Optional[int]:
        """Find index of signal by matching possible names"""
        for i, name in enumerate(sig_names):
            if any(pname.lower() in name.lower() for pname in possible_names):
                return i
        return None
    
    def _assess_signal_quality(self, ecg: np.ndarray, ppg: Optional[np.ndarray]) -> float:
        """Basic signal quality assessment (0.0-1.0)"""
        quality = 0.0
        
        # ECG quality checks
        if ecg is not None:
            # Check for flat line
            if np.std(ecg) > 0.1:
                quality += 0.3
            # Check for reasonable amplitude range
            if 0.1 < np.ptp(ecg) < 10.0:
                quality += 0.2
            # Check for missing values
            if not np.any(np.isnan(ecg)):
                quality += 0.2
        
        # PPG quality checks (if available)
        if ppg is not None:
            if np.std(ppg) > 0.05:
                quality += 0.15
            if not np.any(np.isnan(ppg)):
                quality += 0.15
        
        return min(quality, 1.0)
    
    def load_patient_records(self, subject_id: str) -> List[WaveformRecord]:
        """Load all waveform records for a specific patient"""
        records = []
        patient_dir = None
        
        # Find patient directory (format: p00/p000020/)
        prefix = subject_id[:3]  # e.g., 'p00'
        patient_dir = self.waveform_root / prefix / subject_id
        
        if not patient_dir.exists():
            logger.warning(f"Patient directory not found: {patient_dir}")
            return records
        
        # Load all records for this patient
        for hea_file in patient_dir.glob("*.hea"):
            if not hea_file.stem.endswith('n') and not '_' in hea_file.stem:
                record_path = str(hea_file.with_suffix(''))
                record = self.load_record(record_path)
                if record:
                    records.append(record)
        
        logger.info(f"Loaded {len(records)} records for patient {subject_id}")
        return records


# ============================================================================
# PART 2: MIMIC-III CLINICAL DATA LOADER (Validation & Risk Prediction)
# ============================================================================

@dataclass
class ClinicalData:
    """Container for clinical data from MIMIC-III Clinical Database"""
    subject_id: str
    hadm_id: str  # Hospital admission ID
    icustay_id: str  # ICU stay ID
    
    # Demographics
    gender: str
    age: int
    ethnicity: str
    
    # Admission details
    admittime: datetime
    dischtime: datetime
    icu_los_days: float
    hospital_los_days: float
    
    # Clinical outcomes
    hospital_expire_flag: bool
    discharge_location: str
    
    # Diagnoses (ICD-9 codes)
    diagnoses: List[Dict[str, str]]  # [{'icd9_code': '410.71', 'description': 'AMI'}, ...]
    
    # Procedures
    procedures: List[Dict[str, str]]
    
    # Lab results (relevant for stroke risk)
    lab_results: Dict[str, float]  # {'glucose': 120, 'creatinine': 1.1, ...}
    
    # Medications
    medications: List[str]
    
    # Stroke risk factors (derived)
    has_atrial_fib: bool
    has_diabetes: bool
    has_hypertension: bool
    has_chf: bool  # Congestive heart failure
    
    # CHA2DS2-VASc score components (for stroke risk)
    cha2ds2_vasc_score: int


class MIMICClinicalLoader:
    """
    Loads clinical data from MIMIC-III Clinical Database
    
    HIPAA Compliance Notes:
    - All data is already de-identified per HIPAA Safe Harbor
    - Dates are shifted randomly (preserve intervals)
    - Ages >89 are shifted to >300 years
    - No PHI present in this dataset
    """
    
    def __init__(self, clinical_root: Path):
        self.clinical_root = Path(clinical_root)
        
        # Load core tables into memory
        logger.info("Loading MIMIC-III Clinical Database tables...")
        self.patients = self._load_table('PATIENTS.csv.gz')
        self.admissions = self._load_table('ADMISSIONS.csv.gz')
        self.icustays = self._load_table('ICUSTAYS.csv.gz')
        self.diagnoses = self._load_table('DIAGNOSES_ICD.csv.gz')
        self.procedures = self._load_table('PROCEDURES_ICD.csv.gz')
        self.labevents = self._load_table('LABEVENTS.csv.gz')
        self.prescriptions = self._load_table('PRESCRIPTIONS.csv.gz')
        
        # Load dictionary tables
        self.d_icd_diagnoses = self._load_table('D_ICD_DIAGNOSES.csv.gz')
        self.d_icd_procedures = self._load_table('D_ICD_PROCEDURES.csv.gz')
        self.d_labitems = self._load_table('D_LABITEMS.csv.gz')
        
        logger.info("Clinical database loaded successfully")
    
    def _load_table(self, filename: str) -> pd.DataFrame:
        """Load a CSV table from the clinical database"""
        path = self.clinical_root / filename
        if not path.exists():
            logger.warning(f"Table not found: {filename}")
            return pd.DataFrame()
        
        df = pd.read_csv(path, compression='gzip' if filename.endswith('.gz') else None)
        logger.info(f"Loaded {filename}: {len(df)} rows")
        return df
    
    def get_patient_clinical_data(self, subject_id: str) -> Optional[ClinicalData]:
        """
        Load comprehensive clinical data for a patient
        
        This includes all admissions, but we typically want the ICU stay
        that corresponds to the waveform recording timestamps.
        """
        # Convert subject_id (e.g., 'p000020') to integer
        subject_id_int = int(subject_id.replace('p', ''))
        
        # Get patient demographics
        patient_row = self.patients[self.patients['SUBJECT_ID'] == subject_id_int]
        if patient_row.empty:
            logger.warning(f"No clinical data for subject {subject_id}")
            return None
        
        patient_row = patient_row.iloc[0]
        
        # Calculate age (handle >89 year old patients)
        dob = pd.to_datetime(patient_row['DOB'])
        # Use first admission to calculate age
        first_adm = self.admissions[self.admissions['SUBJECT_ID'] == subject_id_int].iloc[0]
        admit_time = pd.to_datetime(first_adm['ADMITTIME'])
        age = (admit_time - dob).days // 365
        if age > 200:  # De-identified >89 year olds
            age = 90  # Use 90 as proxy
        
        # Get ICU stay information
        icu_stays = self.icustays[self.icustays['SUBJECT_ID'] == subject_id_int]
        if icu_stays.empty:
            logger.warning(f"No ICU stay data for subject {subject_id}")
            return None
        
        icu_stay = icu_stays.iloc[0]  # Use first ICU stay
        icustay_id = icu_stay['ICUSTAY_ID']
        hadm_id = icu_stay['HADM_ID']
        
        # Get admission details
        admission = self.admissions[self.admissions['HADM_ID'] == hadm_id].iloc[0]
        
        # Get diagnoses for this admission
        diagnoses_list = self._get_diagnoses(hadm_id)
        
        # Get procedures
        procedures_list = self._get_procedures(hadm_id)
        
        # Get relevant lab results
        lab_results = self._get_lab_results(hadm_id)
        
        # Get medications
        medications = self._get_medications(hadm_id)
        
        # Derive stroke risk factors
        has_afib = self._check_diagnosis_code(diagnoses_list, ['427.31'])  # Atrial fib
        has_diabetes = self._check_diagnosis_code(diagnoses_list, ['250.'])  # Diabetes (all subtypes)
        has_htn = self._check_diagnosis_code(diagnoses_list, ['401.', '402.', '403.'])  # Hypertension
        has_chf = self._check_diagnosis_code(diagnoses_list, ['428.'])  # CHF
        
        # Calculate CHA2DS2-VASc score
        cha2ds2_vasc = self._calculate_cha2ds2_vasc(
            age=age,
            gender=patient_row['GENDER'],
            has_chf=has_chf,
            has_htn=has_htn,
            has_diabetes=has_diabetes,
            has_stroke_tia=self._check_diagnosis_code(diagnoses_list, ['433.', '434.', '435.']),
            has_vascular_disease=self._check_diagnosis_code(diagnoses_list, ['410.', '411.', '412.'])
        )
        
        # Create clinical data object
        clinical_data = ClinicalData(
            subject_id=subject_id,
            hadm_id=str(hadm_id),
            icustay_id=str(icustay_id),
            gender=patient_row['GENDER'],
            age=age,
            ethnicity=admission['ETHNICITY'],
            admittime=pd.to_datetime(admission['ADMITTIME']),
            dischtime=pd.to_datetime(admission['DISCHTIME']),
            icu_los_days=(pd.to_datetime(icu_stay['OUTTIME']) - pd.to_datetime(icu_stay['INTIME'])).days,
            hospital_los_days=(pd.to_datetime(admission['DISCHTIME']) - pd.to_datetime(admission['ADMITTIME'])).days,
            hospital_expire_flag=bool(admission['HOSPITAL_EXPIRE_FLAG']),
            discharge_location=admission['DISCHARGE_LOCATION'],
            diagnoses=diagnoses_list,
            procedures=procedures_list,
            lab_results=lab_results,
            medications=medications,
            has_atrial_fib=has_afib,
            has_diabetes=has_diabetes,
            has_hypertension=has_htn,
            has_chf=has_chf,
            cha2ds2_vasc_score=cha2ds2_vasc
        )
        
        return clinical_data
    
    def _get_diagnoses(self, hadm_id: int) -> List[Dict[str, str]]:
        """Get all diagnoses for an admission"""
        diag_rows = self.diagnoses[self.diagnoses['HADM_ID'] == hadm_id]
        diagnoses = []
        
        for _, row in diag_rows.iterrows():
            icd9_code = row['ICD9_CODE']
            # Lookup description
            desc_row = self.d_icd_diagnoses[self.d_icd_diagnoses['ICD9_CODE'] == icd9_code]
            description = desc_row.iloc[0]['SHORT_TITLE'] if not desc_row.empty else 'Unknown'
            
            diagnoses.append({
                'icd9_code': icd9_code,
                'description': description,
                'sequence': row['SEQ_NUM']
            })
        
        return diagnoses
    
    def _get_procedures(self, hadm_id: int) -> List[Dict[str, str]]:
        """Get all procedures for an admission"""
        proc_rows = self.procedures[self.procedures['HADM_ID'] == hadm_id]
        procedures = []
        
        for _, row in proc_rows.iterrows():
            icd9_code = row['ICD9_CODE']
            desc_row = self.d_icd_procedures[self.d_icd_procedures['ICD9_CODE'] == icd9_code]
            description = desc_row.iloc[0]['SHORT_TITLE'] if not desc_row.empty else 'Unknown'
            
            procedures.append({
                'icd9_code': icd9_code,
                'description': description
            })
        
        return procedures
    
    def _get_lab_results(self, hadm_id: int) -> Dict[str, float]:
        """Get relevant lab results for stroke risk assessment"""
        labs = {}
        
        if self.labevents.empty:
            return labs
        
        lab_rows = self.labevents[self.labevents['HADM_ID'] == hadm_id]
        
        # Map lab item IDs to clinical names
        lab_mapping = {
            50809: 'glucose',
            50912: 'creatinine',
            51006: 'bun',
            50902: 'chloride',
            50882: 'bicarbonate',
            51221: 'hematocrit',
            51222: 'hemoglobin',
            51265: 'platelet',
            51301: 'wbc'
        }
        
        for itemid, lab_name in lab_mapping.items():
            lab_data = lab_rows[lab_rows['ITEMID'] == itemid]
            if not lab_data.empty:
                # Use mean value if multiple measurements
                labs[lab_name] = lab_data['VALUENUM'].mean()
        
        return labs
    
    def _get_medications(self, hadm_id: int) -> List[str]:
        """Get list of medications"""
        if self.prescriptions.empty:
            return []
        
        med_rows = self.prescriptions[self.prescriptions['HADM_ID'] == hadm_id]
        return med_rows['DRUG'].unique().tolist()
    
    def _check_diagnosis_code(self, diagnoses: List[Dict], code_prefixes: List[str]) -> bool:
        """Check if any diagnosis matches the given ICD-9 code prefixes"""
        for diag in diagnoses:
            for prefix in code_prefixes:
                if diag['icd9_code'].startswith(prefix):
                    return True
        return False
    
    def _calculate_cha2ds2_vasc(self, age: int, gender: str, has_chf: bool, 
                                 has_htn: bool, has_diabetes: bool,
                                 has_stroke_tia: bool, has_vascular_disease: bool) -> int:
        """
        Calculate CHA2DS2-VASc score for stroke risk
        
        Score components:
        - CHF: 1 point
        - Hypertension: 1 point
        - Age ≥75: 2 points
        - Diabetes: 1 point
        - Stroke/TIA history: 2 points
        - Vascular disease: 1 point
        - Age 65-74: 1 point
        - Female gender: 1 point
        """
        score = 0
        
        if has_chf:
            score += 1
        if has_htn:
            score += 1
        if age >= 75:
            score += 2
        elif age >= 65:
            score += 1
        if has_diabetes:
            score += 1
        if has_stroke_tia:
            score += 2
        if has_vascular_disease:
            score += 1
        if gender == 'F':
            score += 1
        
        return score


# ============================================================================
# PART 3: INTEGRATED PATIENT DATA (Waveforms + Clinical)
# ============================================================================

@dataclass
class IntegratedPatientData:
    """Complete patient data combining waveforms and clinical information"""
    subject_id: str
    waveform_records: List[WaveformRecord]
    clinical_data: Optional[ClinicalData]
    
    def has_complete_data(self) -> bool:
        """Check if patient has both waveform and clinical data"""
        has_waveforms = len(self.waveform_records) > 0
        has_clinical = self.clinical_data is not None
        has_ecg_ppg = any(r.has_ecg_ppg_pair() for r in self.waveform_records)
        return has_waveforms and has_clinical and has_ecg_ppg
    
    def get_stroke_risk_score(self) -> float:
        """
        Calculate annual stroke risk percentage based on CHA2DS2-VASc score
        
        CHA2DS2-VASc → Annual stroke risk:
        0 → 0.2%
        1 → 0.6%
        2 → 2.2%
        3 → 3.2%
        4 → 4.8%
        5 → 7.2%
        6 → 9.7%
        7 → 11.2%
        8 → 10.8%
        9 → 12.2%
        """
        if not self.clinical_data:
            return 0.0
        
        score = self.clinical_data.cha2ds2_vasc_score
        risk_mapping = {
            0: 0.2, 1: 0.6, 2: 2.2, 3: 3.2, 4: 4.8,
            5: 7.2, 6: 9.7, 7: 11.2, 8: 10.8, 9: 12.2
        }
        
        return risk_mapping.get(score, 12.2)  # Cap at 12.2%


class MIMICIntegratedLoader:
    """
    Integrated loader that combines waveform and clinical data
    """
    
    def __init__(self, waveform_root: Path, clinical_root: Path):
        self.waveform_loader = MIMICWaveformLoader(waveform_root)
        self.clinical_loader = MIMICClinicalLoader(clinical_root)
        
        logger.info("MIMIC-III Integrated Loader initialized")
    
    def load_patient(self, subject_id: str) -> IntegratedPatientData:
        """Load complete patient data (waveforms + clinical)"""
        
        logger.info(f"Loading patient {subject_id}...")
        
        # Load waveforms
        waveform_records = self.waveform_loader.load_patient_records(subject_id)
        
        # Load clinical data
        clinical_data = self.clinical_loader.get_patient_clinical_data(subject_id)
        
        # Create integrated data
        patient_data = IntegratedPatientData(
            subject_id=subject_id,
            waveform_records=waveform_records,
            clinical_data=clinical_data
        )
        
        logger.info(f"Patient {subject_id}: {len(waveform_records)} waveforms, "
                   f"clinical={'available' if clinical_data else 'missing'}")
        
        return patient_data
    
    def scan_all_patients(self) -> List[str]:
        """Get list of all available patient IDs"""
        # Get unique subject IDs from waveform directory structure
        subject_ids = set()
        
        for pdir in self.waveform_loader.waveform_root.glob("p*/p*/"):
            subject_id = pdir.name
            subject_ids.add(subject_id)
        
        logger.info(f"Found {len(subject_ids)} unique patients in waveform database")
        return sorted(list(subject_ids))
    
    def create_training_dataset(self, min_quality: float = 0.5, 
                                require_clinical: bool = True) -> List[IntegratedPatientData]:
        """
        Create training dataset with quality filtering
        
        Args:
            min_quality: Minimum signal quality threshold (0.0-1.0)
            require_clinical: Whether to require clinical data
        
        Returns:
            List of patients with high-quality data
        """
        all_patients = self.scan_all_patients()
        training_data = []
        
        logger.info(f"Creating training dataset from {len(all_patients)} patients...")
        logger.info(f"Filters: min_quality={min_quality}, require_clinical={require_clinical}")
        
        for subject_id in all_patients:
            try:
                patient = self.load_patient(subject_id)
                
                # Filter by quality
                high_quality_records = [
                    r for r in patient.waveform_records 
                    if r.signal_quality >= min_quality and r.has_ecg_ppg_pair()
                ]
                
                if len(high_quality_records) == 0:
                    continue
                
                # Filter by clinical data requirement
                if require_clinical and patient.clinical_data is None:
                    continue
                
                # Update patient with filtered records
                patient.waveform_records = high_quality_records
                training_data.append(patient)
                
            except Exception as e:
                logger.error(f"Error processing patient {subject_id}: {e}")
                continue
        
        logger.info(f"Training dataset created: {len(training_data)} patients")
        return training_data


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of the MIMIC-III integration system"""
    
    # Configure paths
    waveform_root = Path("/path/to/mimic3wdb-matched/1.0")
    clinical_root = Path("/path/to/mimiciii/1.4")
    
    # Initialize integrated loader
    loader = MIMICIntegratedLoader(waveform_root, clinical_root)
    
    # Example 1: Load single patient
    print("\n" + "="*80)
    print("EXAMPLE 1: Loading single patient")
    print("="*80)
    
    patient = loader.load_patient("p000020")
    
    if patient.has_complete_data():
        print(f"\n✅ Patient {patient.subject_id} - Complete data available")
        print(f"   Waveform records: {len(patient.waveform_records)}")
        
        for i, record in enumerate(patient.waveform_records):
            print(f"\n   Record {i+1}:")
            print(f"     Duration: {record.duration_seconds:.1f} seconds")
            print(f"     ECG: {'✓' if record.ecg_signal is not None else '✗'}")
            print(f"     PPG: {'✓' if record.ppg_signal is not None else '✗'}")
            print(f"     Quality: {record.signal_quality:.2f}")
        
        if patient.clinical_data:
            cd = patient.clinical_data
            print(f"\n   Clinical Data:")
            print(f"     Age: {cd.age} years")
            print(f"     Gender: {cd.gender}")
            print(f"     ICU LOS: {cd.icu_los_days:.1f} days")
            print(f"     Diagnoses: {len(cd.diagnoses)}")
            print(f"     CHA2DS2-VASc: {cd.cha2ds2_vasc_score}")
            print(f"     Annual stroke risk: {patient.get_stroke_risk_score():.1f}%")
            
            print(f"\n   Stroke Risk Factors:")
            print(f"     Atrial Fibrillation: {cd.has_atrial_fib}")
            print(f"     Diabetes: {cd.has_diabetes}")
            print(f"     Hypertension: {cd.has_hypertension}")
            print(f"     CHF: {cd.has_chf}")
    
    # Example 2: Create training dataset
    print("\n" + "="*80)
    print("EXAMPLE 2: Creating training dataset")
    print("="*80)
    
    training_data = loader.create_training_dataset(
        min_quality=0.6,
        require_clinical=True
    )
    
    print(f"\n✅ Training dataset: {len(training_data)} patients")
    
    # Statistics
    total_records = sum(len(p.waveform_records) for p in training_data)
    avg_quality = np.mean([r.signal_quality for p in training_data for r in p.waveform_records])
    
    print(f"   Total waveform records: {total_records}")
    print(f"   Average signal quality: {avg_quality:.2f}")
    
    # Stroke risk distribution
    stroke_risks = [p.get_stroke_risk_score() for p in training_data if p.clinical_data]
    print(f"\n   Stroke Risk Distribution:")
    print(f"     Mean: {np.mean(stroke_risks):.1f}%")
    print(f"     Median: {np.median(stroke_risks):.1f}%")
    print(f"     Range: {np.min(stroke_risks):.1f}% - {np.max(stroke_risks):.1f}%")
    
    # Example 3: Export for pattern discovery pipeline
    print("\n" + "="*80)
    print("EXAMPLE 3: Export for self-supervised learning")
    print("="*80)
    
    export_data = {
        'ecg_signals': [],
        'ppg_signals': [],
        'patient_ids': [],
        'record_ids': [],
        'clinical_metadata': []
    }
    
    for patient in training_data[:10]:  # First 10 patients as example
        for record in patient.waveform_records:
            if record.has_ecg_ppg_pair():
                export_data['ecg_signals'].append(record.ecg_signal)
                export_data['ppg_signals'].append(record.ppg_signal)
                export_data['patient_ids'].append(patient.subject_id)
                export_data['record_ids'].append(record.record_name)
                
                # Add clinical metadata for validation
                if patient.clinical_data:
                    export_data['clinical_metadata'].append({
                        'age': patient.clinical_data.age,
                        'gender': patient.clinical_data.gender,
                        'cha2ds2_vasc': patient.clinical_data.cha2ds2_vasc_score,
                        'stroke_risk': patient.get_stroke_risk_score(),
                        'has_afib': patient.clinical_data.has_atrial_fib,
                        'has_diabetes': patient.clinical_data.has_diabetes,
                        'diagnoses': [d['description'] for d in patient.clinical_data.diagnoses[:5]]
                    })
                else:
                    export_data['clinical_metadata'].append(None)
    
    print(f"✅ Exported {len(export_data['ecg_signals'])} ECG+PPG pairs")
    print(f"   Ready for self-supervised pattern discovery")
    
    # Save to disk
    output_path = Path("mimic_training_data.npz")
    np.savez_compressed(
        output_path,
        ecg_signals=export_data['ecg_signals'],
        ppg_signals=export_data['ppg_signals'],
        patient_ids=export_data['patient_ids'],
        record_ids=export_data['record_ids']
    )
    print(f"   Saved to: {output_path}")
    
    # Save clinical metadata separately
    import json
    with open("mimic_clinical_metadata.json", 'w') as f:
        json.dump(export_data['clinical_metadata'], f, indent=2)
    print(f"   Clinical metadata saved to: mimic_clinical_metadata.json")


if __name__ == "__main__":
    main()